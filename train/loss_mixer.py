import torch
import torch.nn as nn
import torch.nn.functional as F

from utils.geometry import batch_rodrigues


def resolve_supervision_mode(options):
    mode = str(getattr(options, 'supervision_mode', 'fit_or_gt')).strip().lower()
    if mode not in ('gt_only', 'fit_or_gt'):
        raise ValueError(
            f"Unsupported LOSS.SUPERVISION_MODE='{mode}'. "
            "Expected one of: ['gt_only', 'fit_or_gt']"
        )
    return mode


def build_loss_weights(options):
    return {
        'loss_keypoints_2d': float(getattr(options, 'keypoint_2d_loss_weight', 0.0)),
        'loss_keypoints_2d_pixel': float(getattr(options, 'keypoint_2d_pixel_loss_weight', 0.0)),
        'loss_keypoints_3d': float(getattr(options, 'keypoint_3d_loss_weight', 0.0)),
        'loss_regr_pose': float(getattr(options, 'pose_loss_weight', 0.0)),
        'loss_regr_betas': float(getattr(options, 'beta_loss_weight', 0.0)),
        'loss_shape': float(getattr(options, 'shape_loss_weight', 0.0)),
        'loss_depth': float(getattr(options, 'cam_depth_loss_weight', 0.0)),
        'loss_cam': float(getattr(options, 'cam_loss_weight', 0.0)),
        'loss_cam_full': float(getattr(options, 'cam_full_loss_weight', 0.0)),
    }


def compute_cam_full_ramp(step, start_step=0, ramp_steps=0):
    """Linear ramp-up multiplier for cam_full loss in early training.

    Returns a float in [0, 1]:
      - 0 when step < start_step
      - linearly grows from 0 to 1 over the next `ramp_steps`
      - 1 thereafter
    """
    if step is None:
        return 1.0
    step = int(step)
    start = int(start_step or 0)
    ramp = int(ramp_steps or 0)
    if step < start:
        return 0.0
    if ramp <= 0:
        return 1.0
    progress = (step - start) / float(ramp)
    if progress >= 1.0:
        return 1.0
    if progress < 0.0:
        return 0.0
    return float(progress)


class ReFitStyleLossMixer(nn.Module):
    """
    ReFit-style loss module with SPIN/SMPLify-compatible supervision sources.
    """

    def __init__(self, keypoint_2d_beta=1.0):
        """
        Args:
            keypoint_2d_beta: SmoothL1 transition point for `keypoint_2d_loss`,
                in NORMALIZED [-1, 1] coordinate units (1.0 normalized unit =
                img_res/2 = 128 px at img_res 256).

                This is the real knob on how much gradient the 2D term supplies.
                Typical crop-pixel error is ~13.5 px = 0.105 normalized, so:
                  beta=1.0  -> 0.105 < 1.0  -> QUADRATIC branch -> behaves like
                               MSE; d(loss)/d(err) = err/beta, i.e. the gradient
                               fades as the model improves (original SPIN default).
                  beta=0.05 -> 0.105 > 0.05 -> LINEAR branch -> behaves like L1;
                               d(loss)/d(err) = 1, gradient does NOT decay.
                               Measured ~10.8x the gradient of beta=1.0.

                The now-removable `keypoint_2d_pixel_loss` was a roundabout way of
                getting this L1 behaviour: it supervises the SAME quantity in
                crop-pixel units with beta=5 px, which lands in the linear branch
                and additionally picks up the de_pix/de_norm = 128 chain-rule
                factor. Prefer tuning this beta over adding a second 2D loss in a
                different coordinate system.
        """
        super().__init__()
        self.keypoint_2d_beta = float(keypoint_2d_beta)
        self.criterion_keypoints = nn.SmoothL1Loss(reduction='none', beta=self.keypoint_2d_beta)
        self.criterion_regr = nn.MSELoss()
        self.criterion_shape = nn.L1Loss()

    @staticmethod
    def _zero_like_scalar(tensor_ref):
        return torch.zeros((), device=tensor_ref.device, dtype=tensor_ref.dtype)

    @staticmethod
    def _weighted_mean(per_sample_loss, sample_weight=None):
        if sample_weight is None:
            return per_sample_loss.mean()
        sample_weight = sample_weight.to(per_sample_loss.device, dtype=per_sample_loss.dtype).view(-1)
        if sample_weight.numel() != per_sample_loss.numel():
            raise ValueError(
                f'sample_weight length mismatch: weight={sample_weight.numel()} loss={per_sample_loss.numel()}'
            )
        denom = sample_weight.sum().clamp(min=1e-6)
        return (per_sample_loss * sample_weight).sum() / denom

    def keypoint_2d_loss(self, pred_keypoints_2d, gt_keypoints_2d, openpose_weight, gt_weight):
        conf = gt_keypoints_2d[:, :, -1].unsqueeze(-1).clone()
        conf[:, :25] *= float(openpose_weight)
        conf[:, 25:] *= float(gt_weight)
        mse = self.criterion_keypoints(pred_keypoints_2d, gt_keypoints_2d[:, :, :-1])
        return (conf * mse).mean()

    def keypoint_2d_pixel_loss(self, pred_crop_pix, gt_crop_pix, gt_conf,
                                openpose_weight=1.0, gt_weight=1.0, beta=5.0,
                                sample_weight=None):
        """2D keypoint loss in crop-pixel space (not normalized [-1, 1]).

        Plan §6 Root Cause 3 + §10.1: 2D supervision should live in the same
        coordinate system used at evaluation (crop pixels), otherwise gradients
        train the wrong reprojection norm. Uses SmoothL1 with a larger `beta`
        (default 5 pixels) so the loss is robust to keypoint annotation noise.

        Args:
            pred_crop_pix: [B, J, 2] predicted 2D joints in crop pixels (origin top-left)
            gt_crop_pix:   [B, J, 2] GT 2D joints in crop pixels
            gt_conf:       [B, J] or [B, J, 1] confidence in [0, 1]
            openpose_weight, gt_weight: scale factors applied to the first 25
                (OpenPose-style) and the remaining (GT) joints respectively
            beta: SmoothL1 transition point in pixels
            sample_weight: optional [B] tensor to reweight per sample (pGT quality, etc.)
        Returns scalar loss tensor.
        """
        if gt_conf.dim() == 2:
            conf = gt_conf.unsqueeze(-1).clone()
        else:
            conf = gt_conf.clone()
        n_joints = conf.shape[1]
        n_op = min(25, n_joints)
        conf[:, :n_op] = conf[:, :n_op] * float(openpose_weight)
        if n_joints > n_op:
            conf[:, n_op:] = conf[:, n_op:] * float(gt_weight)

        diff = pred_crop_pix - gt_crop_pix
        abs_d = diff.abs()
        smooth = torch.where(
            abs_d < float(beta),
            0.5 * diff ** 2 / float(beta),
            abs_d - 0.5 * float(beta),
        )
        per_joint = (conf * smooth).mean(dim=(1, 2))  # [B]
        return self._weighted_mean(per_joint, sample_weight)

    def keypoint_3d_loss(self, pred_joints, gt_keypoints_3d, has_pose_3d):
        pred_keypoints_3d = pred_joints[:, 25:, :]
        conf = gt_keypoints_3d[:, :, -1].unsqueeze(-1).clone()
        gt_keypoints_3d = gt_keypoints_3d[:, :, :-1].clone()
        valid_mask = has_pose_3d.bool()

        gt_keypoints_3d = gt_keypoints_3d[valid_mask]
        conf = conf[valid_mask]
        pred_keypoints_3d = pred_keypoints_3d[valid_mask]

        if len(gt_keypoints_3d) == 0:
            return self._zero_like_scalar(pred_joints)

        gt_pelvis = (gt_keypoints_3d[:, 2, :] + gt_keypoints_3d[:, 3, :]) / 2.0
        pred_pelvis = (pred_keypoints_3d[:, 2, :] + pred_keypoints_3d[:, 3, :]) / 2.0
        gt_keypoints_3d = gt_keypoints_3d - gt_pelvis[:, None, :]
        pred_keypoints_3d = pred_keypoints_3d - pred_pelvis[:, None, :]

        mse = self.criterion_keypoints(pred_keypoints_3d, gt_keypoints_3d)
        return (conf * mse).mean()

    def smpl_pose_beta_loss(self, pred_rotmat, pred_betas, target_pose, target_betas, supervision_mask,
                            beta_loss_filter_enabled=False, beta_loss_threshold=2.0,
                            pose_weight=None, beta_weight=None):
        """Compute SMPL pose (rotmat MSE) and beta (per-sample MSE) regression losses.

        When `beta_loss_filter_enabled` is True, samples whose per-sample beta MSE
        exceeds `beta_loss_threshold` are dropped from the beta loss mean (pose loss
        is unaffected). Returns (loss_regr_pose, loss_regr_betas, beta_filter_ratio).
        With filter disabled, behavior is mathematically equivalent to the old
        `nn.MSELoss` over all valid (B*10) elements.
        """
        valid_mask = supervision_mask.bool()
        if valid_mask.sum().item() == 0:
            zero = self._zero_like_scalar(pred_betas)
            return zero, zero, zero

        pred_rotmat_valid = pred_rotmat[valid_mask]
        pred_betas_valid = pred_betas[valid_mask]
        gt_rotmat_valid = batch_rodrigues(target_pose.view(-1, 3)).view(-1, 24, 3, 3)[valid_mask]
        gt_betas_valid = target_betas[valid_mask]

        per_sample_pose = ((pred_rotmat_valid - gt_rotmat_valid) ** 2).mean(dim=(1, 2, 3))
        pose_weight_valid = pose_weight[valid_mask] if pose_weight is not None else None
        loss_regr_pose = self._weighted_mean(per_sample_pose, pose_weight_valid)

        per_sample_betas = ((pred_betas_valid - gt_betas_valid) ** 2).mean(dim=-1)  # [N_valid]
        beta_weight_valid = beta_weight[valid_mask] if beta_weight is not None else None

        if beta_loss_filter_enabled and beta_loss_threshold is not None and beta_loss_threshold > 0:
            keep = per_sample_betas < float(beta_loss_threshold)
            n_total = max(per_sample_betas.numel(), 1)
            filter_ratio = (~keep).sum().to(per_sample_betas.dtype) / float(n_total)
            if keep.sum().item() == 0:
                loss_regr_betas = self._zero_like_scalar(pred_betas)
            else:
                bw_keep = beta_weight_valid[keep] if beta_weight_valid is not None else None
                loss_regr_betas = self._weighted_mean(per_sample_betas[keep], bw_keep)
        else:
            loss_regr_betas = self._weighted_mean(per_sample_betas, beta_weight_valid)
            filter_ratio = self._zero_like_scalar(pred_betas)

        return loss_regr_pose, loss_regr_betas, filter_ratio

    def vertex_loss(
        self,
        pred_vertices,
        supervision_mode,
        has_smpl,
        valid_fit,
        smpl_model,
        gt_pose,
        gt_betas,
        opt_vertices,
        gt_vertices=None,
        sample_weight=None,
    ):
        if supervision_mode == 'gt_only':
            valid_mask = has_smpl.bool()
            if valid_mask.sum().item() == 0:
                return self._zero_like_scalar(pred_vertices)
            if gt_vertices is not None:
                target_vertices = gt_vertices
            else:
                gt_out = smpl_model(
                    betas=gt_betas,
                    body_pose=gt_pose[:, 3:],
                    global_orient=gt_pose[:, :3],
                )
                target_vertices = gt_out.vertices
        elif supervision_mode == 'fit_or_gt':
            if opt_vertices is None or valid_fit is None:
                raise ValueError('fit_or_gt mode requires opt_vertices and valid_fit.')
            valid_mask = valid_fit.bool()
            if valid_mask.sum().item() == 0:
                return self._zero_like_scalar(pred_vertices)
            target_vertices = opt_vertices
        else:
            raise ValueError(f'Unsupported supervision mode: {supervision_mode}')

        pred_vertices_valid = pred_vertices[valid_mask]
        target_vertices_valid = target_vertices[valid_mask]
        per_sample = torch.abs(pred_vertices_valid - target_vertices_valid).mean(dim=(1, 2))
        sw_valid = sample_weight[valid_mask] if sample_weight is not None else None
        return self._weighted_mean(per_sample, sw_valid)

    def cam_depth_loss(self, pred_cam_full, min_depth=1.0, mask=None):
        """Penalize full-image camera Tz that is too small.
        [M4-has-cam-intrinsics] mask: optional bool tensor [B]. If provided, only compute
        loss on rows where mask=True. Returns 0 if no valid rows.
        """
        if mask is not None:
            if not mask.any():
                return self._zero_like_scalar(pred_cam_full)
            pred_cam_full = pred_cam_full[mask]
        tz = pred_cam_full[:, 2]
        return F.relu(min_depth - tz).pow(2).mean()

    def cam_loss(self, pred_camera, cam=None, has_cam=None, mask=None):
        """[M4-has-cam-intrinsics] mask: optional bool tensor [B], intersected with has_cam."""
        if cam is None or has_cam is None:
            return self._zero_like_scalar(pred_camera)
        valid_mask = has_cam.bool()
        if mask is not None:
            valid_mask = valid_mask & mask
        if valid_mask.sum().item() == 0:
            return self._zero_like_scalar(pred_camera)
        return F.mse_loss(pred_camera[valid_mask], cam[valid_mask], reduction='mean')

    def cam_full_pair_loss(self, pred_cam_full, target_cam_full, mask, min_z=0.5, sample_weight=None):
        """CLIFF-style paired supervision in full-image perspective space.
        xy: L1.  z: log-space L1 (proportional to relative depth error).
        Returns zero scalar if mask is all-False.
        """
        if not mask.any():
            return self._zero_like_scalar(pred_cam_full)
        pred = pred_cam_full[mask]
        tgt = target_cam_full[mask].detach()
        loss_xy = torch.abs(pred[:, :2] - tgt[:, :2]).mean(dim=-1)
        pred_z = pred[:, 2].clamp(min=min_z)
        tgt_z = tgt[:, 2].clamp(min=min_z)
        loss_z = (torch.log(pred_z) - torch.log(tgt_z)).abs()
        per_sample = loss_xy + loss_z
        sw_valid = sample_weight[mask] if sample_weight is not None else None
        return self._weighted_mean(per_sample, sw_valid)

    @staticmethod
    def mix_losses(raw_losses, loss_weights):
        weighted_losses = {}
        total_loss = None
        for loss_name, raw_val in raw_losses.items():
            weight = float(loss_weights.get(loss_name, 0.0))
            weighted_val = raw_val * weight
            weighted_losses[f'weighted_{loss_name}'] = weighted_val
            if total_loss is None:
                total_loss = weighted_val
            else:
                total_loss = total_loss + weighted_val
        return total_loss, weighted_losses
