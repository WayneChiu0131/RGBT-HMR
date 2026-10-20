"""v4 initialization-anchored SMPL fitting for thermal pseudo-GT generation.

Changes over thermal_camsmplify.py:
  - lambda_bbox: optional bounding-box center alignment loss during optimization.
  - ThermalCamSMPLify_v4 is a drop-in replacement; old ThermalCamSMPLify is unchanged.
"""

from __future__ import annotations

from typing import Dict, Optional

import torch

import config
from models.smpl import SMPL
from smplify.losses import gmof
from utils.geometry import perspective_projection, project_points_full_to_crop


class ThermalCamSMPLify(object):
    """Lightweight CameraHMR-style staged SMPL fitting for sparse thermal 2D KP.

    The public interface is batched. Internally each sample is optimized
    independently so a numerical failure does not invalidate the whole batch.
    """

    def __init__(
        self,
        batch_size: int = 32,
        focal_length: float = 5000.0,
        device: torch.device = torch.device("cuda"),
        img_res: int = 224,
        phase1_iters: int = 50,
        phase2_iters: int = 50,
        phase3_iters: int = 100,
        lr: float = 1e-3,
        lambda_2d: float = 1.0,
        lambda_pose_init: float = 50.0,
        lambda_beta_init_phase2: float = 0.01,
        lambda_beta_init_phase3: float = 30.0,
        lambda_verts_init_phase3: float = 10.0,
        lambda_angle: float = 2.0,
        gmof_sigma: float = 100.0,
        lambda_shape_l2: float = 0.0,
        lambda_bbox: float = 0.0,
        truncated_anchor_multiplier: float = 2.0,
        fullcam: bool = False,
    ) -> None:
        self.device = device
        self.focal_length = float(focal_length)
        self.img_res = int(img_res)
        self.phase1_iters = int(phase1_iters)
        self.phase2_iters = int(phase2_iters)
        self.phase3_iters = int(phase3_iters)
        self.lr = float(lr)
        self.lambda_2d = float(lambda_2d)
        self.lambda_pose_init = float(lambda_pose_init)
        self.lambda_beta_init_phase2 = float(lambda_beta_init_phase2)
        self.lambda_beta_init_phase3 = float(lambda_beta_init_phase3)
        self.lambda_verts_init_phase3 = float(lambda_verts_init_phase3)
        self.lambda_angle = float(lambda_angle)
        self.gmof_sigma = float(gmof_sigma)
        self.lambda_shape_l2 = float(lambda_shape_l2)
        self.lambda_bbox = float(lambda_bbox)
        self.truncated_anchor_multiplier = float(truncated_anchor_multiplier)
        self.fullcam = bool(fullcam)

        self.smpl = SMPL(
            config.SMPL_MODEL_DIR,
            batch_size=batch_size,
            create_transl=False,
        ).to(self.device)

    def __call__(
        self,
        init_pose: torch.Tensor,
        init_betas: torch.Tensor,
        init_cam: torch.Tensor,
        keypoints_crop: torch.Tensor,
        center: torch.Tensor,
        scale: torch.Tensor,
        img_focal: torch.Tensor,
        img_center: torch.Tensor,
        img_res: Optional[int] = None,
        fullcam: Optional[bool] = None,
        freeze_betas: Optional[torch.Tensor] = None,
        strong_anchor: Optional[torch.Tensor] = None,
    ) -> Dict[str, object]:
        if img_res is not None and int(img_res) != self.img_res:
            self.img_res = int(img_res)
        use_fullcam = self.fullcam if fullcam is None else bool(fullcam)

        outputs = []
        for idx in range(init_pose.shape[0]):
            freeze_one = self._flag_at(freeze_betas, idx)
            strong_one = self._flag_at(strong_anchor, idx)
            outputs.append(
                self._fit_one(
                    init_pose[idx : idx + 1],
                    init_betas[idx : idx + 1],
                    init_cam[idx : idx + 1],
                    keypoints_crop[idx : idx + 1],
                    center[idx : idx + 1],
                    scale[idx : idx + 1],
                    img_focal[idx : idx + 1],
                    img_center[idx : idx + 1],
                    use_fullcam,
                    freeze_one,
                    strong_one,
                )
            )

        tensor_keys = ("opt_pose", "opt_betas", "opt_cam", "vertices", "joints")
        result: Dict[str, object] = {
            key: torch.cat([out[key] for out in outputs], dim=0) for key in tensor_keys
        }
        result["valid_mask"] = torch.cat([out["valid_mask"] for out in outputs], dim=0)

        loss_keys = (
            "loss_total",
            "loss_2d",
            "loss_pose_init",
            "loss_beta_init",
            "loss_verts_init",
            "loss_angle",
            "verts_init_dist",
        )
        result["loss_dict"] = {
            key: torch.cat([out["loss_dict"][key] for out in outputs], dim=0) for key in loss_keys
        }
        return result

    def _fit_one(
        self,
        init_pose: torch.Tensor,
        init_betas: torch.Tensor,
        init_cam: torch.Tensor,
        keypoints_crop: torch.Tensor,
        center: torch.Tensor,
        scale: torch.Tensor,
        img_focal: torch.Tensor,
        img_center: torch.Tensor,
        fullcam: bool,
        freeze_betas: bool,
        strong_anchor: bool,
    ) -> Dict[str, object]:
        init_pose = init_pose.detach().to(self.device)
        init_betas = init_betas.detach().to(self.device)
        init_cam = init_cam.detach().to(self.device)
        keypoints_crop = keypoints_crop.detach().to(self.device)
        center = center.detach().to(self.device)
        scale = scale.detach().to(self.device)
        img_focal = img_focal.detach().to(self.device)
        img_center = img_center.detach().to(self.device)

        fallback = self._fallback_result(init_pose, init_betas, init_cam)
        if not (
            torch.isfinite(init_pose).all()
            and torch.isfinite(init_betas).all()
            and torch.isfinite(init_cam).all()
            and torch.isfinite(keypoints_crop).all()
        ):
            return fallback
        if float((keypoints_crop[:, :, 2] > 0).sum().item()) <= 0:
            return fallback

        init_global_orient = init_pose[:, :3].clone()
        init_body_pose = init_pose[:, 3:].clone()
        init_betas_anchor = init_betas.clone()

        with torch.no_grad():
            init_smpl = self.smpl(
                global_orient=init_global_orient,
                body_pose=init_body_pose,
                betas=init_betas_anchor,
            )
            init_vertices = init_smpl.vertices.detach()

        global_orient = init_global_orient.clone()
        body_pose = init_body_pose.clone()
        betas = init_betas_anchor.clone()
        cam = init_cam.clone()

        try:
            phase3_pose_weight = self.lambda_pose_init
            phase3_verts_weight = self.lambda_verts_init_phase3
            if strong_anchor:
                phase3_pose_weight *= self.truncated_anchor_multiplier
                phase3_verts_weight *= self.truncated_anchor_multiplier

            ok = self._run_phase(
                global_orient,
                body_pose,
                betas,
                cam,
                keypoints_crop,
                center,
                scale,
                img_focal,
                img_center,
                init_body_pose,
                init_betas_anchor,
                init_vertices,
                fullcam,
                iters=self.phase1_iters,
                optimize_global=False,
                optimize_body=False,
                optimize_betas=False,
                optimize_cam=True,
                lambda_pose_init=0.0,
                lambda_beta_init=0.0,
                lambda_verts_init=0.0,
                lambda_angle=0.0,
            )
            if not ok:
                return fallback

            ok = self._run_phase(
                global_orient,
                body_pose,
                betas,
                cam,
                keypoints_crop,
                center,
                scale,
                img_focal,
                img_center,
                init_body_pose,
                init_betas_anchor,
                init_vertices,
                fullcam,
                iters=self.phase2_iters,
                optimize_global=True,
                optimize_body=False,
                optimize_betas=not freeze_betas,
                optimize_cam=True,
                lambda_pose_init=0.0,
                lambda_beta_init=self.lambda_beta_init_phase2,
                lambda_verts_init=0.0,
                lambda_angle=0.0,
            )
            if not ok:
                return fallback

            ok = self._run_phase(
                global_orient,
                body_pose,
                betas,
                cam,
                keypoints_crop,
                center,
                scale,
                img_focal,
                img_center,
                init_body_pose,
                init_betas_anchor,
                init_vertices,
                fullcam,
                iters=self.phase3_iters,
                optimize_global=True,
                optimize_body=True,
                optimize_betas=not freeze_betas,
                optimize_cam=True,
                lambda_pose_init=phase3_pose_weight,
                lambda_beta_init=self.lambda_beta_init_phase3,
                lambda_verts_init=phase3_verts_weight,
                lambda_angle=self.lambda_angle,
            )
            if not ok:
                return fallback

            with torch.no_grad():
                terms = self._loss_terms(
                    global_orient,
                    body_pose,
                    betas,
                    cam,
                    keypoints_crop,
                    center,
                    scale,
                    img_focal,
                    img_center,
                    init_body_pose,
                    init_betas_anchor,
                    init_vertices,
                    fullcam,
                    phase3_pose_weight,
                    self.lambda_beta_init_phase3,
                    phase3_verts_weight,
                    self.lambda_angle,
                )
                if not torch.isfinite(terms["loss_total"]).all():
                    return fallback
                pose = torch.cat([global_orient, body_pose], dim=-1).detach()
                return {
                    "opt_pose": pose,
                    "opt_betas": betas.detach(),
                    "opt_cam": cam.detach(),
                    "vertices": terms["vertices"].detach(),
                    "joints": terms["joints"].detach(),
                    "valid_mask": torch.ones(1, dtype=torch.bool, device=self.device),
                    "loss_dict": self._detach_loss_dict(terms),
                }
        except Exception:
            return fallback

    def _run_phase(
        self,
        global_orient: torch.Tensor,
        body_pose: torch.Tensor,
        betas: torch.Tensor,
        cam: torch.Tensor,
        keypoints_crop: torch.Tensor,
        center: torch.Tensor,
        scale: torch.Tensor,
        img_focal: torch.Tensor,
        img_center: torch.Tensor,
        init_body_pose: torch.Tensor,
        init_betas: torch.Tensor,
        init_vertices: torch.Tensor,
        fullcam: bool,
        iters: int,
        optimize_global: bool,
        optimize_body: bool,
        optimize_betas: bool,
        optimize_cam: bool,
        lambda_pose_init: float,
        lambda_beta_init: float,
        lambda_verts_init: float,
        lambda_angle: float,
    ) -> bool:
        if iters <= 0:
            return True

        params = []
        self._set_requires_grad(global_orient, optimize_global, params)
        self._set_requires_grad(body_pose, optimize_body, params)
        self._set_requires_grad(betas, optimize_betas, params)
        self._set_requires_grad(cam, optimize_cam, params)
        if not params:
            return True

        optimizer = torch.optim.Adam(params, lr=self.lr, betas=(0.9, 0.999))
        for _ in range(iters):
            terms = self._loss_terms(
                global_orient,
                body_pose,
                betas,
                cam,
                keypoints_crop,
                center,
                scale,
                img_focal,
                img_center,
                init_body_pose,
                init_betas,
                init_vertices,
                fullcam,
                lambda_pose_init,
                lambda_beta_init,
                lambda_verts_init,
                lambda_angle,
            )
            loss = terms["loss_total"].sum()
            if not torch.isfinite(loss):
                return False
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(params, max_norm=1.0)
            optimizer.step()
            self._clamp_camera_depth(cam)
        return True

    def _loss_terms(
        self,
        global_orient: torch.Tensor,
        body_pose: torch.Tensor,
        betas: torch.Tensor,
        cam: torch.Tensor,
        keypoints_crop: torch.Tensor,
        center: torch.Tensor,
        scale: torch.Tensor,
        img_focal: torch.Tensor,
        img_center: torch.Tensor,
        init_body_pose: torch.Tensor,
        init_betas: torch.Tensor,
        init_vertices: torch.Tensor,
        fullcam: bool,
        lambda_pose_init: float,
        lambda_beta_init: float,
        lambda_verts_init: float,
        lambda_angle: float,
    ) -> Dict[str, torch.Tensor]:
        smpl_output = self.smpl(global_orient=global_orient, body_pose=body_pose, betas=betas)
        joints = smpl_output.joints
        vertices = smpl_output.vertices
        projected = self._project_joints(joints, cam, center, scale, img_focal, img_center, fullcam)

        conf = keypoints_crop[:, :, 2].clamp(min=0.0)
        reprojection_error = gmof(projected - keypoints_crop[:, :, :2], self.gmof_sigma).sum(dim=-1)
        denom = conf.sum(dim=-1).clamp(min=1.0)
        loss_2d = self.lambda_2d * ((conf ** 2) * reprojection_error).sum(dim=-1) / denom

        loss_pose_init = float(lambda_pose_init) * ((body_pose - init_body_pose) ** 2).mean(dim=-1)
        loss_beta_init = float(lambda_beta_init) * ((betas - init_betas) ** 2).mean(dim=-1)
        verts_dist = torch.norm(vertices - init_vertices, dim=-1).mean(dim=-1)
        loss_verts_init = float(lambda_verts_init) * verts_dist
        full_pose = torch.cat([global_orient, body_pose], dim=-1)
        loss_angle = float(lambda_angle) * self._angle_prior_full_pose(full_pose).sum(dim=-1)
        loss_shape_l2 = self.lambda_shape_l2 * (betas ** 2).mean(dim=-1)

        # bbox center alignment: penalize drift of projected mesh center vs annotated kp center
        if self.lambda_bbox > 0:
            vis = conf.clamp(min=0.0)  # [B, J]
            vis_denom = vis.sum(dim=1, keepdim=True).clamp(min=1.0)
            anno_center = (keypoints_crop[:, :, :2] * vis.unsqueeze(-1)).sum(dim=1) / vis_denom  # [B, 2]
            proj_center = (projected * vis.unsqueeze(-1)).sum(dim=1) / vis_denom  # [B, 2]
            loss_bbox = self.lambda_bbox * ((proj_center - anno_center) ** 2).sum(dim=-1)
        else:
            loss_bbox = torch.zeros_like(loss_2d)

        loss_total = loss_2d + loss_pose_init + loss_beta_init + loss_verts_init + loss_angle + loss_shape_l2 + loss_bbox

        return {
            "loss_total": loss_total,
            "loss_2d": loss_2d,
            "loss_pose_init": loss_pose_init,
            "loss_beta_init": loss_beta_init,
            "loss_verts_init": loss_verts_init,
            "loss_angle": loss_angle,
            "verts_init_dist": verts_dist,
            "vertices": vertices,
            "joints": joints,
        }

    def _project_joints(
        self,
        joints: torch.Tensor,
        cam: torch.Tensor,
        center: torch.Tensor,
        scale: torch.Tensor,
        img_focal: torch.Tensor,
        img_center: torch.Tensor,
        fullcam: bool,
    ) -> torch.Tensor:
        batch_size = joints.shape[0]
        rotation = torch.eye(3, device=joints.device, dtype=joints.dtype).unsqueeze(0).expand(batch_size, -1, -1)
        if fullcam:
            _, proj_crop, _ = project_points_full_to_crop(
                joints,
                cam,
                center=center,
                scale=scale,
                img_focal=img_focal,
                img_center=img_center,
                crop_res=self.img_res,
                rotation=rotation,
            )
            return proj_crop

        camera_center = torch.full((batch_size, 2), self.img_res / 2.0, device=joints.device, dtype=joints.dtype)
        return perspective_projection(
            joints,
            rotation=rotation,
            translation=cam,
            focal_length=self.focal_length,
            camera_center=camera_center,
        )

    def _fallback_result(self, init_pose: torch.Tensor, init_betas: torch.Tensor, init_cam: torch.Tensor) -> Dict[str, object]:
        with torch.no_grad():
            smpl_output = self.smpl(
                global_orient=init_pose[:, :3],
                body_pose=init_pose[:, 3:],
                betas=init_betas,
            )
        inf = torch.full((init_pose.shape[0],), float("inf"), dtype=init_pose.dtype, device=self.device)
        zero = torch.zeros((init_pose.shape[0],), dtype=init_pose.dtype, device=self.device)
        return {
            "opt_pose": init_pose.detach(),
            "opt_betas": init_betas.detach(),
            "opt_cam": init_cam.detach(),
            "vertices": smpl_output.vertices.detach(),
            "joints": smpl_output.joints.detach(),
            "valid_mask": torch.zeros(init_pose.shape[0], dtype=torch.bool, device=self.device),
            "loss_dict": {
                "loss_total": inf,
                "loss_2d": inf,
                "loss_pose_init": zero,
                "loss_beta_init": zero,
                "loss_verts_init": zero,
                "loss_angle": zero,
                "verts_init_dist": zero,
            },
        }

    @staticmethod
    def _set_requires_grad(tensor: torch.Tensor, enabled: bool, params) -> None:
        tensor.requires_grad_(enabled)
        if enabled:
            params.append(tensor)

    @staticmethod
    def _clamp_camera_depth(cam: torch.Tensor) -> None:
        with torch.no_grad():
            cam[:, 2].clamp_(min=1e-3, max=1e3)

    @staticmethod
    def _flag_at(value: Optional[torch.Tensor], idx: int) -> bool:
        if value is None:
            return False
        if isinstance(value, bool):
            return bool(value)
        return bool(value.detach().view(-1)[idx].item())

    @staticmethod
    def _angle_prior_full_pose(full_pose: torch.Tensor) -> torch.Tensor:
        weights = torch.tensor([1.0, -1.0, -1.0, -1.0], device=full_pose.device, dtype=full_pose.dtype)
        return torch.exp(full_pose[:, [55, 58, 12, 15]] * weights) ** 2

    @staticmethod
    def _detach_loss_dict(terms: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        keys = (
            "loss_total",
            "loss_2d",
            "loss_pose_init",
            "loss_beta_init",
            "loss_verts_init",
            "loss_angle",
            "verts_init_dist",
        )
        return {key: terms[key].detach() for key in keys}
