import torch

from models.smpl import SMPL
from .losses import camera_fitting_loss_fullcam, body_fitting_loss_fullcam

import config
import constants
from .prior import MaxMixturePrior


class SMPLify():
    """SMPLify with full-image perspective camera translation [Tx, Ty, Tz]
    as the primary camera optimization variable (CLIFF-style)."""

    def __init__(self,
                 step_size=1e-2,
                 batch_size=66,
                 num_iters=100,
                 focal_length=5000,
                 device=torch.device('cuda'),
                 camera_num_iters=None,
                 body_num_iters=None,
                 pose_prior_weight=4.78,
                 shape_prior_weight=5.0,
                 angle_prior_weight=15.2,
                 sigma=100.0,
                 init_pose_reg_weight=0.05,
                 init_betas_reg_weight=0.01,
                 init_global_orient_reg_weight=0.05,
                 prior_folder=None,
                 crop_res=224,
                 **kwargs):
        _ = kwargs

        self.device = device
        self.focal_length = float(focal_length)
        self.step_size = float(step_size)
        self.num_iters = int(num_iters)
        self.camera_num_iters = int(camera_num_iters) if camera_num_iters is not None else self.num_iters
        self.body_num_iters = int(body_num_iters) if body_num_iters is not None else self.num_iters
        self.crop_res = int(crop_res)

        self.pose_prior_weight = float(pose_prior_weight)
        self.shape_prior_weight = float(shape_prior_weight)
        self.angle_prior_weight = float(angle_prior_weight)
        self.sigma = float(sigma)

        self.init_pose_reg_weight = float(init_pose_reg_weight)
        self.init_betas_reg_weight = float(init_betas_reg_weight)
        self.init_global_orient_reg_weight = float(init_global_orient_reg_weight)

        ign_joints = ['OP Neck', 'OP RHip', 'OP LHip', 'Right Hip', 'Left Hip']
        self.ign_joints = [constants.JOINT_IDS[i] for i in ign_joints]

        if prior_folder is None:
            prior_folder = getattr(config, 'SMPLIFY_PRIOR_DIR', 'data')

        self.pose_prior = MaxMixturePrior(
            prior_folder=prior_folder,
            num_gaussians=8,
            dtype=torch.float32,
        ).to(device)

        self.smpl = SMPL(
            config.SMPL_MODEL_DIR,
            batch_size=batch_size,
            create_transl=False,
        ).to(self.device)

    def __call__(self, init_pose, init_betas, init_cam_full, center, scale, img_focal, img_center,
                 keypoints_2d, rot_angle=None, is_flipped=None):
        """
        Inputs:
            init_pose:      [B, 72] axis-angle
            init_betas:     [B, 10]
            init_cam_full:  [B, 3] full-image perspective camera [Tx, Ty, Tz]
            center:         [B, 2] bbox center on full image
            scale:          [B] bbox scale
            img_focal:      [B] focal length
            img_center:     [B, 2] full-image center
            keypoints_2d:   [B, 49, 3] crop pixel coordinates
        Returns:
            vertices, joints, pose, betas, cam_full, reprojection_loss
        """
        joints_2d = keypoints_2d[:, :, :2]
        if keypoints_2d.shape[-1] >= 3:
            joints_conf = keypoints_2d[:, :, -1].clone()
        else:
            joints_conf = torch.ones_like(joints_2d[:, :, 0])

        body_pose = init_pose[:, 3:].detach().clone()
        global_orient = init_pose[:, :3].detach().clone()
        betas = init_betas.detach().clone()
        cam_full = init_cam_full.detach().clone()

        init_body_pose = body_pose.detach().clone()
        init_global_orient = global_orient.detach().clone()
        init_betas_copy = betas.detach().clone()
        init_cam_full_copy = cam_full.detach().clone()

        # Step 1: camera step (optimize global_orient + cam_full)
        body_pose.requires_grad = False
        betas.requires_grad = False
        global_orient.requires_grad = True
        cam_full.requires_grad = True

        camera_optimizer = torch.optim.Adam(
            [global_orient, cam_full],
            lr=self.step_size,
            betas=(0.9, 0.999)
        )

        for _ in range(self.camera_num_iters):
            smpl_output = self.smpl(
                global_orient=global_orient,
                body_pose=body_pose,
                betas=betas
            )
            model_joints = smpl_output.joints

            loss = camera_fitting_loss_fullcam(
                model_joints=model_joints,
                cam_t=cam_full,
                init_cam_t=init_cam_full_copy,
                center=center,
                scale=scale,
                img_focal=img_focal,
                img_center=img_center,
                joints_2d=joints_2d,
                joints_conf=joints_conf,
                crop_res=self.crop_res,
                rot_angle=rot_angle,
                is_flipped=is_flipped,
            )
            if not torch.isfinite(loss):
                break
            camera_optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_([global_orient, cam_full], max_norm=1.0)
            camera_optimizer.step()
            # Prevent depth collapse: Tz → 0 or negative sends projected joints to ∞.
            # Clamp Tz to [0.2 × init_Tz, 5 × init_Tz] to keep optimisation stable.
            with torch.no_grad():
                tz_init = init_cam_full_copy[:, 2:3].abs().clamp(min=1e-2)
                cam_full[:, 2:3] = cam_full[:, 2:3].clamp(
                    min=tz_init * 0.2, max=tz_init * 5.0
                )

        # Step 2: body step (optimize body_pose + betas + global_orient)
        body_pose.requires_grad = True
        betas.requires_grad = True
        global_orient.requires_grad = True
        cam_full.requires_grad = False

        body_optimizer = torch.optim.Adam(
            [body_pose, betas, global_orient],
            lr=self.step_size,
            betas=(0.9, 0.999)
        )

        body_joints_conf = joints_conf.clone()
        body_joints_conf[:, self.ign_joints] = 0.0

        for _ in range(self.body_num_iters):
            smpl_output = self.smpl(
                global_orient=global_orient,
                body_pose=body_pose,
                betas=betas
            )
            model_joints = smpl_output.joints

            loss = body_fitting_loss_fullcam(
                body_pose=body_pose,
                betas=betas,
                model_joints=model_joints,
                cam_t=cam_full,
                center=center,
                scale=scale,
                img_focal=img_focal,
                img_center=img_center,
                joints_2d=joints_2d,
                joints_conf=body_joints_conf,
                pose_prior=self.pose_prior,
                crop_res=self.crop_res,
                sigma=self.sigma,
                pose_prior_weight=self.pose_prior_weight,
                shape_prior_weight=self.shape_prior_weight,
                angle_prior_weight=self.angle_prior_weight,
                init_body_pose=init_body_pose,
                init_betas=init_betas_copy,
                init_global_orient=init_global_orient,
                global_orient=global_orient,
                init_pose_reg_weight=self.init_pose_reg_weight,
                init_betas_reg_weight=self.init_betas_reg_weight,
                init_global_orient_reg_weight=self.init_global_orient_reg_weight,
                rot_angle=rot_angle,
                is_flipped=is_flipped,
                output='sum',
            )
            if not torch.isfinite(loss):
                break
            body_optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_([body_pose, betas, global_orient], max_norm=1.0)
            body_optimizer.step()

        with torch.no_grad():
            smpl_output = self.smpl(
                global_orient=global_orient,
                body_pose=body_pose,
                betas=betas,
                return_full_pose=True
            )
            model_joints = smpl_output.joints
            reprojection_loss = body_fitting_loss_fullcam(
                body_pose=body_pose,
                betas=betas,
                model_joints=model_joints,
                cam_t=cam_full,
                center=center,
                scale=scale,
                img_focal=img_focal,
                img_center=img_center,
                joints_2d=joints_2d,
                joints_conf=body_joints_conf,
                pose_prior=self.pose_prior,
                crop_res=self.crop_res,
                sigma=self.sigma,
                pose_prior_weight=self.pose_prior_weight,
                shape_prior_weight=self.shape_prior_weight,
                angle_prior_weight=self.angle_prior_weight,
                rot_angle=rot_angle,
                is_flipped=is_flipped,
                output='reprojection',
            )

            vertices = smpl_output.vertices.detach()
            joints = smpl_output.joints.detach()
            pose = torch.cat([global_orient, body_pose], dim=-1).detach()
            betas = betas.detach()
            cam_full = cam_full.detach()

        return vertices, joints, pose, betas, cam_full, reprojection_loss

    def get_fitting_loss(self, pose, betas, cam_full, center, scale, img_focal, img_center,
                         keypoints_2d, rot_angle=None, is_flipped=None):
        """Returns reprojection loss matrix [B, 49] (so caller can do mean(dim=-1))."""
        joints_2d = keypoints_2d[:, :, :2]
        if keypoints_2d.shape[-1] >= 3:
            joints_conf = keypoints_2d[:, :, -1].clone()
        else:
            joints_conf = torch.ones_like(joints_2d[:, :, 0])
        joints_conf[:, self.ign_joints] = 0.0

        body_pose = pose[:, 3:]
        global_orient = pose[:, :3]

        with torch.no_grad():
            smpl_output = self.smpl(
                global_orient=global_orient,
                body_pose=body_pose,
                betas=betas,
                return_full_pose=True
            )
            model_joints = smpl_output.joints
            reprojection_loss = body_fitting_loss_fullcam(
                body_pose=body_pose,
                betas=betas,
                model_joints=model_joints,
                cam_t=cam_full,
                center=center,
                scale=scale,
                img_focal=img_focal,
                img_center=img_center,
                joints_2d=joints_2d,
                joints_conf=joints_conf,
                pose_prior=self.pose_prior,
                crop_res=self.crop_res,
                sigma=self.sigma,
                pose_prior_weight=self.pose_prior_weight,
                shape_prior_weight=self.shape_prior_weight,
                angle_prior_weight=self.angle_prior_weight,
                rot_angle=rot_angle,
                is_flipped=is_flipped,
                output='reprojection',
            )
        return reprojection_loss
