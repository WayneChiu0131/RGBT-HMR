import torch
import torch.nn as nn
import numpy as np
import os
import pickle

from utils.geometry import perspective_projection, cam_crop2full, project_points_crop_full, project_points_full_to_crop
import constants


def gmof(x, sigma):
    """
    Geman-McClure error function
    """
    x_squared =  x ** 2
    sigma_squared = sigma ** 2
    return (sigma_squared * x_squared) / (sigma_squared + x_squared)

def angle_prior(pose):
    """
    Angle prior that penalizes unnatural bending of the knees and elbows
    """
    # We subtract 3 because pose does not include the global rotation of the model
    return torch.exp(pose[:, [55-3, 58-3, 12-3, 15-3]] * torch.tensor([1., -1., -1, -1.], device=pose.device)) ** 2

def body_fitting_loss(body_pose, betas, model_joints, camera_t, camera_center,
                      joints_2d, joints_conf, pose_prior,
                      focal_length=5000, sigma=100, pose_prior_weight=4.78,
                      shape_prior_weight=5, angle_prior_weight=15.2,
                      output='sum'):
    """
    Loss function for body fitting
    """

    batch_size = body_pose.shape[0]
    rotation = torch.eye(3, device=body_pose.device).unsqueeze(0).expand(batch_size, -1, -1)
    projected_joints = perspective_projection(model_joints, rotation, camera_t,
                                              focal_length, camera_center)

    # Weighted robust reprojection error
    reprojection_error = gmof(projected_joints - joints_2d, sigma)
    reprojection_loss = (joints_conf ** 2) * reprojection_error.sum(dim=-1)

    # Pose prior loss
    pose_prior_loss = (pose_prior_weight ** 2) * pose_prior(body_pose, betas)

    # Angle prior for knees and elbows
    angle_prior_loss = (angle_prior_weight ** 2) * angle_prior(body_pose).sum(dim=-1)

    # Regularizer to prevent betas from taking large values
    shape_prior_loss = (shape_prior_weight ** 2) * (betas ** 2).sum(dim=-1)

    total_loss = reprojection_loss.sum(dim=-1) + pose_prior_loss + angle_prior_loss + shape_prior_loss

    if output == 'sum':
        return total_loss.sum()
    elif output == 'reprojection':
        return reprojection_loss

def camera_fitting_loss(model_joints, camera_t, camera_t_est, camera_center, joints_2d, joints_conf,
                        focal_length=5000, depth_loss_weight=100):
    """
    Loss function for camera optimization.
    """

    # Project model joints
    batch_size = model_joints.shape[0]
    rotation = torch.eye(3, device=model_joints.device).unsqueeze(0).expand(batch_size, -1, -1)
    projected_joints = perspective_projection(model_joints, rotation, camera_t,
                                              focal_length, camera_center)
    
    op_joints = ['OP RHip', 'OP LHip', 'OP RShoulder', 'OP LShoulder']
    op_joints_ind = [constants.JOINT_IDS[joint] for joint in op_joints]
    gt_joints = ['Right Hip', 'Left Hip', 'Right Shoulder', 'Left Shoulder']
    gt_joints_ind = [constants.JOINT_IDS[joint] for joint in gt_joints]
    reprojection_error_op = (joints_2d[:, op_joints_ind] -
                             projected_joints[:, op_joints_ind]) ** 2
    reprojection_error_gt = (joints_2d[:, gt_joints_ind] -
                             projected_joints[:, gt_joints_ind]) ** 2

    # Check if for each example in the batch all 4 OpenPose detections are valid, otherwise use the GT detections
    # OpenPose joints are more reliable for this task, so we prefer to use them if possible
    is_valid = (joints_conf[:, op_joints_ind].min(dim=-1)[0][:,None,None] > 0).float()
    reprojection_loss = (is_valid * reprojection_error_op + (1-is_valid) * reprojection_error_gt).sum(dim=(1,2))

    # Loss that penalizes deviation from depth estimate
    depth_loss = (depth_loss_weight ** 2) * (camera_t[:, 2] - camera_t_est[:, 2]) ** 2

    total_loss = reprojection_loss + depth_loss
    return total_loss.sum()


def body_fitting_loss_cropcam(body_pose, betas, model_joints, crop_cam, center, scale, img_focal, img_center,
                              joints_2d, joints_conf, pose_prior,
                              crop_res=224, sigma=100, pose_prior_weight=4.78,
                              shape_prior_weight=5, angle_prior_weight=15.2,
                              init_body_pose=None, init_betas=None, init_global_orient=None, global_orient=None,
                              init_pose_reg_weight=0.0, init_betas_reg_weight=0.0, init_global_orient_reg_weight=0.0,
                              rot_angle=None, is_flipped=None,
                              output='sum'):
    """
    Body fitting loss with crop weak-perspective camera projected through full image.
    """
    batch_size = body_pose.shape[0]
    rotation = torch.eye(3, device=body_pose.device).unsqueeze(0).expand(batch_size, -1, -1)
    _, projected_joints_crop, _ = project_points_crop_full(
        model_joints,
        crop_cam,
        center,
        scale,
        img_focal,
        img_center,
        crop_res,
        rotation=rotation,
        rot_angle=rot_angle,
        is_flipped=is_flipped
    )

    reprojection_error = gmof(projected_joints_crop - joints_2d, sigma)
    reprojection_loss = (joints_conf ** 2) * reprojection_error.sum(dim=-1)

    pose_prior_loss = (pose_prior_weight ** 2) * pose_prior(body_pose, betas)
    angle_prior_loss = (angle_prior_weight ** 2) * angle_prior(body_pose).sum(dim=-1)
    shape_prior_loss = (shape_prior_weight ** 2) * (betas ** 2).sum(dim=-1)

    total_loss = reprojection_loss.sum(dim=-1) + pose_prior_loss + angle_prior_loss + shape_prior_loss

    if init_body_pose is not None and init_pose_reg_weight > 0.0:
        total_loss = total_loss + (init_pose_reg_weight ** 2) * ((body_pose - init_body_pose) ** 2).sum(dim=-1)
    if init_betas is not None and init_betas_reg_weight > 0.0:
        total_loss = total_loss + (init_betas_reg_weight ** 2) * ((betas - init_betas) ** 2).sum(dim=-1)
    if global_orient is not None and init_global_orient is not None and init_global_orient_reg_weight > 0.0:
        total_loss = total_loss + (init_global_orient_reg_weight ** 2) * (
            (global_orient - init_global_orient) ** 2
        ).sum(dim=-1)

    if output == 'sum':
        return total_loss.sum()
    elif output == 'reprojection':
        return reprojection_loss


def camera_fitting_loss_cropcam(model_joints, crop_cam, init_crop_cam, center, scale, img_focal, img_center,
                                joints_2d, joints_conf, crop_res=224, depth_loss_weight=100,
                                rot_angle=None, is_flipped=None):
    """
    Camera fitting loss in crop weak-perspective space with full-image depth regularization.
    (Legacy version kept for backward compatibility.)
    """
    batch_size = model_joints.shape[0]
    rotation = torch.eye(3, device=model_joints.device).unsqueeze(0).expand(batch_size, -1, -1)
    _, projected_joints_crop, curr_full_t = project_points_crop_full(
        model_joints,
        crop_cam,
        center,
        scale,
        img_focal,
        img_center,
        crop_res,
        rotation=rotation,
        rot_angle=rot_angle,
        is_flipped=is_flipped
    )
    init_full_t = cam_crop2full(init_crop_cam, center, scale, img_focal, img_center)

    op_joints = ['OP RHip', 'OP LHip', 'OP RShoulder', 'OP LShoulder']
    op_joints_ind = [constants.JOINT_IDS[joint] for joint in op_joints]
    gt_joints = ['Right Hip', 'Left Hip', 'Right Shoulder', 'Left Shoulder']
    gt_joints_ind = [constants.JOINT_IDS[joint] for joint in gt_joints]

    reprojection_error_op = (joints_2d[:, op_joints_ind] - projected_joints_crop[:, op_joints_ind]) ** 2
    reprojection_error_gt = (joints_2d[:, gt_joints_ind] - projected_joints_crop[:, gt_joints_ind]) ** 2
    is_valid = (joints_conf[:, op_joints_ind].min(dim=-1)[0][:, None, None] > 0).float()
    reprojection_loss = (is_valid * reprojection_error_op + (1 - is_valid) * reprojection_error_gt).sum(dim=(1, 2))

    depth_loss = (depth_loss_weight ** 2) * (curr_full_t[:, 2] - init_full_t[:, 2]) ** 2
    total_loss = reprojection_loss + depth_loss
    return total_loss.sum()


def body_fitting_loss_fullcam(body_pose, betas, model_joints, cam_t, center, scale, img_focal, img_center,
                              joints_2d, joints_conf, pose_prior,
                              crop_res=224, sigma=100, pose_prior_weight=4.78,
                              shape_prior_weight=5, angle_prior_weight=15.2,
                              init_body_pose=None, init_betas=None, init_global_orient=None, global_orient=None,
                              init_pose_reg_weight=0.0, init_betas_reg_weight=0.0, init_global_orient_reg_weight=0.0,
                              rot_angle=None, is_flipped=None,
                              output='sum'):
    """Body fitting loss with full-image perspective camera translation."""
    batch_size = body_pose.shape[0]
    rotation = torch.eye(3, device=body_pose.device).unsqueeze(0).expand(batch_size, -1, -1)
    _, projected_joints_crop, _ = project_points_full_to_crop(
        model_joints, cam_t, center, scale, img_focal, img_center, crop_res,
        rotation=rotation, rot_angle=rot_angle, is_flipped=is_flipped,
    )

    reprojection_error = gmof(projected_joints_crop - joints_2d, sigma)
    reprojection_loss = (joints_conf ** 2) * reprojection_error.sum(dim=-1)

    pose_prior_loss = (pose_prior_weight ** 2) * pose_prior(body_pose, betas)
    angle_prior_loss = (angle_prior_weight ** 2) * angle_prior(body_pose).sum(dim=-1)
    shape_prior_loss = (shape_prior_weight ** 2) * (betas ** 2).sum(dim=-1)

    total_loss = reprojection_loss.sum(dim=-1) + pose_prior_loss + angle_prior_loss + shape_prior_loss

    if init_body_pose is not None and init_pose_reg_weight > 0.0:
        total_loss = total_loss + (init_pose_reg_weight ** 2) * ((body_pose - init_body_pose) ** 2).sum(dim=-1)
    if init_betas is not None and init_betas_reg_weight > 0.0:
        total_loss = total_loss + (init_betas_reg_weight ** 2) * ((betas - init_betas) ** 2).sum(dim=-1)
    if global_orient is not None and init_global_orient is not None and init_global_orient_reg_weight > 0.0:
        total_loss = total_loss + (init_global_orient_reg_weight ** 2) * (
            (global_orient - init_global_orient) ** 2
        ).sum(dim=-1)

    if output == 'sum':
        return total_loss.sum()
    elif output == 'reprojection':
        return reprojection_loss


def camera_fitting_loss_fullcam(model_joints, cam_t, init_cam_t, center, scale, img_focal, img_center,
                                joints_2d, joints_conf, crop_res=224, depth_loss_weight=100,
                                rot_angle=None, is_flipped=None):
    """Camera fitting loss with full-image perspective camera translation."""
    batch_size = model_joints.shape[0]
    rotation = torch.eye(3, device=model_joints.device).unsqueeze(0).expand(batch_size, -1, -1)
    _, projected_joints_crop, _ = project_points_full_to_crop(
        model_joints, cam_t, center, scale, img_focal, img_center, crop_res,
        rotation=rotation, rot_angle=rot_angle, is_flipped=is_flipped,
    )

    op_joints = ['OP RHip', 'OP LHip', 'OP RShoulder', 'OP LShoulder']
    op_joints_ind = [constants.JOINT_IDS[joint] for joint in op_joints]
    gt_joints = ['Right Hip', 'Left Hip', 'Right Shoulder', 'Left Shoulder']
    gt_joints_ind = [constants.JOINT_IDS[joint] for joint in gt_joints]

    reprojection_error_op = (joints_2d[:, op_joints_ind] - projected_joints_crop[:, op_joints_ind]) ** 2
    reprojection_error_gt = (joints_2d[:, gt_joints_ind] - projected_joints_crop[:, gt_joints_ind]) ** 2
    is_valid = (joints_conf[:, op_joints_ind].min(dim=-1)[0][:, None, None] > 0).float()
    reprojection_loss = (is_valid * reprojection_error_op + (1 - is_valid) * reprojection_error_gt).sum(dim=(1, 2))

    depth_loss = (depth_loss_weight ** 2) * (cam_t[:, 2] - init_cam_t[:, 2]) ** 2
    total_loss = reprojection_loss + depth_loss
    return total_loss.sum()
