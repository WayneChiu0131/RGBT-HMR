import argparse
import copy
import json
import os
import sys

import numpy as np

import config

try:
    import yaml
except Exception:
    yaml = None


class _CompatBooleanOptionalAction(argparse.Action):
    """Python <3.9 fallback for argparse.BooleanOptionalAction."""

    def __init__(self, option_strings, dest, default=None, required=False, help=None):
        option_variants = []
        for option in option_strings:
            option_variants.append(option)
            if option.startswith('--'):
                option_variants.append('--no-' + option[2:])
        super().__init__(
            option_strings=option_variants,
            dest=dest,
            nargs=0,
            default=default,
            required=required,
            help=help,
        )

    def __call__(self, parser, namespace, values, option_string=None):
        setattr(namespace, self.dest, not option_string.startswith('--no-'))


BOOLEAN_OPTIONAL_ACTION = getattr(argparse, 'BooleanOptionalAction', _CompatBooleanOptionalAction)


FIELD_MAP = {
    'name': ('EXPERIMENT', 'NAME'),
    'time_to_run': ('EXPERIMENT', 'TIME_TO_RUN'),
    'resume': ('EXPERIMENT', 'RESUME'),
    'log_dir': ('IO', 'LOG_DIR'),
    'checkpoint': ('IO', 'CHECKPOINT'),
    'pretrained_checkpoint': ('IO', 'PRETRAINED_CHECKPOINT'),
    'backbone': ('MODEL', 'BACKBONE'),
    'img_res': ('MODEL', 'IMG_RES'),
    'strict_hrnet': ('MODEL', 'STRICT_HRNET'),
    'hrnet_backbone_input_height': ('MODEL', 'HRNET_BACKBONE_INPUT_HEIGHT'),
    'hrnet_backbone_input_width': ('MODEL', 'HRNET_BACKBONE_INPUT_WIDTH'),
    'hrnet_center_crop_input': ('MODEL', 'HRNET_CENTER_CROP_INPUT'),
    'model_pretrained': ('MODEL', 'PRETRAINED_INIT'),
    'strict_vitpose': ('MODEL', 'STRICT_VITPOSE'),
    'vitpose_checkpoint': ('MODEL', 'VITPOSE_CHECKPOINT'),
    'vitpose_input_height': ('MODEL', 'VITPOSE_INPUT_HEIGHT'),
    'vitpose_input_width': ('MODEL', 'VITPOSE_INPUT_WIDTH'),
    'vitpose_center_crop_input': ('MODEL', 'VITPOSE_CENTER_CROP_INPUT'),
    'vitpose_freeze': ('MODEL', 'VITPOSE_FREEZE'),
    'vitpose_pool': ('MODEL', 'VITPOSE_POOL'),
    'head_type': ('MODEL', 'HMR_HEAD_TYPE'),
    'decoder_dim': ('MODEL', 'DECODER_DIM'),
    'decoder_depth': ('MODEL', 'DECODER_DEPTH'),
    'decoder_heads': ('MODEL', 'DECODER_HEADS'),
    'decoder_mlp_ratio': ('MODEL', 'DECODER_MLP_RATIO'),
    'decoder_dropout': ('MODEL', 'DECODER_DROPOUT'),
    'camera_rep': ('MODEL', 'CAMERA_REP'),
    'full_persp_init_depth': ('MODEL', 'FULL_PERSP_INIT_DEPTH'),
    'full_persp_min_depth': ('MODEL', 'FULL_PERSP_MIN_DEPTH'),
    'full_persp_tz_param': ('MODEL', 'FULL_PERSP_TZ_PARAM'),
    'use_bbox_token': ('MODEL', 'USE_BBOX_TOKEN'),
    'bbox_token_hidden': ('MODEL', 'BBOX_TOKEN_HIDDEN'),
    'dataset_list': ('DATASET', 'LIST'),
    'dataset_partition': ('DATASET', 'PARTITION'),
    'dataset_npz_overrides': ('DATASET', 'NPZ_OVERRIDES'),
    'num_epochs': ('TRAIN', 'NUM_EPOCHS'),
    'lr': ('TRAIN', 'LR'),
    'backbone_lr': ('TRAIN', 'BACKBONE_LR'),
    'head_lr': ('TRAIN', 'HEAD_LR'),
    'cam_head_lr': ('TRAIN', 'CAM_HEAD_LR'),
    # MU-HMR (Stage2 modality-unified): model wrapper + adapter/unifier LRs.
    # Nested MODEL.MODALITY_UNIFIER / MODEL.THERMAL_ADAPTER are read directly
    # off the structured config namespace tree (options.MODEL.*), not flattened.
    'model_type': ('MODEL', 'TYPE'),
    'unifier_lr': ('TRAIN', 'UNIFIER_LR'),
    'thermal_adapter_lr': ('TRAIN', 'THERMAL_ADAPTER_LR'),
    # PBF-HMR2 (pre-backbone RGB-T fusion) differential LRs.
    'input_adapter_lr': ('TRAIN', 'INPUT_ADAPTER_LR'),
    'spatial_gate_lr': ('TRAIN', 'SPATIAL_GATE_LR'),
    # LT-RHMR (late-token RGB-T fusion) differential LRs.
    'thermal_token_adapter_lr': ('TRAIN', 'THERMAL_TOKEN_ADAPTER_LR'),
    'late_token_unifier_lr': ('TRAIN', 'LATE_TOKEN_UNIFIER_LR'),
    # Stage2C MMPedUnifiedHMR2 unified encoder / adapter LRs.
    'unified_encoder_lr': ('TRAIN', 'UNIFIED_ENCODER_LR'),
    'ir_token_adapter_lr': ('TRAIN', 'IR_TOKEN_ADAPTER_LR'),
    'body_adapter_lr': ('TRAIN', 'BODY_ADAPTER_LR'),
    'ir_patch_stem_lr': ('TRAIN', 'IR_PATCH_STEM_LR'),
    'go_residual_lr': ('TRAIN', 'GO_RESIDUAL_LR'),
    'xmodal_deep_lr': ('TRAIN', 'XMODAL_DEEP_LR'),
    # MU-HMR core selection: 'vitpose' (default, project hmr()) | 'hmr2' (4D-Humans port)
    'core_type': ('MODEL', 'CORE_TYPE'),
    'hmr2_checkpoint': ('MODEL', 'HMR2_CHECKPOINT'),
    'batch_size': ('TRAIN', 'BATCH_SIZE'),
    'weight_decay': ('TRAIN', 'WEIGHT_DECAY'),
    'optimizer': ('TRAIN', 'OPTIMIZER'),
    'focal_estimate_mode': ('CAMERA', 'FOCAL_ESTIMATE_MODE'),
    'num_workers': ('DATALOADER', 'NUM_WORKERS'),
    'pin_memory': ('DATALOADER', 'PIN_MEMORY'),
    'shuffle_train': ('DATALOADER', 'SHUFFLE_TRAIN'),
    'summary_steps': ('LOGGING', 'SUMMARY_STEPS'),
    'test_steps': ('LOGGING', 'TEST_STEPS'),
    'checkpoint_steps': ('LOGGING', 'CHECKPOINT_STEPS'),
    'render_summaries': ('LOGGING', 'RENDER_SUMMARIES'),
    'use_augmentation': ('AUG', 'USE_AUGMENTATION'),
    'rot_factor': ('AUG', 'ROT_FACTOR'),
    'noise_factor': ('AUG', 'NOISE_FACTOR'),
    'scale_factor': ('AUG', 'SCALE_FACTOR'),
    'rot_zero_prob': ('AUG', 'ROT_ZERO_PROB'),
    'occ_prob': ('AUG', 'OCC_PROB'),
    'flip_prob': ('AUG', 'FLIP_PROB'),
    'extreme_crop_aug_rate': ('AUG', 'EXTREME_CROP_AUG_RATE'),
    'extreme_crop_aug_level': ('AUG', 'EXTREME_CROP_AUG_LEVEL'),
    'grayscale_prob': ('AUG', 'GRAYSCALE_PROB'),
    'blur_prob': ('AUG', 'BLUR_PROB'),
    'blur_max_sigma': ('AUG', 'BLUR_MAX_SIGMA'),
    'ignore_3d': ('DEBUG', 'IGNORE_3D'),
    'skip_dataset_image_check': ('DEBUG', 'SKIP_DATASET_IMAGE_CHECK'),
    'shape_loss_weight': ('LOSS', 'SHAPE_LOSS_WEIGHT'),
    'keypoint_loss_weight': ('LOSS', 'KEYPOINT_LOSS_WEIGHT'),
    'keypoint_2d_loss_weight': ('LOSS', 'KEYPOINT_2D_LOSS_WEIGHT'),
    'keypoint_3d_loss_weight': ('LOSS', 'KEYPOINT_3D_LOSS_WEIGHT'),
    'pose_loss_weight': ('LOSS', 'POSE_LOSS_WEIGHT'),
    'beta_loss_weight': ('LOSS', 'BETA_LOSS_WEIGHT'),
    'global_orient_anchor_weight': ('LOSS', 'GLOBAL_ORIENT_ANCHOR_WEIGHT'),
    'head_l2sp_weight': ('LOSS', 'HEAD_L2SP_WEIGHT'),
    'beta_loss_filter_enabled': ('LOSS', 'BETA_LOSS_FILTER_ENABLED'),
    'beta_loss_threshold': ('LOSS', 'BETA_LOSS_THRESHOLD'),
    'cam_depth_loss_weight': ('LOSS', 'CAM_DEPTH_LOSS_WEIGHT'),
    'cam_loss_weight': ('LOSS', 'CAM_LOSS_WEIGHT'),
    'cam_full_loss_weight': ('LOSS', 'CAM_FULL_LOSS_WEIGHT'),
    'openpose_train_weight': ('LOSS', 'OPENPOSE_TRAIN_WEIGHT'),
    'gt_train_weight': ('LOSS', 'GT_TRAIN_WEIGHT'),
    'supervision_mode': ('LOSS', 'SUPERVISION_MODE'),
    # Scheme-1 gate losses (MMPed per-token reliability fusion).
    # gate_anchor_ir lives under MODEL.UNIFIED_ENCODER and is read directly from
    # that cfg in the trainer (nested key, not a flat FIELD_MAP entry).
    'gate_anchor_loss_weight': ('LOSS', 'GATE_ANCHOR_LOSS_WEIGHT'),
    'oracle_gate_loss_weight': ('LOSS', 'ORACLE_GATE_LOSS_WEIGHT'),
    'oracle_gate_target_npz': ('LOSS', 'ORACLE_GATE_TARGET_NPZ'),
    'oracle_gate_ramp_steps': ('LOSS', 'ORACLE_GATE_RAMP_STEPS'),
    'oracle_gate_decay_steps': ('LOSS', 'ORACLE_GATE_DECAY_STEPS'),
    'run_smplify': ('SMPLIFY', 'RUN'),
    'smplify_threshold': ('SMPLIFY', 'THRESHOLD'),
    'num_smplify_iters': ('SMPLIFY', 'NUM_ITERS'),
    'num_smplify_cam_iters': ('SMPLIFY', 'NUM_CAM_ITERS'),
    'num_smplify_body_iters': ('SMPLIFY', 'NUM_BODY_ITERS'),
    'smplify_step_size': ('SMPLIFY', 'STEP_SIZE'),
    'smplify_pose_prior_weight': ('SMPLIFY', 'POSE_PRIOR_WEIGHT'),
    'smplify_shape_prior_weight': ('SMPLIFY', 'SHAPE_PRIOR_WEIGHT'),
    'smplify_angle_prior_weight': ('SMPLIFY', 'ANGLE_PRIOR_WEIGHT'),
    'smplify_sigma': ('SMPLIFY', 'SIGMA'),
    'smplify_init_pose_reg_weight': ('SMPLIFY', 'INIT_POSE_REG_WEIGHT'),
    'smplify_init_betas_reg_weight': ('SMPLIFY', 'INIT_BETAS_REG_WEIGHT'),
    'smplify_init_global_orient_reg_weight': ('SMPLIFY', 'INIT_GLOBAL_ORIENT_REG_WEIGHT'),
    'valid_step': ('VALIDATION', 'INTERVAL'),
    'save_step': ('TRAIN', 'SAVE_STEP'),
    'max_step': ('TRAIN', 'MAX_STEP'),
    'lr_schedule': ('TRAIN', 'LR_SCHEDULE'),
    'warmup_steps': ('TRAIN', 'WARMUP_STEPS'),
    'loss_record_steps': ('LOGGING', 'LOSS_RECORD_STEPS'),
    'validation_enabled': ('VALIDATION', 'ENABLED'),
    'validation_dataset': ('VALIDATION', 'DATASET'),
    'validation_metric': ('VALIDATION', 'METRIC'),
    'validation_save_best': ('VALIDATION', 'SAVE_BEST'),
    'validation_batch_size': ('VALIDATION', 'BATCH_SIZE'),
    'validation_num_workers': ('VALIDATION', 'NUM_WORKERS'),
    'validation_npz_path': ('VALIDATION', 'PW3D_NPZ'),
    'validation_sequence_path': ('VALIDATION', 'PW3D_SEQUENCE_DIR'),
    # Scheme A: modality embedding
    'modality_embed_dim': ('MODEL', 'MODALITY_EMBED_DIM'),
    'num_modalities': ('MODEL', 'NUM_MODALITIES'),
    # Scheme A: thermal-like augmentation
    'invert_prob': ('AUG', 'INVERT_PROB'),
    'clahe_prob': ('AUG', 'CLAHE_PROB'),
    'thermal_noise_prob': ('AUG', 'THERMAL_NOISE_PROB'),
    'thermal_noise_std': ('AUG', 'THERMAL_NOISE_STD'),
    'thermal_like_prob': ('AUG', 'THERMAL_LIKE_PROB'),
    'bridge_aug_as_thermal_modality': ('AUG', 'BRIDGE_AUG_AS_THERMAL_MODALITY'),
    # Scheme A: pGT weighting
    'use_pgt_weight': ('LOSS', 'USE_PGT_WEIGHT'),
    'thermal_smpl_global_weight': ('LOSS', 'THERMAL_SMPL_GLOBAL_WEIGHT'),
    'thermal_shape_global_weight': ('LOSS', 'THERMAL_SHAPE_GLOBAL_WEIGHT'),
    'thermal_cam_global_weight': ('LOSS', 'THERMAL_CAM_GLOBAL_WEIGHT'),
    # Scheme A: grad clip + OTP2 validation
    'grad_clip_norm': ('TRAIN', 'GRAD_CLIP_NORM'),
    'validation_otp2_npz': ('VALIDATION', 'OTP2_NPZ'),
    'validation_llvip_npz': ('VALIDATION', 'LLVIP_NPZ'),
    # Plan 06.28: focal policy + weak-cam clamp + 2D pixel loss + multi-val
    'focal_policy': ('CAMERA', 'FOCAL_POLICY'),
    'fixed_focal': ('CAMERA', 'FIXED_FOCAL'),
    'log_focal_source': ('CAMERA', 'LOG_FOCAL_SOURCE'),
    'weak_cam_scale_min': ('MODEL', 'WEAK_CAM_SCALE_MIN'),
    'weak_cam_scale_max': ('MODEL', 'WEAK_CAM_SCALE_MAX'),
    'keypoint_2d_pixel_loss_weight': ('LOSS', 'KEYPOINT_2D_PIXEL_LOSS_WEIGHT'),
    'keypoint_2d_loss_beta': ('LOSS', 'KEYPOINT_2D_LOSS_BETA'),
    'cam_full_loss_start_step': ('LOSS', 'CAM_FULL_LOSS_START_STEP'),
    'cam_full_loss_ramp_steps': ('LOSS', 'CAM_FULL_LOSS_RAMP_STEPS'),
    'validation_datasets': ('VALIDATION', 'DATASETS'),
    'validation_primary_metric': ('VALIDATION', 'PRIMARY_METRIC'),
    'thermal_projection_weights': ('VALIDATION', 'THERMAL_PROJECTION_WEIGHTS'),
}


REQUIRED_FIELDS = [
    'name',
    'time_to_run',
    'log_dir',
    'num_epochs',
    'lr',
    'batch_size',
    'backbone',
    'img_res',
    'dataset_list',
    'dataset_partition',
]


class TrainOptions:
    def __init__(self):
        self.parser = argparse.ArgumentParser()
        self._add_arguments()

    def _add_arguments(self):
        io = self.parser.add_argument_group('Config I/O')
        io.add_argument('--paths_cfg', default=None, type=str, help='Machine-local paths yaml/json')
        io.add_argument('--paths_yaml', default=None, type=str, help='Alias of --paths_cfg')
        io.add_argument('--train_cfg', default=None, type=str, help='Train params yaml/json')
        io.add_argument('--params_cfg', default=None, type=str, help='Alias of --train_cfg')
        io.add_argument('--cfg', default=None, type=str, help='Alias of --train_cfg')
        io.add_argument('--from_yaml', default=None, type=str, help='Legacy alias of --train_cfg')
        io.add_argument('--from_json', default=None, type=str, help='Legacy json config path')
        io.add_argument('--dump_json_compat', action='store_true', default=False, help='Also dump config.json')

        self.parser.add_argument(
            '--dataset_root_override',
            action='append',
            default=[],
            help='Runtime dataset root override, format: KEY=VALUE (e.g. H36M_ROOT=/path)',
        )
        self.parser.add_argument(
            '--path_override',
            action='append',
            default=[],
            help='Runtime path override, format: KEY=VALUE (e.g. SMPL_MODEL_DIR=/path)',
        )

        self.parser.add_argument('--name', default=None, type=str)
        self.parser.add_argument('--time_to_run', default=None, type=int)
        self.parser.add_argument('--resume', action=BOOLEAN_OPTIONAL_ACTION, default=None)
        self.parser.add_argument('--log_dir', default=None, type=str)
        self.parser.add_argument('--checkpoint', default=None, type=str)
        self.parser.add_argument('--pretrained_checkpoint', default=None, type=str)

        self.parser.add_argument('--num_epochs', default=None, type=int)
        self.parser.add_argument('--lr', default=None, type=float)
        self.parser.add_argument('--backbone_lr', default=None, type=float,
                                 help='Differential LR for ViTPose/HRNet backbone. None -> fallback to --lr')
        self.parser.add_argument('--head_lr', default=None, type=float,
                                 help='Differential LR for HMR head. None -> fallback to --lr')
        self.parser.add_argument('--cam_head_lr', default=None, type=float,
                                 help='Differential LR for camera head (deccam). None -> fallback to --lr')
        self.parser.add_argument('--model_type', default=None, type=str,
                                 help="MODEL.TYPE; 'modality_unified_hmr' wraps the HMR core in MU-HMR")
        self.parser.add_argument('--unifier_lr', default=None, type=float,
                                 help='Differential LR for ModalityUnifier (RGMR). None -> fallback to --lr')
        self.parser.add_argument('--thermal_adapter_lr', default=None, type=float,
                                 help='Differential LR for ThermalAdapter. None -> fallback to --lr')
        self.parser.add_argument('--input_adapter_lr', default=None, type=float,
                                 help='PBF-HMR2: LR for pre-backbone ThermalInputAdapter. None -> --lr')
        self.parser.add_argument('--spatial_gate_lr', default=None, type=float,
                                 help='PBF-HMR2: LR for SpatialReliabilityGate. None -> --lr')
        self.parser.add_argument('--thermal_token_adapter_lr', default=None, type=float,
                                 help='LT-RHMR: LR for late-token ThermalTokenAdapter. None -> --lr')
        self.parser.add_argument('--late_token_unifier_lr', default=None, type=float,
                                 help='LT-RHMR: LR for LateTokenReliabilityUnifier. None -> --lr')
        self.parser.add_argument('--unified_encoder_lr', default=None, type=float,
                                 help='Stage2C: LR for MMPed-style HMR2 unified encoder. None -> --lr')
        self.parser.add_argument('--ir_token_adapter_lr', default=None, type=float,
                                 help='Stage2C-B: LR for the bounded IR token adapter. None -> --lr')
        self.parser.add_argument('--body_adapter_lr', default=None, type=float,
                                 help='Scheme-2: LR for the ViT body block adapters. None -> --lr')
        self.parser.add_argument('--ir_patch_stem_lr', default=None, type=float,
                                 help='P2b: LR for the trainable IR patch stem. None -> --lr')
        self.parser.add_argument('--go_residual_lr', default=None, type=float,
                                 help='Tier-1: LR for the bounded global-orient residual. None -> --lr')
        self.parser.add_argument('--xmodal_deep_lr', default=None, type=float)
        self.parser.add_argument('--gate_anchor_loss_weight', default=None, type=float,
                                 help='Scheme-1: weight for the gate anchor loss (alpha_ir mean -> anchor).')
        self.parser.add_argument('--oracle_gate_loss_weight', default=None, type=float,
                                 help='Scheme-1: weight for the oracle gate BCE loss.')
        self.parser.add_argument('--oracle_gate_target_npz', default=None, type=str,
                                 help='Scheme-1: path to precomputed oracle gate targets (keyed by imgname).')
        self.parser.add_argument('--oracle_gate_ramp_steps', default=None, type=int,
                                 help='Scheme-1: linear ramp steps for the oracle gate loss weight.')
        self.parser.add_argument('--oracle_gate_decay_steps', default=None, type=int,
                                 help='Scheme-2: linear decay steps for the oracle gate loss weight (target goes stale).')
        self.parser.add_argument('--core_type', default=None, type=str,
                                 help="MU-HMR core: 'vitpose' (default) | 'hmr2' (4D-Humans port)")
        self.parser.add_argument('--hmr2_checkpoint', default=None, type=str,
                                 help='Path to HMR2.0 (.ckpt) for core_type=hmr2')
        self.parser.add_argument('--focal_estimate_mode', default=None, type=str,
                                 choices=['diagonal', 'max', 'height'],
                                 help='Pseudo-focal estimation mode for samples without real intrinsics')
        self.parser.add_argument('--lr_schedule', default=None, type=str,
                                 choices=['constant', 'cosine_warmup'])
        self.parser.add_argument('--warmup_steps', default=None, type=int)
        self.parser.add_argument('--batch_size', default=None, type=int)
        self.parser.add_argument('--summary_steps', default=None, type=int)
        self.parser.add_argument('--test_steps', default=None, type=int)
        self.parser.add_argument('--checkpoint_steps', default=None, type=int)
        self.parser.add_argument('--loss_record_steps', default=None, type=int)
        self.parser.add_argument('--valid_step', default=None, type=int)
        self.parser.add_argument('--save_step', default=None, type=int)
        self.parser.add_argument('--max_step', default=None, type=int)
        self.parser.add_argument('--img_res', default=None, type=int)
        self.parser.add_argument('--backbone', default=None, type=str, choices=['resnet50', 'hrnet_w48', 'vitpose_huge'])
        self.parser.add_argument('--model_pretrained', action=BOOLEAN_OPTIONAL_ACTION, default=None)
        self.parser.add_argument('--strict_hrnet', action=BOOLEAN_OPTIONAL_ACTION, default=None)
        self.parser.add_argument('--hrnet_backbone_input_height', default=None, type=int)
        self.parser.add_argument('--hrnet_backbone_input_width', default=None, type=int)
        self.parser.add_argument('--hrnet_center_crop_input', action=BOOLEAN_OPTIONAL_ACTION, default=None)
        self.parser.add_argument('--hrnet_pretrained', default=None, type=str, help='Path-like field, runtime override only')
        self.parser.add_argument('--strict_vitpose', action=BOOLEAN_OPTIONAL_ACTION, default=None)
        self.parser.add_argument('--vitpose_checkpoint', default=None, type=str)
        self.parser.add_argument('--vitpose_input_height', default=None, type=int)
        self.parser.add_argument('--vitpose_input_width', default=None, type=int)
        self.parser.add_argument('--vitpose_center_crop_input', action=BOOLEAN_OPTIONAL_ACTION, default=None)
        self.parser.add_argument('--vitpose_freeze', action=BOOLEAN_OPTIONAL_ACTION, default=None)
        self.parser.add_argument('--vitpose_pool', default=None, type=str, choices=['mean', 'cls', 'token'])
        self.parser.add_argument('--head_type', default=None, type=str,
                                 choices=['iterative_mlp', 'transformer_decoder'],
                                 help='HMR regression head: legacy iterative MLP or HMR2 transformer decoder')
        self.parser.add_argument('--decoder_dim', default=None, type=int)
        self.parser.add_argument('--decoder_depth', default=None, type=int)
        self.parser.add_argument('--decoder_heads', default=None, type=int)
        self.parser.add_argument('--decoder_mlp_ratio', default=None, type=float)
        self.parser.add_argument('--decoder_dropout', default=None, type=float)
        self.parser.add_argument('--camera_rep', default=None, type=str,
                                 choices=['weak_perspective', 'full_perspective'],
                                 help="Camera representation: weak_perspective (legacy s,tx,ty) or "
                                      "full_perspective (tx,ty,tz). Only transformer_decoder supports full_perspective.")
        self.parser.add_argument('--full_persp_init_depth', default=None, type=float,
                                 help='Mean init depth (tz, metres) for full_perspective camera_rep.')
        self.parser.add_argument('--full_persp_min_depth', default=None, type=float,
                                 help='Clamp threshold on tz for full_perspective camera_rep (numerical safety).')
        self.parser.add_argument('--full_persp_tz_param', default=None, type=str,
                                 choices=['clamp', 'softplus'],
                                 help="tz positivity parametrisation for full_perspective: 'clamp' "
                                      "(Step 3 back-compat) or 'softplus' (Step 6 ablation, fully differentiable).")
        self.parser.add_argument('--use_bbox_token', action=BOOLEAN_OPTIONAL_ACTION, default=None,
                                 help='Project bbox_info into one extra memory token for the '
                                      'transformer_decoder head (Step 4).')
        self.parser.add_argument('--bbox_token_hidden', default=None, type=int,
                                 help='Hidden width of the bbox_info -> token MLP (default 128).')
        self.parser.add_argument('--weight_decay', default=None, type=float)
        self.parser.add_argument('--optimizer', default=None, type=str,
                                 choices=['adam', 'adamw'])

        self.parser.add_argument('--dataset_list', nargs='+', default=None)
        self.parser.add_argument('--dataset_partition', nargs='+', type=float, default=None)
        self.parser.add_argument('--rot_factor', default=None, type=float)
        self.parser.add_argument('--noise_factor', default=None, type=float)
        self.parser.add_argument('--scale_factor', default=None, type=float)
        self.parser.add_argument('--rot_zero_prob', default=None, type=float)
        self.parser.add_argument('--occ_prob', default=None, type=float)
        self.parser.add_argument('--flip_prob', default=None, type=float)
        self.parser.add_argument('--extreme_crop_aug_rate', default=None, type=float)
        self.parser.add_argument('--extreme_crop_aug_level', default=None, type=int)
        self.parser.add_argument('--grayscale_prob', default=None, type=float)
        self.parser.add_argument('--blur_prob', default=None, type=float)
        self.parser.add_argument('--blur_max_sigma', default=None, type=float)
        self.parser.add_argument('--use_augmentation', action=BOOLEAN_OPTIONAL_ACTION, default=None)
        self.parser.add_argument('--skip_dataset_image_check', action=BOOLEAN_OPTIONAL_ACTION, default=None)
        self.parser.add_argument('--render_summaries', action=BOOLEAN_OPTIONAL_ACTION, default=None)
        self.parser.add_argument('--ignore_3d', action=BOOLEAN_OPTIONAL_ACTION, default=None)
        self.parser.add_argument('--shape_loss_weight', default=None, type=float)
        self.parser.add_argument('--keypoint_loss_weight', default=None, type=float)
        self.parser.add_argument('--keypoint_2d_loss_weight', default=None, type=float)
        self.parser.add_argument('--keypoint_3d_loss_weight', default=None, type=float)
        self.parser.add_argument('--pose_loss_weight', default=None, type=float)
        self.parser.add_argument('--beta_loss_weight', default=None, type=float)
        self.parser.add_argument('--global_orient_anchor_weight', default=None, type=float)
        self.parser.add_argument('--head_l2sp_weight', default=None, type=float,
                                 help='L2-SP anchor pulling the (unfrozen) SMPL head back '
                                      'toward the pretrained HMR2.0 head; stops the pose '
                                      'drift that breaks validation when FREEZE_HEAD=false.')
        self.parser.add_argument('--beta_loss_filter_enabled', action=BOOLEAN_OPTIONAL_ACTION, default=None)
        self.parser.add_argument('--beta_loss_threshold', default=None, type=float)
        self.parser.add_argument('--cam_depth_loss_weight', default=None, type=float)
        self.parser.add_argument('--cam_loss_weight', default=None, type=float)
        self.parser.add_argument('--cam_full_loss_weight', default=None, type=float)
        self.parser.add_argument('--openpose_train_weight', default=None, type=float)
        self.parser.add_argument('--gt_train_weight', default=None, type=float)
        self.parser.add_argument('--supervision_mode', default=None, type=str, choices=['gt_only', 'fit_or_gt'])
        self.parser.add_argument('--validation_enabled', action=BOOLEAN_OPTIONAL_ACTION, default=None)
        self.parser.add_argument('--validation_dataset', default=None, type=str)
        self.parser.add_argument('--validation_metric', default=None, type=str)
        self.parser.add_argument('--validation_save_best', action=BOOLEAN_OPTIONAL_ACTION, default=None)
        self.parser.add_argument('--validation_batch_size', default=None, type=int)
        self.parser.add_argument('--validation_num_workers', default=None, type=int)
        self.parser.add_argument('--validation_npz_path', default=None, type=str)
        self.parser.add_argument('--validation_sequence_path', default=None, type=str)
        self.parser.add_argument('--run_smplify', action=BOOLEAN_OPTIONAL_ACTION, default=None)
        self.parser.add_argument('--smplify_threshold', default=None, type=float)
        self.parser.add_argument('--num_smplify_iters', default=None, type=int)
        self.parser.add_argument('--num_smplify_cam_iters', default=None, type=int)
        self.parser.add_argument('--num_smplify_body_iters', default=None, type=int)
        self.parser.add_argument('--smplify_step_size', default=None, type=float)
        self.parser.add_argument('--smplify_pose_prior_weight', default=None, type=float)
        self.parser.add_argument('--smplify_shape_prior_weight', default=None, type=float)
        self.parser.add_argument('--smplify_angle_prior_weight', default=None, type=float)
        self.parser.add_argument('--smplify_sigma', default=None, type=float)
        self.parser.add_argument('--smplify_init_pose_reg_weight', default=None, type=float)
        self.parser.add_argument('--smplify_init_betas_reg_weight', default=None, type=float)
        self.parser.add_argument('--smplify_init_global_orient_reg_weight', default=None, type=float)
        self.parser.add_argument('--shuffle_train', action=BOOLEAN_OPTIONAL_ACTION, default=None)
        # Scheme A new args
        self.parser.add_argument('--modality_embed_dim', default=None, type=int)
        self.parser.add_argument('--num_modalities', default=None, type=int)
        self.parser.add_argument('--invert_prob', default=None, type=float)
        self.parser.add_argument('--clahe_prob', default=None, type=float)
        self.parser.add_argument('--thermal_noise_prob', default=None, type=float)
        self.parser.add_argument('--thermal_noise_std', default=None, type=float)
        self.parser.add_argument('--thermal_like_prob', default=None, type=float)
        self.parser.add_argument('--bridge_aug_as_thermal_modality', action=BOOLEAN_OPTIONAL_ACTION, default=None)
        self.parser.add_argument('--use_pgt_weight', action=BOOLEAN_OPTIONAL_ACTION, default=None)
        self.parser.add_argument('--thermal_smpl_global_weight', default=None, type=float)
        self.parser.add_argument('--thermal_shape_global_weight', default=None, type=float)
        self.parser.add_argument('--thermal_cam_global_weight', default=None, type=float)
        self.parser.add_argument('--grad_clip_norm', default=None, type=float)
        self.parser.add_argument('--validation_otp2_npz', default=None, type=str)
        self.parser.add_argument('--validation_llvip_npz', default=None, type=str)
        # Plan 06.28: focal policy + weak-cam scale clamp + 2D pixel loss
        self.parser.add_argument('--focal_policy', default=None, type=str,
                                 choices=['real_else_fixed', 'real_else_diagonal',
                                          'real_else_moge', 'fixed_all', 'diagonal_all'])
        self.parser.add_argument('--fixed_focal', default=None, type=float)
        self.parser.add_argument('--log_focal_source', action=BOOLEAN_OPTIONAL_ACTION, default=None)
        self.parser.add_argument('--weak_cam_scale_min', default=None, type=float)
        self.parser.add_argument('--weak_cam_scale_max', default=None, type=float)
        self.parser.add_argument('--keypoint_2d_pixel_loss_weight', default=None, type=float)
        self.parser.add_argument('--keypoint_2d_loss_beta', default=None, type=float,
                                 help='SmoothL1 beta for keypoint_2d_loss, in NORMALIZED units. '
                                      'Typical error is ~0.105; beta=1.0 (default) -> quadratic/MSE '
                                      'branch, beta<0.105 -> linear/L1 branch (~10x more gradient).')
        self.parser.add_argument('--cam_full_loss_start_step', default=None, type=int)
        self.parser.add_argument('--cam_full_loss_ramp_steps', default=None, type=int)
        self.parser.add_argument('--validation_primary_metric', default=None, type=str)

    def parse_args(self):
        cli_args = self.parser.parse_args()
        explicit_dests = self._collect_explicit_cli_dests()

        paths_cfg_path = self._resolve_paths_config_path(cli_args)
        paths_cfg = self._load_paths_config(paths_cfg_path)
        paths_cfg = self._apply_runtime_path_overrides(paths_cfg, cli_args, explicit_dests)
        if paths_cfg:
            config.apply_paths_config(paths_cfg, verbose=False)
        cli_args.paths_config_source = paths_cfg_path

        cfg_path = self._resolve_train_config_path(cli_args)
        structured_cfg = self._load_train_structured_config(cfg_path)
        structured_cfg = self._sanitize_params_config(structured_cfg, cfg_path)

        merged_values = self._build_values_from_structured_cfg(structured_cfg)
        self._warn_on_duplicate_legacy_keys(structured_cfg)
        for dest in explicit_dests:
            if hasattr(cli_args, dest):
                val = getattr(cli_args, dest)
                if val is not None:
                    merged_values[dest] = val
        self._apply_backward_compat_defaults(merged_values, structured_cfg)
        self._coerce_values_to_declared_types(merged_values)

        args = argparse.Namespace(**merged_values)
        args.from_json = cli_args.from_json
        args.from_yaml = cli_args.from_yaml
        args.cfg = cli_args.cfg
        args.train_cfg = cli_args.train_cfg
        args.params_cfg = cli_args.params_cfg
        args.paths_cfg = cli_args.paths_cfg
        args.paths_yaml = cli_args.paths_yaml
        args.config_source = cfg_path
        args.paths_config_source = paths_cfg_path
        args.dump_json_compat = bool(getattr(cli_args, 'dump_json_compat', False))

        self._sync_structured_cfg_from_flat(structured_cfg, args)
        self._attach_structured_sections(args, structured_cfg)
        self._attach_paths_section(args, paths_cfg)
        self._require_fields(args)
        self._normalize_dataset_options(args)
        self._validate_hrnet_options(args)
        self._finalize_paths(args)

        npz_overrides = getattr(args, 'dataset_npz_overrides', None)
        if npz_overrides and isinstance(npz_overrides, dict):
            config.apply_dataset_npz_overrides(npz_overrides)

        self.args = args
        self.save_dump()
        return self.args

    def _warn(self, msg):
        print(f'[TrainOptions][Warn] {msg}')

    def _project_root(self):
        return os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))

    def _default_train_cfg_path(self):
        p = os.path.join(self._project_root(), 'configs', 'train', 'default.yaml')
        return os.path.abspath(p) if os.path.isfile(p) else None

    def _default_paths_cfg_path(self):
        p = os.path.join(self._project_root(), 'configs', 'paths', 'local.yaml')
        return os.path.abspath(p) if os.path.isfile(p) else None

    def _resolve_paths_config_path(self, args):
        path = args.paths_cfg or args.paths_yaml or self._default_paths_cfg_path()
        return os.path.abspath(path) if path else None

    def _resolve_train_config_path(self, args):
        path = args.train_cfg or args.params_cfg or args.cfg or args.from_yaml or args.from_json
        if path is None:
            path = self._default_train_cfg_path()
        return os.path.abspath(path) if path else None

    def _load_paths_config(self, paths_cfg_path):
        if paths_cfg_path is None:
            return {}
        if not os.path.isfile(paths_cfg_path):
            raise FileNotFoundError(f'paths config not found: {paths_cfg_path}')
        return config.load_paths_config(paths_cfg_path)

    def _load_train_structured_config(self, cfg_path):
        # YAML is the single source. This function keeps backward compatibility:
        # default train yaml -> merge custom cfg (if any).
        structured = config.build_train_config_defaults()
        if cfg_path is None:
            return structured
        if not os.path.isfile(cfg_path):
            raise FileNotFoundError(f'train config not found: {cfg_path}')
        loaded = config.load_train_params_config(cfg_path)
        return config.merge_config_with_defaults(structured, loaded)

    def _parse_kv_list(self, kv_list, kind):
        out = {}
        for item in kv_list:
            if '=' not in item:
                raise ValueError(f'Invalid {kind} override "{item}", expected KEY=VALUE.')
            key, val = item.split('=', 1)
            key = key.strip()
            val = val.strip()
            if key == '' or val == '':
                raise ValueError(f'Invalid {kind} override "{item}", expected KEY=VALUE.')
            out[key] = val
        return out

    def _apply_runtime_path_overrides(self, paths_cfg, cli_args, explicit_dests):
        merged = copy.deepcopy(paths_cfg) if isinstance(paths_cfg, dict) else {}
        if 'DATASET_ROOTS' not in merged:
            merged['DATASET_ROOTS'] = {}
        if 'PATHS' not in merged:
            merged['PATHS'] = {}

        ds_overrides = self._parse_kv_list(getattr(cli_args, 'dataset_root_override', []), 'dataset_root')
        path_overrides = self._parse_kv_list(getattr(cli_args, 'path_override', []), 'path')
        merged['DATASET_ROOTS'].update(ds_overrides)
        merged['PATHS'].update(path_overrides)

        if 'hrnet_pretrained' in explicit_dests and cli_args.hrnet_pretrained:
            merged['PATHS']['HRNET_PRETRAINED'] = cli_args.hrnet_pretrained
        return merged

    def _sanitize_params_config(self, cfg_dict, cfg_path):
        if not isinstance(cfg_dict, dict):
            return {}
        sanitized = copy.deepcopy(cfg_dict)
        cfg_name = cfg_path if cfg_path is not None else '<default train config>'

        if 'PATHS' in sanitized or 'paths' in sanitized:
            self._warn(f'Ignoring PATHS section in params config: {cfg_name}')
            sanitized.pop('PATHS', None)
            sanitized.pop('paths', None)
        if 'DATASET_ROOTS' in sanitized or 'dataset_roots' in sanitized:
            self._warn(f'Ignoring DATASET_ROOTS section in params config: {cfg_name}')
            sanitized.pop('DATASET_ROOTS', None)
            sanitized.pop('dataset_roots', None)

        model_cfg = sanitized.get('MODEL', sanitized.get('model', None))
        if isinstance(model_cfg, dict):
            if 'HRNET_PRETRAINED' in model_cfg or 'hrnet_pretrained' in model_cfg:
                self._warn(f'Ignoring MODEL.HRNET_PRETRAINED in params config: {cfg_name}')
                model_cfg.pop('HRNET_PRETRAINED', None)
                model_cfg.pop('hrnet_pretrained', None)

        for k in list(sanitized.keys()):
            if isinstance(k, str) and k.upper() in config.PATH_FIELD_KEYS:
                self._warn(f"Ignoring path-like key '{k}' in params config: {cfg_name}")
                sanitized.pop(k, None)
        return sanitized

    def _collect_explicit_cli_dests(self):
        explicit = set()
        option_map = self.parser._option_string_actions
        for token in sys.argv[1:]:
            if not token.startswith('--'):
                continue
            key = token.split('=', 1)[0]
            action = option_map.get(key)
            if action is not None:
                explicit.add(action.dest)
        return explicit

    def _section_any(self, cfg_dict, section):
        if not isinstance(cfg_dict, dict):
            return {}
        return cfg_dict.get(section, cfg_dict.get(section.lower(), {}))

    def _cfg_value(self, cfg_dict, section, key, flat_key):
        sec = self._section_any(cfg_dict, section)
        if isinstance(sec, dict):
            if key in sec:
                return sec[key]
            if key.lower() in sec:
                return sec[key.lower()]
        if flat_key in cfg_dict:
            return cfg_dict[flat_key]
        if flat_key.upper() in cfg_dict:
            return cfg_dict[flat_key.upper()]
        return None

    def _warn_on_duplicate_legacy_keys(self, structured_cfg):
        # Warn when a YAML defines the same concept under multiple legacy keys.
        # Fallback order elsewhere is unchanged; this only surfaces ambiguity.
        groups = [
            ('validation interval',
             [('VALIDATION', 'INTERVAL'),
              ('TRAIN', 'VALID_STEP'),
              ('LOGGING', 'TEST_STEPS')]),
            ('checkpoint step',
             [('TRAIN', 'SAVE_STEP'),
              ('LOGGING', 'CHECKPOINT_STEPS')]),
            # NOTE: the legacy 'keypoint loss weight' group was removed. The old
            # combined LOSS.KEYPOINT_LOSS_WEIGHT key is dead (never consumed by
            # the loss mixer, which reads KEYPOINT_2D_LOSS_WEIGHT and
            # KEYPOINT_3D_LOSS_WEIGHT independently). Grouping the 2D and 3D
            # weights produced a false-positive warning whenever a config set
            # both — which is the correct, intended usage.
        ]
        for label, keys in groups:
            present = []
            for section, key in keys:
                val = self._cfg_value(structured_cfg, section, key, key.lower())
                if val is not None:
                    present.append(f'{section}.{key}={val}')
            if len(present) > 1:
                print(f'[TrainOptions][WARN] multiple legacy keys set for {label}: '
                      f'{present}. Fallback order is unchanged; consider keeping one.')

        # 語意矛盾檢查：ENABLED=False 但 INTERVAL>0 時 warn
        validation_enabled = self._cfg_value(
            structured_cfg, 'VALIDATION', 'ENABLED', 'validation_enabled'
        )
        validation_interval = self._cfg_value(
            structured_cfg, 'VALIDATION', 'INTERVAL', 'valid_step'
        )
        try:
            interval_int = int(validation_interval) if validation_interval is not None else 0
        except (TypeError, ValueError):
            interval_int = 0
        if validation_enabled is False and interval_int > 0:
            print(
                f'[TrainOptions][WARN] VALIDATION.ENABLED=False but '
                f'VALIDATION.INTERVAL={interval_int} > 0; validation will be '
                f'skipped. If you intend to run validation, set ENABLED=True.'
            )

    def _build_values_from_structured_cfg(self, structured_cfg):
        values = {}
        for dest, (section, key) in FIELD_MAP.items():
            values[dest] = self._cfg_value(structured_cfg, section, key, dest)
        values['hrnet_pretrained'] = None
        return values

    @staticmethod
    def _parse_bool_like(value, field_name):
        if isinstance(value, bool):
            return value
        if isinstance(value, str):
            v = value.strip().lower()
            if v in ('1', 'true', 'yes', 'y', 'on'):
                return True
            if v in ('0', 'false', 'no', 'n', 'off'):
                return False
        raise ValueError(f"Invalid boolean value for '{field_name}': {value!r}")

    def _coerce_values_to_declared_types(self, values):
        """
        Coerce structured-config values to argparse-declared types so yaml strings
        such as "1e-4" are accepted for float fields.
        """
        action_by_dest = {}
        for action in self.parser._actions:
            if action.dest:
                action_by_dest[action.dest] = action

        for dest, val in list(values.items()):
            if val is None or dest not in action_by_dest:
                continue
            action = action_by_dest[dest]
            cast_type = getattr(action, 'type', None)
            nargs = getattr(action, 'nargs', None)

            # BooleanOptionalAction (and compat fallback) has nargs=0, type=None.
            if cast_type is None and nargs == 0:
                if isinstance(val, str):
                    values[dest] = self._parse_bool_like(val, dest)
                continue

            if cast_type is None:
                continue

            try:
                if isinstance(val, (list, tuple)):
                    values[dest] = [cast_type(x) for x in val]
                else:
                    values[dest] = cast_type(val)
            except Exception as e:
                tname = getattr(cast_type, '__name__', str(cast_type))
                raise ValueError(
                    f"Invalid value for '{dest}': {val!r} (expected {tname})"
                ) from e

    # ── LOSS.THERMAL_*_GLOBAL_WEIGHT 的單一旋鈕遷移 ────────────────────────────
    # 舊版把 pGT 的 SMPL 監督用「兩道串聯閥門」控制：
    #   閥門 1  LOSS.POSE_LOSS_WEIGHT / BETA_LOSS_WEIGHT / SHAPE_LOSS_WEIGHT / CAM_LOSS_WEIGHT
    #   閥門 2  LOSS.THERMAL_{SMPL,SHAPE,CAM}_GLOBAL_WEIGHT
    # 閥門 2 在 trainer 裡是 `sample_weight[thermal_mask] *= value`。但
    # `Trainer._thermal_sample_mask` 只看 dataset 名稱，而本 repo 訓練用的
    # llvip / llvip-pgt / otp2 / otp2-pgt 全在 thermal 名單裡 → 該遮罩對每個
    # 訓練 batch 都是全 True，閥門 2 因此等價於一個「乘在權重上的全域純量」。
    #
    # 這個等價性正是它危險的地方：閥門 2 維持 0.0 時，閥門 1 開再大也會被
    # 靜默乘成 0，而 config dump 看起來完全正常（POSE_LOSS_WEIGHT: 1.0）。
    # 原始 7 個 scheme run 與 stage2c_scheme4_trainable_fusion 都踩過這個坑。
    #
    # 現在改成單一旋鈕：**只有 LOSS.*_LOSS_WEIGHT 控制 pGT loss**。
    # THERMAL_* 仍可被解析（不讓舊 config 掛掉），但會在此處摺進對應的
    # *_LOSS_WEIGHT 後歸一化成 1.0，trainer 端的乘法已移除。因為遮罩恆為全 True，
    # 摺疊對每一個既有 config 都是逐位元等價的行為保留。
    _LEGACY_THERMAL_VALVES = (
        ('thermal_smpl_global_weight', 'THERMAL_SMPL_GLOBAL_WEIGHT', ('pose_loss_weight',)),
        ('thermal_shape_global_weight', 'THERMAL_SHAPE_GLOBAL_WEIGHT',
         ('beta_loss_weight', 'shape_loss_weight')),
        ('thermal_cam_global_weight', 'THERMAL_CAM_GLOBAL_WEIGHT', ('cam_loss_weight',)),
    )

    def _fold_legacy_thermal_global_weights(self, values, structured_cfg):
        loss_cfg = (structured_cfg or {}).get('LOSS')
        if not isinstance(loss_cfg, dict):
            loss_cfg = None
        folds = []

        for dest, yaml_key, targets in self._LEGACY_THERMAL_VALVES:
            raw = values.get(dest)
            # 一律歸一化成 1.0：trainer 已不再讀它，留著只是為了舊 dump 的相容性。
            values[dest] = 1.0
            if loss_cfg is not None and yaml_key in loss_cfg:
                loss_cfg[yaml_key] = 1.0
            if raw is None:
                continue  # 沒設 -> 本來就是單一旋鈕語意，不需要摺疊
            mult = float(raw)
            if mult == 1.0:
                continue  # 恆等，摺了也一樣
            for target in targets:
                before = values.get(target)
                before = 0.0 if before is None else float(before)
                after = before * mult
                values[target] = after
                target_yaml = FIELD_MAP.get(target, (None, target.upper()))[1]
                if loss_cfg is not None and target_yaml in loss_cfg:
                    loss_cfg[target_yaml] = after
                folds.append((yaml_key, mult, target_yaml, before, after))

        if folds:
            print('[DEPRECATED] LOSS.THERMAL_*_GLOBAL_WEIGHT 已停用（單一旋鈕遷移）。'
                  '其值已摺進對應的 LOSS.*_LOSS_WEIGHT：')
            for yaml_key, mult, target_yaml, before, after in folds:
                flag = '  <-- 監督被完全關閉' if after == 0.0 and before != 0.0 else ''
                print(f'[DEPRECATED]   {yaml_key}={mult:g}  x  {target_yaml}={before:g}'
                      f'  ->  {target_yaml}={after:g}{flag}')
            print('[DEPRECATED] 請改為直接設定 LOSS.*_LOSS_WEIGHT，並移除 THERMAL_*_GLOBAL_WEIGHT。')
        self._legacy_thermal_folds = folds

    def _apply_backward_compat_defaults(self, values, structured_cfg):
        # HRNet pretrained path:
        # - None means "let backbone choose default ckpt by crop_size".
        # - explicit YAML/CLI path still has priority.
        if values.get('hrnet_pretrained') in ['', None]:
            cfg_hrnet_pretrained = getattr(config, 'HRNET_PRETRAINED', None)
            values['hrnet_pretrained'] = (
                cfg_hrnet_pretrained
                if cfg_hrnet_pretrained not in ['', None]
                else None
            )

        # TokenHMR-style HRNet backbone input (center-crop inside encoder).
        # External crop-space stays at img_res x img_res.
        if values.get('hrnet_backbone_input_height') in [None, '']:
            values['hrnet_backbone_input_height'] = 256
        if values.get('hrnet_backbone_input_width') in [None, '']:
            values['hrnet_backbone_input_width'] = 192
        if values.get('hrnet_center_crop_input') is None:
            values['hrnet_center_crop_input'] = True

        if values.get('strict_vitpose') is None:
            values['strict_vitpose'] = True
        if values.get('vitpose_checkpoint') in [None, '']:
            values['vitpose_checkpoint'] = None  # unused with CORE_TYPE=hmr2
        if values.get('vitpose_input_height') is None:
            values['vitpose_input_height'] = 256
        if values.get('vitpose_input_width') is None:
            # Default to 192 to match the pretrained ViTPose-Huge COCO 256x192 grid.
            values['vitpose_input_width'] = 192
        if values.get('vitpose_center_crop_input') is None:
            values['vitpose_center_crop_input'] = True
        if values.get('vitpose_freeze') is None:
            values['vitpose_freeze'] = False
        if values.get('vitpose_pool') in [None, '']:
            values['vitpose_pool'] = 'mean'
        # HMR head defaults: keep legacy iterative MLP as default (Plan A).
        if values.get('head_type') in [None, '']:
            values['head_type'] = 'iterative_mlp'
        if values.get('decoder_dim') is None:
            values['decoder_dim'] = 1024
        if values.get('decoder_depth') is None:
            values['decoder_depth'] = 1
        if values.get('decoder_heads') is None:
            values['decoder_heads'] = 8
        if values.get('decoder_mlp_ratio') is None:
            values['decoder_mlp_ratio'] = 4.0
        if values.get('decoder_dropout') is None:
            values['decoder_dropout'] = 0.0
        # Camera representation default: keep legacy weak-perspective so any
        # YAML/CLI that doesn't set it behaves identically to before Step 3.
        if values.get('camera_rep') in [None, '']:
            values['camera_rep'] = 'weak_perspective'
        if values.get('full_persp_init_depth') is None:
            values['full_persp_init_depth'] = 5.0
        if values.get('full_persp_min_depth') is None:
            values['full_persp_min_depth'] = 1e-3
        if values.get('full_persp_tz_param') in [None, '']:
            # 'clamp' matches Step 3 behaviour so existing full-perspective
            # YAMLs without this key keep their old depth parametrisation.
            values['full_persp_tz_param'] = 'clamp'
        if values.get('use_bbox_token') is None:
            # step4to7_fi_v2/v3: bbox token removed from the stable mainline.
            # The field stays in FIELD_MAP for back-compat, but is forced to
            # False here.
            values['use_bbox_token'] = False
        elif bool(values.get('use_bbox_token')) is True:
            # step4to7_fi_v3 §4: refuse globally, not only on transformer_decoder.
            raise ValueError(
                'USE_BBOX_TOKEN=true is no longer supported in the stable '
                'mainline. Set USE_BBOX_TOKEN=false or remove it. Use '
                'configs/train/stage1a_hmr2_token_weak_h36m_warmup.yaml.'
            )
        if values.get('bbox_token_hidden') is None:
            values['bbox_token_hidden'] = 128

        # step4to7_fi_v3 §2 + §5: transformer_decoder mainline must use
        # ViTPose-Huge + VITPOSE_POOL=token. Reject at config-parse time so we
        # never get to a forward that hands a 2D [B, C] tensor to the
        # token-shaped decoder head.
        if str(values.get('head_type', '')) == 'transformer_decoder':
            if str(values.get('backbone', '')) != 'vitpose_huge':
                raise ValueError(
                    'HMR_HEAD_TYPE=transformer_decoder requires '
                    'BACKBONE=vitpose_huge. ResNet path is removed from the '
                    'stable Stage 1 mainline.'
                )
            if str(values.get('vitpose_pool', '')) != 'token':
                raise ValueError(
                    'HMR_HEAD_TYPE=transformer_decoder requires '
                    'VITPOSE_POOL=token. The stable mainline must use '
                    'spatial tokens and no global pooling.'
                )

        # step4to7_fi_v3 §6: full_perspective is no longer Stage 1 mainline.
        # Code path stays available for ablation, but warn loudly so it
        # doesn't get used by accident.
        if str(values.get('camera_rep', '')) == 'full_perspective':
            self._warn(
                'CAMERA_REP=full_perspective is experimental and not part of '
                'the stable Stage 1 mainline. Use weak_perspective for '
                'Stage 1A / Stage 1B.'
            )

        # CAMERA section default — diagonal pseudo-focal is the new convention,
        # but individual YAMLs may override.
        if values.get('focal_estimate_mode') in [None, '']:
            values['focal_estimate_mode'] = 'diagonal'

        # Differential LR fallback: None means "use single global LR" — the
        # trainer treats missing per-group LR as equal to args.lr, so existing
        # YAML / CLI configs that only set TRAIN.LR keep working unchanged.
        # We leave backbone_lr / head_lr / cam_head_lr as None here.
        # MU-HMR: model_type None means "plain HMR" (no wrapper); unifier_lr /
        # thermal_adapter_lr None falls back to args.lr in the trainer.
        if values.get('model_type') in ['', None]:
            values['model_type'] = None
        # unifier_lr / thermal_adapter_lr intentionally left as-is (None ok).
        if values.get('core_type') in ['', None]:
            values['core_type'] = 'vitpose'
        # hmr2_checkpoint intentionally left as-is (None ok).

        if values.get('rot_zero_prob') is None:
            values['rot_zero_prob'] = 0.6

        # Backward compatibility for old LOSS.KEYPOINT_LOSS_WEIGHT.
        old_kpt = self._cfg_value(structured_cfg, 'LOSS', 'KEYPOINT_LOSS_WEIGHT', 'keypoint_loss_weight')
        if values.get('keypoint_2d_loss_weight') is None:
            values['keypoint_2d_loss_weight'] = old_kpt if old_kpt is not None else 5.0
        if values.get('keypoint_3d_loss_weight') is None:
            values['keypoint_3d_loss_weight'] = old_kpt if old_kpt is not None else 5.0

        if values.get('cam_depth_loss_weight') is None:
            values['cam_depth_loss_weight'] = 1.0
        if values.get('cam_loss_weight') is None:
            values['cam_loss_weight'] = 0.0
        if values.get('cam_full_loss_weight') is None:
            values['cam_full_loss_weight'] = 0.0
        if values.get('supervision_mode') in [None, '']:
            values['supervision_mode'] = 'fit_or_gt'

        # ReFit-style train loop controls, fallback order:
        # VALIDATION.INTERVAL > TRAIN.VALID_STEP > LOGGING.TEST_STEPS
        if values.get('valid_step') is None:
            legacy_train_valid = self._cfg_value(structured_cfg, 'TRAIN', 'VALID_STEP', 'valid_step')
            if legacy_train_valid is not None:
                values['valid_step'] = legacy_train_valid
            else:
                legacy_test = self._cfg_value(structured_cfg, 'LOGGING', 'TEST_STEPS', 'test_steps')
                values['valid_step'] = legacy_test if legacy_test is not None else 0
        if values.get('save_step') is None:
            legacy_save = self._cfg_value(structured_cfg, 'LOGGING', 'CHECKPOINT_STEPS', 'checkpoint_steps')
            values['save_step'] = legacy_save if legacy_save is not None else 0
        if values.get('max_step') is None:
            values['max_step'] = 0
        if values.get('lr_schedule') in [None, '']:
            values['lr_schedule'] = 'constant'
        if values.get('warmup_steps') is None:
            values['warmup_steps'] = 0
        if values.get('optimizer') in [None, '']:
            values['optimizer'] = 'adamw'
        if values.get('loss_record_steps') is None:
            values['loss_record_steps'] = 200

        if values.get('validation_enabled') is None:
            values['validation_enabled'] = bool(int(values.get('valid_step', 0) or 0) > 0)
        if values.get('validation_dataset') in [None, '']:
            values['validation_dataset'] = '3dpw'
        if values.get('validation_metric') in [None, '']:
            values['validation_metric'] = 'MPJPE'
        if values.get('validation_save_best') is None:
            values['validation_save_best'] = True
        if values.get('validation_batch_size') is None:
            values['validation_batch_size'] = 0
        if values.get('validation_num_workers') is None:
            values['validation_num_workers'] = 4
        if values.get('validation_npz_path') in [None, '']:
            values['validation_npz_path'] = '/home/q36131134/HMR_models/SPIN_smplify/data/dataset_extras/3dpw_val.npz'
        if values.get('validation_sequence_path') in [None, '']:
            values['validation_sequence_path'] = '/home/q36131134/HMR_models/dataset/pw3d/sequenceFiles/validation'

        # Scheme A defaults
        if values.get('modality_embed_dim') is None:
            values['modality_embed_dim'] = 0
        if values.get('num_modalities') is None:
            values['num_modalities'] = 3
        if values.get('invert_prob') is None:
            values['invert_prob'] = 0.0
        if values.get('clahe_prob') is None:
            values['clahe_prob'] = 0.0
        if values.get('thermal_noise_prob') is None:
            values['thermal_noise_prob'] = 0.0
        if values.get('thermal_noise_std') is None:
            values['thermal_noise_std'] = 0.03
        if values.get('thermal_like_prob') is None:
            values['thermal_like_prob'] = 0.0
        if values.get('bridge_aug_as_thermal_modality') is None:
            values['bridge_aug_as_thermal_modality'] = False
        if values.get('use_pgt_weight') is None:
            values['use_pgt_weight'] = True
        # LOSS.THERMAL_*_GLOBAL_WEIGHT is DEPRECATED -- see
        # _fold_legacy_thermal_global_weights() for the single-knob migration.
        self._fold_legacy_thermal_global_weights(values, structured_cfg)
        if values.get('grad_clip_norm') is None:
            values['grad_clip_norm'] = 5.0
        if values.get('validation_otp2_npz') in [None, '']:
            values['validation_otp2_npz'] = None
        if values.get('validation_llvip_npz') in [None, '']:
            values['validation_llvip_npz'] = None

        # Plan 06.28 defaults
        if values.get('focal_policy') in [None, '']:
            values['focal_policy'] = 'real_else_fixed'
        if values.get('fixed_focal') is None:
            values['fixed_focal'] = 5000.0
        if values.get('log_focal_source') is None:
            values['log_focal_source'] = True
        if values.get('weak_cam_scale_min') is None:
            values['weak_cam_scale_min'] = 0.05
        if values.get('weak_cam_scale_max') is None:
            values['weak_cam_scale_max'] = 10.0
        if values.get('keypoint_2d_pixel_loss_weight') is None:
            values['keypoint_2d_pixel_loss_weight'] = 0.0
        # Default 1.0 preserves the historical quadratic/MSE-like behaviour for
        # every config that predates this option.
        if values.get('keypoint_2d_loss_beta') is None:
            values['keypoint_2d_loss_beta'] = 1.0
        if values.get('cam_full_loss_start_step') is None:
            values['cam_full_loss_start_step'] = 0
        if values.get('cam_full_loss_ramp_steps') is None:
            values['cam_full_loss_ramp_steps'] = 0
        if values.get('validation_datasets') in [None, '']:
            values['validation_datasets'] = None
        if values.get('validation_primary_metric') in [None, '']:
            values['validation_primary_metric'] = None
        if values.get('thermal_projection_weights') in [None, '']:
            values['thermal_projection_weights'] = None

    def _sync_structured_cfg_from_flat(self, structured_cfg, args):
        for dest, (section, key) in FIELD_MAP.items():
            if section not in structured_cfg or not isinstance(structured_cfg.get(section), dict):
                structured_cfg[section] = {}
            structured_cfg[section][key] = copy.deepcopy(getattr(args, dest))

    def _attach_structured_sections(self, args, structured):
        args.MODEL = config.as_namespace_tree(structured.get('MODEL', {}))
        args.DATASET = config.as_namespace_tree(structured.get('DATASET', {}))
        args.TRAIN = config.as_namespace_tree(structured.get('TRAIN', {}))
        args.SMPLIFY = config.as_namespace_tree(structured.get('SMPLIFY', {}))
        args.AUG = config.as_namespace_tree(structured.get('AUG', {}))
        args.LOSS = config.as_namespace_tree(structured.get('LOSS', {}))
        args.DATALOADER = config.as_namespace_tree(structured.get('DATALOADER', {}))
        args.LOGGING = config.as_namespace_tree(structured.get('LOGGING', {}))
        args.VALIDATION = config.as_namespace_tree(structured.get('VALIDATION', {}))
        args.DEBUG = config.as_namespace_tree(structured.get('DEBUG', {}))
        args.EXPERIMENT = config.as_namespace_tree(structured.get('EXPERIMENT', {}))
        args.IO = config.as_namespace_tree(structured.get('IO', {}))
        args.CAMERA = config.as_namespace_tree(structured.get('CAMERA', {}))
        # MU-HMR: expose MODALITY_DROPOUT as a namespace tree (read in trainer).
        args.MODALITY_DROPOUT = config.as_namespace_tree(structured.get('MODALITY_DROPOUT', {}))
        # Ablation diagnostics: step-0 baseline + delta / early-stop guard.
        args.BASELINE = config.as_namespace_tree(structured.get('BASELINE', {}))
        args._structured_config = copy.deepcopy(structured)

    def _attach_paths_section(self, args, paths_cfg):
        args.PATHS = config.as_namespace_tree(paths_cfg)
        args._paths_config = copy.deepcopy(paths_cfg)

    def _require_fields(self, args):
        missing = [k for k in REQUIRED_FIELDS if getattr(args, k, None) is None]
        if missing:
            raise ValueError(
                'Missing required train configuration fields. '
                f'Please complete configs/train/default.yaml (or override via CLI): {missing}'
            )

    def _normalize_dataset_options(self, args):
        dataset_list = list(getattr(args, 'dataset_list', []))
        dataset_partition = list(getattr(args, 'dataset_partition', []))
        if len(dataset_list) == 0:
            raise ValueError('dataset_list must not be empty')
        if len(dataset_list) != len(dataset_partition):
            raise ValueError('dataset_partition length must equal dataset_list length')

        part = np.asarray(dataset_partition, dtype=np.float64)
        if np.any(part <= 0):
            raise ValueError('dataset_partition values must be positive')
        if float(part.sum()) <= 0:
            raise ValueError('dataset_partition sum must be > 0')

        part = part / part.sum()
        args.dataset_list = dataset_list
        args.dataset_partition = part.tolist()

    def _validate_hrnet_options(self, args):
        backbone = getattr(args, 'backbone', 'resnet50')
        strict_hrnet = bool(getattr(args, 'strict_hrnet', True))
        hrnet_pretrained = getattr(args, 'hrnet_pretrained', None)
        if backbone == 'hrnet_w48':
            if hrnet_pretrained not in [None, ''] and not os.path.isfile(hrnet_pretrained):
                raise FileNotFoundError(f'HRNet pretrained checkpoint not found: {hrnet_pretrained}')
            if strict_hrnet and hrnet_pretrained in [None, '']:
                # Allowed: backbone wrapper will auto-select ckpt by crop_size.
                pass

    def _finalize_paths(self, args):
        base_log_dir = os.path.abspath(args.log_dir)
        if os.path.basename(os.path.normpath(base_log_dir)) != args.name:
            args.log_dir = os.path.join(base_log_dir, args.name)
        else:
            args.log_dir = base_log_dir
        args.summary_dir = os.path.join(args.log_dir, 'tensorboard')
        args.checkpoint_dir = os.path.join(args.log_dir, 'checkpoints')

        os.makedirs(args.log_dir, exist_ok=True)
        os.makedirs(args.summary_dir, exist_ok=True)
        os.makedirs(args.checkpoint_dir, exist_ok=True)

    def save_dump(self):
        os.makedirs(self.args.log_dir, exist_ok=True)
        train_dump = copy.deepcopy(getattr(self.args, '_structured_config', {}))
        paths_dump = copy.deepcopy(getattr(self.args, '_paths_config', {}))
        runtime_dump = {
            'SUMMARY_DIR': self.args.summary_dir,
            'CHECKPOINT_DIR': self.args.checkpoint_dir,
            'CONFIG_SOURCE': getattr(self.args, 'config_source', None),
            'PATHS_CONFIG_SOURCE': getattr(self.args, 'paths_config_source', None),
        }
        folds = getattr(self, '_legacy_thermal_folds', None)
        if folds:
            # train_config.yaml 存的是「摺疊後」的有效權重（可直接重跑）；
            # 這裡保留摺疊前的原值，避免 provenance 遺失。
            runtime_dump['LEGACY_THERMAL_GLOBAL_WEIGHT_FOLDS'] = [
                {'LEGACY_KEY': yk, 'LEGACY_VALUE': m,
                 'TARGET': ty, 'BEFORE': b, 'AFTER': a}
                for yk, m, ty, b, a in folds
            ]

        if yaml is None:
            raise ImportError('PyYAML is required to dump yaml configs. Please install with: pip install pyyaml')

        with open(os.path.join(self.args.log_dir, 'train_config.yaml'), 'w', encoding='utf-8') as f:
            yaml.safe_dump(train_dump, f, sort_keys=False, allow_unicode=False)
        with open(os.path.join(self.args.log_dir, 'paths_config.yaml'), 'w', encoding='utf-8') as f:
            yaml.safe_dump(paths_dump, f, sort_keys=False, allow_unicode=False)
        with open(os.path.join(self.args.log_dir, 'runtime.yaml'), 'w', encoding='utf-8') as f:
            yaml.safe_dump(runtime_dump, f, sort_keys=False, allow_unicode=False)

        if bool(getattr(self.args, 'dump_json_compat', False)):
            with open(os.path.join(self.args.log_dir, 'config.json'), 'w', encoding='utf-8') as f:
                json.dump(self._flatten_for_json(), f, indent=4)

    def _flatten_for_json(self):
        out = {}
        for k, v in vars(self.args).items():
            if k.startswith('_'):
                continue
            if isinstance(v, argparse.Namespace):
                continue
            if isinstance(v, (list, tuple, str, int, float, bool)) or v is None:
                out[k] = v
        return out
