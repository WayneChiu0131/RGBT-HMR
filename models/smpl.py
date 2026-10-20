import torch
import numpy as np

# Compatibility shim for legacy chumpy usage inside some SMPL/SMPLX stacks.
if not hasattr(np, 'bool'):
    np.bool = bool  # type: ignore[attr-defined]
if not hasattr(np, 'int'):
    np.int = int  # type: ignore[attr-defined]
if not hasattr(np, 'float'):
    np.float = float  # type: ignore[attr-defined]
if not hasattr(np, 'complex'):
    np.complex = complex  # type: ignore[attr-defined]
if not hasattr(np, 'object'):
    np.object = object  # type: ignore[attr-defined]
if not hasattr(np, 'str'):
    np.str = str  # type: ignore[attr-defined]
if not hasattr(np, 'unicode'):
    np.unicode = str  # type: ignore[attr-defined]

import smplx
from smplx import SMPL as _SMPL
from smplx.lbs import vertices2joints
try:
    from smplx.body_models import ModelOutput
except ImportError:
    from smplx.utils import SMPLOutput as ModelOutput

import config
import constants

class SMPL(_SMPL):
    """ Extension of the official SMPL implementation to support more joints """

    def __init__(self, *args, **kwargs):
        super(SMPL, self).__init__(*args, **kwargs)
        joints = [constants.JOINT_MAP[i] for i in constants.JOINT_NAMES]
        J_regressor_extra = np.load(config.JOINT_REGRESSOR_TRAIN_EXTRA)
        self.register_buffer('J_regressor_extra', torch.tensor(J_regressor_extra, dtype=torch.float32))
        self.joint_map = torch.tensor(joints, dtype=torch.long)
        J_regressor_h36m = np.load(config.JOINT_REGRESSOR_H36M)
        self.register_buffer('J_regressor_h36m', torch.tensor(J_regressor_h36m, dtype=torch.float32))

    def get_h36m_joints(self, vertices):
        return vertices2joints(self.J_regressor_h36m, vertices)

    def forward(self, *args, **kwargs):
        kwargs['get_skin'] = True
        smpl_output = super(SMPL, self).forward(*args, **kwargs)
        extra_joints = vertices2joints(self.J_regressor_extra, smpl_output.vertices)
        joints = torch.cat([smpl_output.joints, extra_joints], dim=1)
        joints = joints[:, self.joint_map, :]
        output = ModelOutput(vertices=smpl_output.vertices,
                             global_orient=smpl_output.global_orient,
                             body_pose=smpl_output.body_pose,
                             joints=joints,
                             betas=smpl_output.betas,
                             full_pose=smpl_output.full_pose)
        return output
