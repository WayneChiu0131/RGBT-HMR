import torch
from torch.nn import functional as F
import numpy as np

"""
Useful geometric operations, e.g. Perspective projection and a differentiable Rodrigues formula
Parts of the code are taken from https://github.com/MandyMo/pytorch_HMR
"""
def batch_rodrigues(theta):
    """Convert axis-angle representation to rotation matrix.
    Args:
        theta: size = [B, 3]
    Returns:
        Rotation matrix corresponding to the quaternion -- size = [B, 3, 3]
    """
    l1norm = torch.norm(theta + 1e-8, p = 2, dim = 1)
    angle = torch.unsqueeze(l1norm, -1)
    normalized = torch.div(theta, angle)
    angle = angle * 0.5
    v_cos = torch.cos(angle)
    v_sin = torch.sin(angle)
    quat = torch.cat([v_cos, v_sin * normalized], dim = 1)
    return quat_to_rotmat(quat)

def quat_to_rotmat(quat):
    """Convert quaternion coefficients to rotation matrix.
    Args:
        quat: size = [B, 4] 4 <===>(w, x, y, z)
    Returns:
        Rotation matrix corresponding to the quaternion -- size = [B, 3, 3]
    """ 
    norm_quat = quat
    norm_quat = norm_quat/norm_quat.norm(p=2, dim=1, keepdim=True)
    w, x, y, z = norm_quat[:,0], norm_quat[:,1], norm_quat[:,2], norm_quat[:,3]

    B = quat.size(0)

    w2, x2, y2, z2 = w.pow(2), x.pow(2), y.pow(2), z.pow(2)
    wx, wy, wz = w*x, w*y, w*z
    xy, xz, yz = x*y, x*z, y*z

    rotMat = torch.stack([w2 + x2 - y2 - z2, 2*xy - 2*wz, 2*wy + 2*xz,
                          2*wz + 2*xy, w2 - x2 + y2 - z2, 2*yz - 2*wx,
                          2*xz - 2*wy, 2*wx + 2*yz, w2 - x2 - y2 + z2], dim=1).view(B, 3, 3)
    return rotMat    

def rot6d_to_rotmat(x):
    """Convert 6D rotation representation to 3x3 rotation matrix.
    Based on Zhou et al., "On the Continuity of Rotation Representations in Neural Networks", CVPR 2019
    Input:
        (B,6) Batch of 6-D rotation representations
    Output:
        (B,3,3) Batch of corresponding rotation matrices
    """
    x = x.view(-1,3,2)
    a1 = x[:, :, 0]
    a2 = x[:, :, 1]
    b1 = F.normalize(a1)
    b2 = F.normalize(a2 - torch.einsum('bi,bi->b', b1, a2).unsqueeze(-1) * b1)
    b3 = torch.cross(b1, b2)
    return torch.stack((b1, b2, b3), dim=-1)
     
def perspective_projection(points, rotation, translation,
                           focal_length, camera_center):
    """
    This function computes the perspective projection of a set of points.
    Input:
        points (bs, N, 3): 3D points
        rotation (bs, 3, 3): Camera rotation
        translation (bs, 3): Camera translation
        focal_length (bs,) or scalar: Focal length
        camera_center (bs, 2): Camera center
    """
    batch_size = points.shape[0]
    K = torch.zeros([batch_size, 3, 3], device=points.device)
    K[:,0,0] = focal_length
    K[:,1,1] = focal_length
    K[:,2,2] = 1.
    K[:,:-1, -1] = camera_center

    # Transform points
    points = torch.einsum('bij,bkj->bki', rotation, points)
    points = points + translation.unsqueeze(1)

    # Apply perspective distortion
    projected_points = points / points[:,:,-1].unsqueeze(-1)

    # Apply camera intrinsics
    projected_points = torch.einsum('bij,bkj->bki', K, projected_points)

    return projected_points[:, :, :-1]


def estimate_translation_np(S, joints_2d, joints_conf, focal_length=5000, img_size=224):
    """Find camera translation that brings 3D joints S closest to 2D the corresponding joints_2d.
    Input:
        S: (25, 3) 3D joint locations
        joints: (25, 3) 2D joint locations and confidence
    Returns:
        (3,) camera translation vector
    """

    num_joints = S.shape[0]
    # focal length
    f = np.array([focal_length,focal_length])
    # optical center
    center = np.array([img_size/2., img_size/2.])

    # transformations
    Z = np.reshape(np.tile(S[:,2],(2,1)).T,-1)
    XY = np.reshape(S[:,0:2],-1)
    O = np.tile(center,num_joints)
    F = np.tile(f,num_joints)
    weight2 = np.reshape(np.tile(np.sqrt(joints_conf),(2,1)).T,-1)

    # least squares
    Q = np.array([F*np.tile(np.array([1,0]),num_joints), F*np.tile(np.array([0,1]),num_joints), O-np.reshape(joints_2d,-1)]).T
    c = (np.reshape(joints_2d,-1)-O)*Z - F*XY

    # weighted least squares
    W = np.diagflat(weight2)
    Q = np.dot(W,Q)
    c = np.dot(W,c)

    # square matrix
    A = np.dot(Q.T,Q)
    b = np.dot(Q.T,c)

    # solution
    trans = np.linalg.solve(A, b)

    return trans


def estimate_translation(S, joints_2d, focal_length=5000., img_size=224.):
    """Find camera translation that brings 3D joints S closest to 2D the corresponding joints_2d.
    Input:
        S: (B, 49, 3) 3D joint locations
        joints: (B, 49, 3) 2D joint locations and confidence
    Returns:
        (B, 3) camera translation vectors
    """

    device = S.device
    # Use only joints 25:49 (GT joints)
    S = S[:, 25:, :].cpu().numpy()
    joints_2d = joints_2d[:, 25:, :].cpu().numpy()
    joints_conf = joints_2d[:, :, -1]
    joints_2d = joints_2d[:, :, :-1]
    trans = np.zeros((S.shape[0], 3), dtype=np.float32)
    # Find the translation for each example in the batch
    for i in range(S.shape[0]):
        S_i = S[i]
        joints_i = joints_2d[i]
        conf_i = joints_conf[i]
        trans[i] = estimate_translation_np(S_i, joints_i, conf_i, focal_length=focal_length, img_size=img_size)
    return torch.from_numpy(trans).to(device)


def estimate_focal_length(img_h, img_w, mode='diagonal'):
    """Pseudo-focal estimate for samples without real camera intrinsics.

    mode:
      'diagonal' (default): sqrt(W^2 + H^2)   — used by demo / new training pipeline
      'max':                max(W, H)         — legacy, kept for backward compat
      'height':             H                 — legacy alternative
    Accepts scalar / numpy / torch input; returns float when input is scalar,
    otherwise returns a torch tensor matching the input dtype/device.
    """
    is_scalar = not (torch.is_tensor(img_h) or torch.is_tensor(img_w) or
                     hasattr(img_h, '__len__') or hasattr(img_w, '__len__'))

    # Resolve a target device: prefer existing tensor's device for consistency.
    target_device = None
    if torch.is_tensor(img_h):
        target_device = img_h.device
    elif torch.is_tensor(img_w):
        target_device = img_w.device

    if not torch.is_tensor(img_h):
        img_h = torch.tensor(img_h, dtype=torch.float32,
                             device=target_device if target_device is not None else 'cpu')
    if not torch.is_tensor(img_w):
        img_w = torch.tensor(img_w, dtype=torch.float32,
                             device=target_device if target_device is not None else 'cpu')

    # Ensure both tensors live on the same device (prefer img_h's device).
    if img_w.device != img_h.device:
        img_w = img_w.to(img_h.device)

    img_h = img_h.float()
    img_w = img_w.float()

    mode = str(mode).lower()
    if mode == 'diagonal':
        out = torch.sqrt(img_w ** 2 + img_h ** 2)
    elif mode == 'max':
        out = torch.maximum(img_w, img_h)
    elif mode == 'height':
        out = img_h
    else:
        raise ValueError(
            f"Unsupported focal_estimate_mode='{mode}'. "
            "Expected one of: ['diagonal', 'max', 'height']"
        )

    if is_scalar:
        return float(out.item())
    return out


def bbox_info_from_center_scale(center, scale, img_focal, img_center):
    b = scale * 200.0
    info = torch.stack([
        center[:, 0] - img_center[:, 0],
        center[:, 1] - img_center[:, 1],
        b
    ], dim=-1)
    info[:, :2] = info[:, :2] / img_focal.unsqueeze(-1) * 2.8
    info[:, 2] = (info[:, 2] - 0.24 * img_focal) / (0.06 * img_focal + 1e-9)
    return info


def cam_crop2full(crop_cam, center, scale, img_focal, img_center):
    b = scale * 200.0
    s = crop_cam[:, 0]
    tx = crop_cam[:, 1]
    ty = crop_cam[:, 2]

    bs = b * s + 1e-9
    tz_full = 2.0 * img_focal / bs
    tx_full = tx + 2.0 * (center[:, 0] - img_center[:, 0]) / bs
    ty_full = ty + 2.0 * (center[:, 1] - img_center[:, 1]) / bs

    return torch.stack([tx_full, ty_full, tz_full], dim=-1)


def cam_crop2perspective_crop(crop_cam, focal_length, crop_res):
    s = crop_cam[:, 0]
    tx = crop_cam[:, 1]
    ty = crop_cam[:, 2]
    tz = 2.0 * focal_length / (crop_res * s + 1e-9)
    return torch.stack([tx, ty, tz], dim=-1)


def cam_perspective_crop2cropcam(cam_t, focal_length, crop_res):
    tz = cam_t[:, 2]
    s = 2.0 * focal_length / (crop_res * tz + 1e-9)
    tx = cam_t[:, 0]
    ty = cam_t[:, 1]
    return torch.stack([s, tx, ty], dim=-1)


def get_transform_torch(center, scale, crop_res, rot=None, device=None, dtype=torch.float32):
    """
    Torch version of the affine transform used by utils.imutils.transform / crop.
    center: [B,2]
    scale: [B]
    crop_res: int
    rot: [B] degrees
    return: [B,3,3]
    """
    if device is None:
        device = center.device

    B = center.shape[0]
    h = 200.0 * scale

    t = torch.zeros(B, 3, 3, device=device, dtype=dtype)
    t[:, 0, 0] = crop_res / h
    t[:, 1, 1] = crop_res / h
    t[:, 0, 2] = crop_res * (-center[:, 0] / h + 0.5)
    t[:, 1, 2] = crop_res * (-center[:, 1] / h + 0.5)
    t[:, 2, 2] = 1.0

    if rot is not None:
        rot = -rot
        rot_rad = rot * np.pi / 180.0
        sn = torch.sin(rot_rad)
        cs = torch.cos(rot_rad)

        rot_mat = torch.zeros(B, 3, 3, device=device, dtype=dtype)
        rot_mat[:, 0, 0] = cs
        rot_mat[:, 0, 1] = -sn
        rot_mat[:, 1, 0] = sn
        rot_mat[:, 1, 1] = cs
        rot_mat[:, 2, 2] = 1.0

        t_mat = torch.eye(3, device=device, dtype=dtype).unsqueeze(0).repeat(B, 1, 1)
        t_mat[:, 0, 2] = -crop_res / 2.0
        t_mat[:, 1, 2] = -crop_res / 2.0

        t_inv = torch.eye(3, device=device, dtype=dtype).unsqueeze(0).repeat(B, 1, 1)
        t_inv[:, 0, 2] = crop_res / 2.0
        t_inv[:, 1, 2] = crop_res / 2.0

        t = torch.bmm(t_inv, torch.bmm(rot_mat, torch.bmm(t_mat, t)))

    return t


def transform_points_full_to_crop(points2d_full, center, scale, crop_res, rot_angle=None, is_flipped=None):
    """
    Map full-image 2D points into augmented crop pixel coordinates.
    points2d_full: [B,N,2]
    center: [B,2]
    scale: [B]
    rot_angle: [B]
    is_flipped: [B]
    return: [B,N,2]
    """
    B, N, _ = points2d_full.shape
    device = points2d_full.device
    dtype = points2d_full.dtype

    T = get_transform_torch(center, scale, crop_res, rot=rot_angle, device=device, dtype=dtype)

    ones = torch.ones(B, N, 1, device=device, dtype=dtype)
    pts_h = torch.cat([points2d_full + 1.0, ones], dim=-1)   # match old transform(kp+1,...)
    pts_crop = torch.bmm(pts_h, T.transpose(1, 2))[..., :2]

    if is_flipped is not None:
        is_flipped = is_flipped.bool()
        if is_flipped.any():
            pts_crop_flipped = pts_crop.clone()
            # Keep x-flip exactly consistent with BaseDataset.j2d_processing + flip_kp():
            # normalized x' = -x  <=>  pixel x' = crop_res - x
            pts_crop_flipped[is_flipped, :, 0] = float(crop_res) - pts_crop_flipped[is_flipped, :, 0]
            pts_crop = pts_crop_flipped

    return pts_crop


def project_points_crop_full(points, crop_cam, center, scale, img_focal, img_center, crop_res,
                             rotation=None, rot_angle=None, is_flipped=None):
    full_trans = cam_crop2full(crop_cam, center, scale, img_focal, img_center)

    if rotation is None:
        batch_size = points.shape[0]
        rotation = torch.eye(3, device=points.device).unsqueeze(0).expand(batch_size, -1, -1)

    points2d_full = perspective_projection(
        points,
        rotation=rotation,
        translation=full_trans,
        focal_length=img_focal,
        camera_center=img_center
    )

    points2d_crop = transform_points_full_to_crop(
        points2d_full,
        center=center,
        scale=scale,
        crop_res=crop_res,
        rot_angle=rot_angle,
        is_flipped=is_flipped
    )

    return points2d_full, points2d_crop, full_trans


# =====================================================================
# Full-image perspective camera functions (CLIFF-style)
# =====================================================================

def build_full_cam_init_from_bbox(center, scale, img_focal, img_center):
    """Build a reasonable full-image perspective camera init from bbox info.

    Args:
        center: [B, 2] bbox center in full-image pixels
        scale: [B] bbox scale (bbox_size = scale * 200)
        img_focal: [B] full-image focal length
        img_center: [B, 2] full-image optical center
    Returns:
        [B, 3] full-image camera translation [Tx, Ty, Tz]
    """
    b = scale * 200.0
    Tz = 2.0 * img_focal / (b + 1e-9)
    Tx = 2.0 * (center[:, 0] - img_center[:, 0]) / (b + 1e-9)
    Ty = 2.0 * (center[:, 1] - img_center[:, 1]) / (b + 1e-9)
    return torch.stack([Tx, Ty, Tz], dim=-1)


def transform_points_crop_to_full(points2d_crop, center, scale, crop_res, rot_angle=None, is_flipped=None):
    """Inverse of transform_points_full_to_crop: map augmented crop pixel 2D
    points back to full-image 2D pixel coordinates.

    Args:
        points2d_crop: [B, N, 2]
        center: [B, 2]
        scale: [B]
        crop_res: int
        rot_angle: [B] degrees (optional)
        is_flipped: [B] bool (optional)
    Returns:
        [B, N, 2] points in full-image pixel space
    """
    B, N, _ = points2d_crop.shape
    device = points2d_crop.device
    dtype = points2d_crop.dtype

    pts = points2d_crop.clone()

    if is_flipped is not None:
        is_flipped = is_flipped.bool()
        if is_flipped.any():
            pts_unflipped = pts.clone()
            pts_unflipped[is_flipped, :, 0] = float(crop_res) - pts_unflipped[is_flipped, :, 0]
            pts = pts_unflipped

    T = get_transform_torch(center, scale, crop_res, rot=rot_angle, device=device, dtype=dtype)
    T_inv = torch.inverse(T)

    ones = torch.ones(B, N, 1, device=device, dtype=dtype)
    pts_h = torch.cat([pts, ones], dim=-1)
    pts_full_h = torch.bmm(pts_h, T_inv.transpose(1, 2))[..., :2]

    pts_full = pts_full_h - 1.0

    return pts_full


def estimate_translation_np_perspective(S, joints_2d, joints_conf, focal_length, camera_center):
    """Find camera translation via weighted least squares with per-sample
    focal length and optical center (full-image perspective).

    Args:
        S: (K, 3) 3D joint locations
        joints_2d: (K, 2) 2D joint locations in full-image pixels
        joints_conf: (K,) confidence
        focal_length: scalar focal length for this sample
        camera_center: (2,) optical center [cx, cy]
    Returns:
        (3,) camera translation vector
    """
    num_joints = S.shape[0]
    f = np.array([focal_length, focal_length])
    center = np.array(camera_center)

    Z = np.reshape(np.tile(S[:, 2], (2, 1)).T, -1)
    XY = np.reshape(S[:, 0:2], -1)
    O = np.tile(center, num_joints)
    F = np.tile(f, num_joints)
    weight2 = np.reshape(np.tile(np.sqrt(joints_conf), (2, 1)).T, -1)

    Q = np.array([
        F * np.tile(np.array([1, 0]), num_joints),
        F * np.tile(np.array([0, 1]), num_joints),
        O - np.reshape(joints_2d, -1)
    ]).T
    c = (np.reshape(joints_2d, -1) - O) * Z - F * XY

    W = np.diagflat(weight2)
    Q = np.dot(W, Q)
    c = np.dot(W, c)

    A = np.dot(Q.T, Q)
    b = np.dot(Q.T, c)

    trans = np.linalg.solve(A, b)
    return trans


def estimate_translation_fullimg(S, joints_2d_crop, center, scale, img_focal, img_center,
                                 crop_res, rot_angle=None, is_flipped=None):
    """Batched full-image camera translation estimation from crop-space 2D keypoints.

    Args:
        S: [B, 49, 3] model joints (3D)
        joints_2d_crop: [B, 49, 3] crop pixel coords + confidence
        center: [B, 2] bbox center
        scale: [B] bbox scale
        img_focal: [B] per-sample focal length
        img_center: [B, 2] per-sample optical center
        crop_res: int
        rot_angle: [B] (optional)
        is_flipped: [B] (optional)
    Returns:
        [B, 3] full-image camera translation
    """
    device = S.device

    S_gt = S[:, 25:, :]
    joints_2d_crop_sel = joints_2d_crop[:, 25:, :]
    joints_conf = joints_2d_crop_sel[:, :, -1]
    joints_2d_xy = joints_2d_crop_sel[:, :, :2]

    joints_2d_full = transform_points_crop_to_full(
        joints_2d_xy,
        center=center,
        scale=scale,
        crop_res=crop_res,
        rot_angle=rot_angle,
        is_flipped=is_flipped,
    )

    # Pure-GPU vectorized batch equivalent of estimate_translation_np_perspective:
    # for each sample build (2K,3) Q and (2K,) c, weight rows by sqrt(conf),
    # solve (Q^T W^T W Q) t = Q^T W^T W c via batched linalg.solve.
    B, K, _ = S_gt.shape
    f = img_focal.view(-1).float()                           # (B,)
    cx = img_center[:, 0].float()                            # (B,)
    cy = img_center[:, 1].float()                            # (B,)
    x = S_gt[:, :, 0].float()                                # (B, K)
    y = S_gt[:, :, 1].float()                                # (B, K)
    z = S_gt[:, :, 2].float()                                # (B, K)
    u = joints_2d_full[:, :, 0].float()                      # (B, K)
    v = joints_2d_full[:, :, 1].float()                      # (B, K)
    conf = joints_conf.float()                               # (B, K)

    cx_b = cx.view(B, 1)
    cy_b = cy.view(B, 1)
    f_b = f.view(B, 1)

    two_k = 2 * K
    Q = torch.zeros(B, two_k, 3, device=device, dtype=torch.float32)
    Q[:, 0::2, 0] = f_b
    Q[:, 1::2, 1] = f_b
    Q[:, 0::2, 2] = cx_b - u
    Q[:, 1::2, 2] = cy_b - v

    c_vec = torch.zeros(B, two_k, device=device, dtype=torch.float32)
    c_vec[:, 0::2] = (u - cx_b) * z - f_b * x
    c_vec[:, 1::2] = (v - cy_b) * z - f_b * y

    weight2 = torch.sqrt(conf.clamp(min=0.0))                # (B, K)
    weight2_full = weight2.repeat_interleave(2, dim=1)       # (B, 2K)

    Q_w = (Q * weight2_full.unsqueeze(-1)).double()
    c_w = (c_vec * weight2_full).double()

    A = torch.bmm(Q_w.transpose(1, 2), Q_w)
    b_rhs = torch.bmm(Q_w.transpose(1, 2), c_w.unsqueeze(-1))
    trans = torch.linalg.solve(A, b_rhs).squeeze(-1).float()
    return trans


def project_points_full_to_crop(points, cam_t, center, scale, img_focal, img_center, crop_res,
                                rotation=None, rot_angle=None, is_flipped=None):
    """Project 3D points using full-image perspective camera, then map to crop.

    Args:
        points: [B, N, 3]
        cam_t: [B, 3] full-image perspective camera translation [Tx, Ty, Tz]
        center, scale, img_focal, img_center: bbox / intrinsics info
        crop_res: int
        rotation: [B, 3, 3] (optional, defaults to identity)
        rot_angle: [B] (optional)
        is_flipped: [B] (optional)
    Returns:
        points2d_full: [B, N, 2]
        points2d_crop: [B, N, 2]
        cam_t: [B, 3] (pass-through)
    """
    if rotation is None:
        batch_size = points.shape[0]
        rotation = torch.eye(3, device=points.device).unsqueeze(0).expand(batch_size, -1, -1)

    points2d_full = perspective_projection(
        points,
        rotation=rotation,
        translation=cam_t,
        focal_length=img_focal,
        camera_center=img_center,
    )

    points2d_crop = transform_points_full_to_crop(
        points2d_full,
        center=center,
        scale=scale,
        crop_res=crop_res,
        rot_angle=rot_angle,
        is_flipped=is_flipped,
    )

    return points2d_full, points2d_crop, cam_t


def cam_full2crop_perspective(full_cam, center, scale, img_focal, img_center,
                              focal_length_crop, crop_res):
    """Convert full-image camera translation to crop-space perspective camera
    translation for renderer visualization only.

    Args:
        full_cam: [B, 3] full-image camera [Tx, Ty, Tz]
        center: [B, 2]
        scale: [B]
        img_focal: [B]
        img_center: [B, 2]
        focal_length_crop: scalar crop focal length
        crop_res: int
    Returns:
        [B, 3] crop-space perspective camera translation [tx_crop, ty_crop, tz_crop]
    """
    b = scale * 200.0
    s = 2.0 * img_focal / (b * full_cam[:, 2] + 1e-9)
    tx_crop = full_cam[:, 0] - (center[:, 0] - img_center[:, 0]) * full_cam[:, 2] / (img_focal + 1e-9)
    ty_crop = full_cam[:, 1] - (center[:, 1] - img_center[:, 1]) * full_cam[:, 2] / (img_focal + 1e-9)
    tz_crop = 2.0 * focal_length_crop / (crop_res * s + 1e-9)
    return torch.stack([tx_crop, ty_crop, tz_crop], dim=-1)


# ---------------------------------------------------------------------------
# Plan 06.28 — Focal policy, weak-scale clamp, camera outlier ratio
# ---------------------------------------------------------------------------

FOCAL_SOURCE_REAL = 0
FOCAL_SOURCE_FIXED = 1
FOCAL_SOURCE_DIAGONAL = 2
FOCAL_SOURCE_MOGE = 3


def resolve_focal_policy(policy, real_focal, has_intrinsics, img_h, img_w,
                         moge_focal=None, fixed_focal=5000.0):
    """Resolve a per-sample focal length tensor and a "source code" tensor.

    Policies (all return both pixel-space focal lengths and an int8 source code
    tensor so callers can log how each sample's focal was obtained):

    - ``real_else_fixed``      : real intrinsics if available else ``fixed_focal``
    - ``real_else_diagonal``   : real intrinsics if available else image diagonal
    - ``real_else_moge``       : real intrinsics if available else MoGe estimate
                                  (falls back to ``fixed_focal`` when MoGe missing)
    - ``fixed_all``            : always ``fixed_focal``
    - ``diagonal_all``         : always image diagonal

    Args:
        policy: str
        real_focal: [B] tensor of real focal lengths (zeros allowed where missing)
        has_intrinsics: [B] bool/byte tensor, 1 where real_focal is trustworthy
        img_h, img_w: [B] tensors of image height/width in pixels
        moge_focal: optional [B] tensor of MoGe focal estimates
        fixed_focal: float, default 5000.0
    Returns:
        focal: [B] float tensor
        source: [B] int8 tensor encoding which policy branch fired per sample
    """
    if not torch.is_tensor(real_focal):
        raise TypeError("real_focal must be a tensor")
    device = real_focal.device
    dtype = real_focal.dtype if real_focal.is_floating_point() else torch.float32
    real_focal = real_focal.to(dtype=dtype)

    if not torch.is_tensor(has_intrinsics):
        has_intrinsics = torch.tensor(has_intrinsics, device=device)
    has_mask = has_intrinsics.to(device=device).bool()

    img_h_t = img_h if torch.is_tensor(img_h) else torch.tensor(img_h, device=device)
    img_w_t = img_w if torch.is_tensor(img_w) else torch.tensor(img_w, device=device)
    img_h_t = img_h_t.to(device=device, dtype=dtype)
    img_w_t = img_w_t.to(device=device, dtype=dtype)
    diag = torch.sqrt(img_w_t ** 2 + img_h_t ** 2)
    fixed_t = torch.full_like(diag, float(fixed_focal))

    policy_l = str(policy).lower()
    source = torch.zeros_like(diag, dtype=torch.int8)

    if policy_l == 'fixed_all':
        focal = fixed_t
        source.fill_(FOCAL_SOURCE_FIXED)
        return focal, source

    if policy_l == 'diagonal_all':
        focal = diag
        source.fill_(FOCAL_SOURCE_DIAGONAL)
        return focal, source

    # remaining policies always honor real intrinsics first
    if policy_l == 'real_else_fixed':
        fallback = fixed_t
        fallback_code = FOCAL_SOURCE_FIXED
    elif policy_l == 'real_else_diagonal':
        fallback = diag
        fallback_code = FOCAL_SOURCE_DIAGONAL
    elif policy_l == 'real_else_moge':
        if moge_focal is not None and torch.is_tensor(moge_focal):
            moge_t = moge_focal.to(device=device, dtype=dtype)
            moge_valid = (moge_t > 1e-3)
            fallback = torch.where(moge_valid, moge_t, fixed_t)
            fallback_code_t = torch.where(
                moge_valid,
                torch.full_like(source, FOCAL_SOURCE_MOGE),
                torch.full_like(source, FOCAL_SOURCE_FIXED),
            )
        else:
            fallback = fixed_t
            fallback_code_t = torch.full_like(source, FOCAL_SOURCE_FIXED)
        focal = torch.where(has_mask, real_focal, fallback)
        source = torch.where(has_mask, torch.full_like(source, FOCAL_SOURCE_REAL),
                             fallback_code_t)
        return focal, source
    else:
        raise ValueError(
            f"Unsupported focal_policy='{policy}'. Expected one of: "
            "real_else_fixed | real_else_diagonal | real_else_moge | "
            "fixed_all | diagonal_all"
        )

    focal = torch.where(has_mask, real_focal, fallback)
    source = torch.where(
        has_mask,
        torch.full_like(source, FOCAL_SOURCE_REAL),
        torch.full_like(source, fallback_code),
    )
    return focal, source


def clamp_weak_scale(s, scale_min=0.05, scale_max=10.0):
    """Clamp weak-perspective scale ``s`` to ``[scale_min, scale_max]``.

    Returns the clamped tensor plus a bool mask flagging which samples were
    clamped so the trainer can report a `weak_scale_clamp_ratio` for telemetry.
    """
    if not torch.is_tensor(s):
        s = torch.tensor(s, dtype=torch.float32)
    clamp_mask = (s < scale_min) | (s > scale_max)
    s_clamped = torch.clamp(s, min=float(scale_min), max=float(scale_max))
    return s_clamped, clamp_mask


def compute_camera_outlier_ratio(tz, min_z=0.5, max_z=50.0):
    """Fraction of camera Tz values outside ``[min_z, max_z]`` (sane meter range).

    Returns a float in [0, 1].
    """
    if not torch.is_tensor(tz):
        tz = torch.tensor(tz, dtype=torch.float32)
    if tz.numel() == 0:
        return 0.0
    outlier = (tz < float(min_z)) | (tz > float(max_z))
    return float(outlier.float().mean().item())
