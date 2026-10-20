"""
This file contains definitions of useful data structures and the paths
for the datasets and data files necessary to run the code.
Things you need to change: *_ROOT that indicate the path to each dataset
"""
import os
from os.path import join, exists
import numpy as np
import json
import copy
import argparse

try:
    import yaml
except Exception:
    yaml = None


def _require_yaml():
    if yaml is None:
        raise ImportError(
            "PyYAML is required for YAML config support. "
            "Please install with: pip install pyyaml"
        )


def load_yaml_config(path):
    _require_yaml()
    with open(path, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f)
    if data is None:
        data = {}
    if not isinstance(data, dict):
        raise ValueError(f"YAML config root must be a dict: {path}")
    return data


def load_any_config(path):
    if path is None:
        return {}
    ext = os.path.splitext(path)[1].lower()
    if ext in [".yaml", ".yml"]:
        return load_yaml_config(path)
    if ext == ".json":
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        if data is None:
            data = {}
        if not isinstance(data, dict):
            raise ValueError(f"JSON config root must be a dict: {path}")
        return data
    raise ValueError(f"Unsupported config extension: {path}")


def merge_config_with_defaults(base_cfg, override_cfg):
    base = copy.deepcopy(base_cfg) if isinstance(base_cfg, dict) else {}
    override = override_cfg if isinstance(override_cfg, dict) else {}
    for k, v in override.items():
        if isinstance(v, dict) and isinstance(base.get(k), dict):
            base[k] = merge_config_with_defaults(base[k], v)
        else:
            base[k] = copy.deepcopy(v)
    return base


def get_nested(cfg, path, default=None):
    if not isinstance(cfg, dict):
        return default
    cur = cfg
    for key in path.split("."):
        if not isinstance(cur, dict):
            return default
        if key not in cur:
            return default
        cur = cur[key]
    return cur


def as_namespace_tree(d):
    if isinstance(d, dict):
        return argparse.Namespace(**{k: as_namespace_tree(v) for k, v in d.items()})
    if isinstance(d, list):
        return [as_namespace_tree(x) for x in d]
    return d


def _first_existing_path(*candidates):
    for p in candidates:
        if exists(p):
            return p
    return candidates[0]

LSP_ROOT = '/home/q36131134/HMR_models/smplify_fixed/images/lsp'
LSP_ORIGINAL_ROOT = '/home/q36131134/HMR_models/smplify_fixed/images/lsp'

H36M_ROOT = '/home/q36131134/HMR_models/dataset/h36m'
H36M_TRAIN_ROOT = '/home/q36131134/HMR_models/dataset/h36m-train_extracted/h36m-train'
H36M_VAL_ROOT = '/home/q36131134/HMR_models/dataset/h36m-train_extracted/h36m-val'
LSPET_ROOT = '/home/q36131134/HMR_models/dataset/lspet'
MPII_ROOT = '/home/q36131134/HMR_models/dataset/mpii'
COCO_ROOT = '/home/q36131134/HMR_models/dataset/COCO'
MPI_INF_3DHP_ROOT = '/home/q36131134/HMR_models/dataset/mpi_inf_3dhp'
PW3D_ROOT = '/home/q36131134/HMR_models/dataset/pw3d'
AGORA_ROOT = '/home/q36131134/HMR_models/dataset/agora'
UPI_S1H_ROOT = ''
EMDB_ROOT = '/home/q36131134/HMR_models/dataset/EMDB'
LLVIP_ROOT = '/home/q36131134/HMR_models/dataset/infrared/LLVIP'
# MU-HMR: LLVIP visible (RGB) partner images, pixel-aligned with the infrared
# frames used by the pGT NPZ. Mapped via stripping the 'infrared/' prefix from
# the thermal imgname_rel. Verified byte-identical infrared + 1024x1280 align.
LLVIP_VISIBLE_ROOT = '/home/q36131134/HMR_models/dataset/infrared/LLVIP/visible'
OTP2_DATASET_ROOT = '/home/q36131134/HMR_models/dataset/infrared/otp2_dataset'
OTP2_ROOT = '/home/q36131134/HMR_models/dataset/infrared/otp2_dataset/train'

# Output folder to save test/train npz files
DATASET_NPZ_PATH = 'data/dataset_extras'
# Mapping file:
# key   = imgname stored inside ReFit h36m_train.npz
# value = relative path under H36M_ROOT that points to your real local image
# Example:
# "S1_Directions.54138969_000001.jpg":
#     "images/s_01_act_02_subact_01_ca_01/s_01_act_02_subact_01_ca_01_000001.jpg"
H36M_REFIT_REMAP_FILE = _first_existing_path(
    join(DATASET_NPZ_PATH, 'h36m_refit_imgname_to_local.json'),
    os.path.normpath(join(H36M_ROOT, '..', 'h36m_flat', 'h36m_refit_imgname_to_local.json')),
)

# MoGe-estimated focal length JSON files, keyed by dataset name.
# Each JSON maps imgname (same format as NPZ 'imgname') -> {'img_focal': float, 'img_center': [cx, cy]}.
# Only needed for datasets without real camera intrinsics in their NPZ.
MOGE_FOCAL_FILES = {
    'coco': join(DATASET_NPZ_PATH, 'coco_moge_focal.json'),
    'mpii': join(DATASET_NPZ_PATH, 'mpii_moge_focal.json'),
}

# Lazy-loaded cache
_H36M_REFIT_REMAP_CACHE = None

# Output folder to store the openpose detections
# This is required only in case you want to regenerate
# the .npz files with the annotations.
OPENPOSE_PATH = 'datasets/openpose'

#  occluders for synthetic occlusion augmentation
PASCAL_OCCLUDERS = join(DATASET_NPZ_PATH, 'pascal_occluders.pkl')

# Optional SMPLify prior folder (for GMM prior)
SMPLIFY_PRIOR_DIR = 'data'

# Optional pretrain dir for future backbone weights
PRETRAIN_DIR = 'data/pretrain'

# Fallback-only defaults for backward compatibility.
# Primary user-editable source should be:
#   - configs/paths/local.yaml
#   - configs/train/default.yaml
#   - configs/demo/default.yaml
MODEL_BACKBONE = 'resnet50'
HRNET_PRETRAINED = '/home/q36131134/HMR_models/SPIN_smplify/data/pose_hrnet_w48_256x192.pth'

# Fallback-only dataset mix (used only when train yaml is missing)
DATASET_DEFAULT_LIST = ['coco', 'mpii', 'mpi-inf-3dhp', 'h36m', '3dpw']
DATASET_DEFAULT_PARTITION = [0.25, 0.05, 0.1, 0.4, 0.2]

# Path to test/train npz files
def _first_existing_npz(*candidates):
    for name in candidates:
        p = join(DATASET_NPZ_PATH, name)
        if exists(p):
            return p
    return join(DATASET_NPZ_PATH, candidates[0])


def _warn_if_empty_root(name, path):
    if path == '':
        print(f'Warning: dataset root for {name} is empty in config.py')


def validate_dataset_roots(dataset_names):
    missing = []
    empty = []
    not_found = []
    for name in dataset_names:
        if name not in DATASET_FOLDERS:
            missing.append(name)
            continue
        root = DATASET_FOLDERS[name]
        if root == '':
            empty.append(name)
            continue
        if not os.path.isdir(root):
            not_found.append((name, root))
    if missing:
        raise KeyError(f"Missing dataset roots in config.DATASET_FOLDERS for: {missing}")
    if empty:
        raise ValueError(f"Dataset roots are empty in config.DATASET_FOLDERS for: {empty}")
    if not_found:
        raise FileNotFoundError(f"Dataset root paths do not exist: {not_found}")
    return True


def validate_dataset_npzs(dataset_names, is_train=True):
    split = 1 if is_train else 0
    for name in dataset_names:
        if name not in DATASET_FILES[split]:
            raise KeyError(f"Dataset '{name}' is missing in config.DATASET_FILES[{split}]")
        npz_path = DATASET_FILES[split][name]
        if not exists(npz_path):
            raise FileNotFoundError(
                f"Dataset npz not found for '{name}' in split {split}: {npz_path}"
            )
    return True


def validate_dataset_npz_contents(dataset_names, is_train=True):
    split = 1 if is_train else 0
    required_keys = ['imgname', 'center', 'scale']
    optional_keys = [
        'pose', 'shape', 'has_smpl', 'part', 'openpose', 'S',
        'gender', 'focal_length', 'img_focal', 'img_center'
    ]
    for name in dataset_names:
        if name not in DATASET_FILES[split]:
            raise KeyError(f"Dataset '{name}' is missing in config.DATASET_FILES[{split}]")
        npz_path = DATASET_FILES[split][name]
        if not exists(npz_path):
            raise FileNotFoundError(f"Dataset npz not found for '{name}': {npz_path}")

        data = np.load(npz_path, allow_pickle=True)
        for key in required_keys:
            if key not in data:
                raise KeyError(f"Dataset '{name}' npz missing required key '{key}': {npz_path}")

        n_img = len(data['imgname'])
        if n_img <= 0:
            raise ValueError(f"Dataset '{name}' has zero samples in npz: {npz_path}")
        n_center = len(data['center'])
        n_scale = len(data['scale'])
        if not (n_img == n_center == n_scale):
            raise ValueError(
                f"Dataset '{name}' npz has inconsistent lengths: "
                f"imgname={n_img}, center={n_center}, scale={n_scale}, path={npz_path}"
            )
        for key in optional_keys:
            if key in data:
                try:
                    k_len = len(data[key])
                except TypeError:
                    raise ValueError(
                        f"Dataset '{name}' npz key '{key}' must be per-sample array with length={n_img}, "
                        f"but got non-array value. path={npz_path}"
                    )
                if k_len != n_img:
                    raise ValueError(
                        f"Dataset '{name}' npz has inconsistent length for key '{key}': "
                        f"{k_len} vs imgname={n_img}, path={npz_path}"
                    )

        center = np.asarray(data['center'])
        if center.ndim != 2 or center.shape[1] != 2:
            raise ValueError(
                f"Dataset '{name}' npz invalid key 'center' shape: actual={center.shape}, "
                f"expected=(N,2), path={npz_path}"
            )

        scale = np.asarray(data['scale'])
        if not ((scale.ndim == 1 and scale.shape[0] == n_img) or (scale.ndim == 2 and scale.shape == (n_img, 1))):
            raise ValueError(
                f"Dataset '{name}' npz invalid key 'scale' shape: actual={scale.shape}, "
                f"expected=(N,) or (N,1), path={npz_path}"
            )

        if 'pose' in data:
            pose = np.asarray(data['pose'])
            if pose.ndim != 2 or pose.shape != (n_img, 72):
                raise ValueError(
                    f"Dataset '{name}' npz invalid key 'pose' shape: actual={pose.shape}, "
                    f"expected=({n_img},72), path={npz_path}"
                )
        if 'shape' in data:
            betas = np.asarray(data['shape'])
            if betas.ndim != 2 or betas.shape != (n_img, 10):
                raise ValueError(
                    f"Dataset '{name}' npz invalid key 'shape' shape: actual={betas.shape}, "
                    f"expected=({n_img},10), path={npz_path}"
                )
        if 'part' in data:
            part = np.asarray(data['part'])
            if part.ndim != 3 or part.shape != (n_img, 24, 3):
                raise ValueError(
                    f"Dataset '{name}' npz invalid key 'part' shape: actual={part.shape}, "
                    f"expected=({n_img},24,3), path={npz_path}"
                )
        if 'openpose' in data:
            openpose = np.asarray(data['openpose'])
            if openpose.ndim != 3 or openpose.shape != (n_img, 25, 3):
                raise ValueError(
                    f"Dataset '{name}' npz invalid key 'openpose' shape: actual={openpose.shape}, "
                    f"expected=({n_img},25,3), path={npz_path}"
                )
        if 'S' in data:
            S = np.asarray(data['S'])
            if S.ndim != 3 or S.shape != (n_img, 24, 4):
                raise ValueError(
                    f"Dataset '{name}' npz invalid key 'S' shape: actual={S.shape}, "
                    f"expected=({n_img},24,4), path={npz_path}"
                )
        if 'has_smpl' in data:
            has_smpl = np.asarray(data['has_smpl'])
            if has_smpl.ndim == 0 or has_smpl.shape[0] != n_img:
                raise ValueError(
                    f"Dataset '{name}' npz invalid key 'has_smpl' shape: actual={has_smpl.shape}, "
                    f"expected first dim={n_img}, path={npz_path}"
                )
        if 'focal_length' in data:
            focal_length = np.asarray(data['focal_length'])
            if not (
                (focal_length.ndim == 1 and focal_length.shape[0] == n_img) or
                (focal_length.ndim == 2 and focal_length.shape == (n_img, 1))
            ):
                raise ValueError(
                    f"Dataset '{name}' npz invalid key 'focal_length' shape: actual={focal_length.shape}, "
                    f"expected=({n_img},) or ({n_img},1), path={npz_path}"
                )
        if 'img_focal' in data:
            img_focal = np.asarray(data['img_focal'])
            if not (
                (img_focal.ndim == 1 and img_focal.shape[0] == n_img) or
                (img_focal.ndim == 2 and img_focal.shape == (n_img, 1))
            ):
                raise ValueError(
                    f"Dataset '{name}' npz invalid key 'img_focal' shape: actual={img_focal.shape}, "
                    f"expected=({n_img},) or ({n_img},1), path={npz_path}"
                )
        if 'img_center' in data:
            img_center = np.asarray(data['img_center'])
            if img_center.ndim != 2 or img_center.shape != (n_img, 2):
                raise ValueError(
                    f"Dataset '{name}' npz invalid key 'img_center' shape: actual={img_center.shape}, "
                    f"expected=({n_img},2), path={npz_path}"
                )
    return True


def _imgname_to_str(x):
    if isinstance(x, bytes):
        return x.decode('utf-8')
    if isinstance(x, np.ndarray) and x.shape == ():
        x = x.item()
        if isinstance(x, bytes):
            return x.decode('utf-8')
    return str(x)

def _norm_path_str(x):
    x = _imgname_to_str(x).replace('\\', '/')
    while x.startswith('./'):
        x = x[2:]
    return x


def _load_h36m_refit_remap():
    """
    Load a JSON mapping:
        ReFit imgname  -> local relative path under H36M_ROOT
    Example:
        "S1_Directions.54138969_000001.jpg":
            "images/s_01_act_02_subact_01_ca_01/s_01_act_02_subact_01_ca_01_000001.jpg"
    """
    global _H36M_REFIT_REMAP_CACHE
    if _H36M_REFIT_REMAP_CACHE is not None:
        return _H36M_REFIT_REMAP_CACHE

    remap_exact = {}
    remap_basename_unique = {}
    remap_basename_collisions = {}
    remap_exists = exists(H36M_REFIT_REMAP_FILE)
    if remap_exists:
        with open(H36M_REFIT_REMAP_FILE, 'r', encoding='utf-8') as f:
            obj = json.load(f)

        if not isinstance(obj, dict):
            raise ValueError(
                f'H36M_REFIT_REMAP_FILE must be a JSON dict, got: {type(obj)} '
                f'path={H36M_REFIT_REMAP_FILE}'
            )

        basename_to_values = {}
        for k, v in obj.items():
            k_norm = _norm_path_str(k)
            if os.path.isabs(str(v)):
                v_norm = str(v)
            else:
                v_norm = _norm_path_str(v)

            # exact mapping is the only formal mapping table.
            remap_exact[k_norm] = v_norm

            # basename fallback is optional/debug-only and tracked separately.
            base = os.path.basename(k_norm)
            basename_to_values.setdefault(base, set()).add(v_norm)

        remap_basename_unique = {
            b: sorted(values)[0]
            for b, values in basename_to_values.items()
            if len(values) == 1
        }
        remap_basename_collisions = {
            b: sorted(values)
            for b, values in basename_to_values.items()
            if len(values) > 1
        }

    _H36M_REFIT_REMAP_CACHE = {
        'remap_exact': remap_exact,
        'remap_basename_unique': remap_basename_unique,
        'remap_basename_collisions': remap_basename_collisions,
        'remap_exists': remap_exists,
    }
    return _H36M_REFIT_REMAP_CACHE


def _get_h36m_refit_mapping_lookup(imgname):
    remap = _load_h36m_refit_remap()
    imgname_norm = _norm_path_str(imgname)
    basename = os.path.basename(imgname_norm)
    exact_mapped = remap['remap_exact'].get(imgname_norm, None)
    basename_mapped = remap['remap_basename_unique'].get(basename, None)
    basename_collision_values = remap['remap_basename_collisions'].get(basename, None)
    return {
        'imgname_norm': imgname_norm,
        'basename': basename,
        'exact_mapped': exact_mapped,
        'basename_mapped': basename_mapped,
        'basename_collision_values': basename_collision_values,
    }


def get_h36m_refit_mapped_path(imgname, allow_basename_fallback=False):
    lookup = _get_h36m_refit_mapping_lookup(imgname)
    if lookup['exact_mapped'] is not None:
        return lookup['exact_mapped']
    if (
        allow_basename_fallback and
        lookup['basename_mapped'] is not None and
        lookup['basename_collision_values'] is None
    ):
        return lookup['basename_mapped']
    return None


def summarize_h36m_refit_remap():
    remap = _load_h36m_refit_remap()
    remap_exact = remap['remap_exact']
    remap_basename_unique = remap['remap_basename_unique']
    remap_basename_collisions = remap['remap_basename_collisions']
    sample_collision_examples = [
        {'basename': k, 'values': v}
        for k, v in list(remap_basename_collisions.items())[:5]
    ]
    sample_exact_items = [
        {'imgname': k, 'mapped': v}
        for k, v in list(remap_exact.items())[:5]
    ]
    return {
        'remap_file': H36M_REFIT_REMAP_FILE,
        'remap_exists': remap['remap_exists'],
        'num_exact_keys': len(remap_exact),
        'num_basename_keys': len(remap_basename_unique),
        'num_basename_collisions': len(remap_basename_collisions),
        'sample_collision_examples': sample_collision_examples,
        'sample_exact_items': sample_exact_items,
    }


def get_dataset_image_path_candidates(dataset_name, imgname, allow_h36m_basename_fallback=False):
    if dataset_name not in DATASET_FOLDERS:
        raise KeyError(f"Dataset '{dataset_name}' is not defined in config.DATASET_FOLDERS")
    root = DATASET_FOLDERS[dataset_name]
    if root == '':
        raise ValueError(f"Dataset root for '{dataset_name}' is empty in config.DATASET_FOLDERS")

    raw_imgname = _imgname_to_str(imgname).replace('\\', '/')
    imgname = _norm_path_str(raw_imgname)
    basename = os.path.basename(imgname)

    candidates = []

    # absolute path from npz itself
    if os.path.isabs(imgname):
        candidates.append(imgname)

    # dataset-specific resolution
    # common prefix-strip candidates
    if imgname.startswith('images/'):
        candidates.append(join(root, imgname[len('images/'):]))
    if raw_imgname.startswith('./'):
        candidates.append(join(root, raw_imgname[2:]))

    if dataset_name == 'h36m':
        # 1) highest priority: exact mapping from h36m_refit_imgname_to_local.json
        lookup = _get_h36m_refit_mapping_lookup(imgname)
        mapped_exact = lookup['exact_mapped']
        if mapped_exact is not None:
            if os.path.isabs(mapped_exact):
                candidates.append(mapped_exact)
            else:
                # Mapping value is defined as relative under H36M_ROOT.
                candidates.append(join(root, mapped_exact))

        # 2) optional/debug-only basename fallback (never mixed into exact mapping)
        if (
            allow_h36m_basename_fallback and
            lookup['basename_mapped'] is not None and
            lookup['basename_collision_values'] is None
        ):
            mapped_base = lookup['basename_mapped']
            if os.path.isabs(mapped_base):
                candidates.append(mapped_base)
            else:
                candidates.append(join(root, mapped_base))

        # 3) generic fallback candidates
        candidates.extend([
            join(root, imgname),
            join(root, basename),
            join(root, 'images', imgname),
            join(root, 'images', basename),
        ])
        if imgname.startswith('images/'):
            candidates.extend([
                join(root, imgname),
                join(root, imgname[len('images/'):]),
            ])

    elif dataset_name in ('h36m-p1', 'h36m-p2'):
        # Eval splits keep generic h36m path rules.
        candidates.extend([
            join(root, imgname),
            join(root, basename),
            join(root, 'images', imgname),
            join(root, 'images', basename),
        ])
        if imgname.startswith('images/'):
            candidates.extend([
                join(root, imgname),
                join(root, imgname[len('images/'):]),
            ])

    else:
        # generic fallback for non-h36m datasets
        candidates.extend([
            join(root, imgname),
            join(root, basename),
            join(root, 'images', imgname),
            join(root, 'images', basename),
        ])

        if dataset_name == 'coco':
            if imgname.startswith('train2014/'):
                candidates.extend([join(root, imgname), join(root, 'images', imgname)])
            if imgname.startswith('val2014/'):
                candidates.extend([join(root, imgname), join(root, 'images', imgname)])
            candidates.extend([
                join(root, 'train2014', basename),
                join(root, 'val2014', basename),
                join(root, 'images', 'train2014', basename),
                join(root, 'images', 'val2014', basename),
            ])
        elif dataset_name == 'mpii':
            candidates.extend([
                join(root, 'images', basename),
                join(root, basename),
            ])
        elif dataset_name == '3dpw':
            if imgname.startswith('imageFiles/'):
                candidates.extend([
                    join(root, imgname[len('imageFiles/'):]),
                    join(root, imgname),
                ])
            candidates.extend([
                join(root, 'imageFiles', imgname),
                join(root, 'imageFiles', basename),
                join(root, 'images', imgname),
                join(root, 'images', basename),
            ])
        elif dataset_name == 'mpi-inf-3dhp':
            imgname_seq = imgname.replace('/imageFrames/', '/imageSequence/')
            if imgname.startswith('imageSequence/'):
                candidates.append(join(root, imgname))
            if imgname.startswith('imageFrames/'):
                candidates.append(join(root, imgname.replace('imageFrames/', 'imageSequence/')))
            candidates.extend([
                join(root, 'images', imgname),
                join(root, 'images', basename),
                join(root, 'imageSequence', basename),
                join(root, imgname_seq),
            ])

    deduped = []
    seen = set()
    for c in candidates:
        if c not in seen:
            deduped.append(c)
            seen.add(c)
    return deduped


def resolve_dataset_image_path(dataset_name, imgname, must_exist=False, allow_h36m_basename_fallback=False):
    candidates = get_dataset_image_path_candidates(
        dataset_name,
        imgname,
        allow_h36m_basename_fallback=allow_h36m_basename_fallback
    )
    for c in candidates:
        if exists(c):
            return c
    if must_exist:
        raise FileNotFoundError(
            f"Could not resolve dataset image path. dataset={dataset_name}, "
            f"root={DATASET_FOLDERS.get(dataset_name, '<missing>')}, "
            f"imgname={_imgname_to_str(imgname)}, candidates={candidates}"
        )
    return candidates[0]


def _build_check_indices(total_n, num_check, random_check, seed):
    seq_n = min(int(num_check), int(total_n))
    seq_indices = list(range(seq_n))
    rand_n = min(int(random_check), int(total_n))
    if rand_n <= 0:
        return seq_indices
    rng = np.random.RandomState(int(seed))
    rand_indices = rng.choice(total_n, size=rand_n, replace=False).tolist()
    seq_set = set(seq_indices)
    return seq_indices + [i for i in rand_indices if i not in seq_set]


def validate_dataset_image_paths(dataset_names, is_train=True, num_check=20, random_check=100, seed=123):
    split = 1 if is_train else 0
    validate_dataset_roots(dataset_names)
    validate_dataset_npzs(dataset_names, is_train=is_train)

    for name in dataset_names:
        npz_path = DATASET_FILES[split][name]
        data = np.load(npz_path, allow_pickle=True)
        if 'imgname' not in data:
            raise KeyError(f"Dataset '{name}' npz has no 'imgname' field: {npz_path}")
        imgnames = data['imgname']
        check_indices = _build_check_indices(len(imgnames), num_check=num_check, random_check=random_check, seed=seed)
        for i in check_indices:
            imgname = _imgname_to_str(imgnames[i])
            _ = resolve_dataset_image_path(name, imgname, must_exist=True)
    return True

def summarize_dataset_path_resolution(
    dataset_name,
    is_train=True,
    num_check=20,
    random_check=50,
    seed=123,
    allow_h36m_basename_fallback=False
):
    split = 1 if is_train else 0
    if dataset_name not in DATASET_FOLDERS:
        raise KeyError(f"Dataset '{dataset_name}' missing in DATASET_FOLDERS")
    if dataset_name not in DATASET_FILES[split]:
        raise KeyError(f"Dataset '{dataset_name}' missing in DATASET_FILES[{split}]")

    root = DATASET_FOLDERS[dataset_name]
    npz_path = DATASET_FILES[split][dataset_name]
    if not exists(npz_path):
        raise FileNotFoundError(f"Dataset npz not found: {npz_path}")

    data = np.load(npz_path, allow_pickle=True)
    if 'imgname' not in data:
        raise KeyError(f"Dataset '{dataset_name}' npz missing 'imgname': {npz_path}")

    imgnames = data['imgname']
    check_indices = _build_check_indices(len(imgnames), num_check=num_check, random_check=random_check, seed=seed)
    num_resolved = 0
    failed = []
    for i in check_indices:
        raw_imgname = _imgname_to_str(imgnames[i])
        try:
            resolved = resolve_dataset_image_path(
                dataset_name,
                raw_imgname,
                must_exist=True,
                allow_h36m_basename_fallback=allow_h36m_basename_fallback
            )
            num_resolved += 1
        except Exception:
            resolved = None
            candidates = get_dataset_image_path_candidates(
                dataset_name,
                raw_imgname,
                allow_h36m_basename_fallback=allow_h36m_basename_fallback
            )
            failed.append({
                'index': int(i),
                'raw_imgname': raw_imgname,
                'resolved': resolved,
                'candidates': candidates,
            })

    return {
        'dataset_name': dataset_name,
        'dataset_root': root,
        'npz_path': npz_path,
        'total_samples': int(len(imgnames)),
        'checked_indices': [int(i) for i in check_indices],
        'num_resolved': int(num_resolved),
        'num_failed': int(len(failed)),
        'failed_examples': failed[:10],
    }


def validate_h36m_refit_mapping(num_check=20, random_check=100, seed=123, allow_basename_fallback=False):
    """
    Validate that h36m train npz imgname entries can be resolved via mapping.
    """
    remap_summary = summarize_h36m_refit_remap()
    if not remap_summary['remap_exists']:
        raise FileNotFoundError(
            f"H36M remap file does not exist: {remap_summary['remap_file']}"
        )

    if 'h36m' not in DATASET_FILES[1]:
        raise KeyError("config.DATASET_FILES[1] has no 'h36m' entry")

    npz_path = DATASET_FILES[1]['h36m']
    if not exists(npz_path):
        raise FileNotFoundError(f"h36m train npz not found: {npz_path}")

    data = np.load(npz_path, allow_pickle=True)
    if 'imgname' not in data:
        raise KeyError(f"'imgname' missing in npz: {npz_path}")

    imgnames = data['imgname']
    total_n = len(imgnames)
    if total_n <= 0:
        raise ValueError(f"h36m train npz has zero samples: {npz_path}")

    # Exact-key coverage check: h36m train should be fully covered by remap_exact.
    remap_exact = _load_h36m_refit_remap()['remap_exact']
    npz_unique_norm = {_norm_path_str(x) for x in imgnames}
    missing_exact = sorted(list(npz_unique_norm - set(remap_exact.keys())))
    if missing_exact:
        raise ValueError(
            f"h36m remap exact coverage incomplete. remap_file={remap_summary['remap_file']}, "
            f"num_npz_unique={len(npz_unique_norm)}, num_exact_keys={len(remap_exact)}, "
            f"num_missing={len(missing_exact)}, missing_examples={missing_exact[:10]}"
        )

    indices = _build_check_indices(total_n, num_check=num_check, random_check=random_check, seed=seed)

    for i in indices:
        raw_imgname = _imgname_to_str(imgnames[i])
        lookup = _get_h36m_refit_mapping_lookup(raw_imgname)
        candidates = get_dataset_image_path_candidates(
            'h36m',
            raw_imgname,
            allow_h36m_basename_fallback=allow_basename_fallback
        )
        try:
            _ = resolve_dataset_image_path(
                'h36m',
                raw_imgname,
                must_exist=True,
                allow_h36m_basename_fallback=allow_basename_fallback
            )
        except Exception as e:
            raise FileNotFoundError(
                f"h36m remap validation failed. "
                f"raw_imgname={raw_imgname}, remap_file={remap_summary['remap_file']}, "
                f"exact_mapped={lookup['exact_mapped']}, "
                f"basename_mapped={lookup['basename_mapped']}, "
                f"basename_collision_values={lookup['basename_collision_values']}, "
                f"candidates={candidates}, error={e}"
            ) from e
    return True
    
def validate_smpl_files(require_gender=False):
    required = ['SMPL_NEUTRAL.pkl']
    if require_gender:
        required += ['SMPL_MALE.pkl', 'SMPL_FEMALE.pkl']

    missing = []
    for name in required:
        p = join(SMPL_MODEL_DIR, name)
        if not exists(p):
            missing.append(p)

    if missing:
        raise FileNotFoundError(f"Missing SMPL model files: {missing}")
    return True


def validate_core_data_files():
    required_files = [
        SMPL_MEAN_PARAMS,
        JOINT_REGRESSOR_TRAIN_EXTRA,
        JOINT_REGRESSOR_H36M,
    ]
    missing = [p for p in required_files if not exists(p)]
    if missing:
        raise FileNotFoundError(f"Missing core data files: {missing}")
    return True


def validate_renderer_data_files():
    required_files = [
        VERTEX_TEXTURE_FILE,
        CUBE_PARTS_FILE,
    ]
    missing = [p for p in required_files if not exists(p)]
    if missing:
        raise FileNotFoundError(f"Missing renderer data files: {missing}")
    return True


def validate_hrnet_pretrained():
    if HRNET_PRETRAINED == '':
        raise ValueError('HRNET_PRETRAINED is empty in config.py')
    if not exists(HRNET_PRETRAINED):
        raise FileNotFoundError(f"HRNet pretrained checkpoint not found: {HRNET_PRETRAINED}")
    return True


def summarize_dataset_npz(dataset_name, is_train=True):
    split = 1 if is_train else 0
    if dataset_name not in DATASET_FILES[split]:
        raise KeyError(f"Dataset '{dataset_name}' is missing in config.DATASET_FILES[{split}]")
    npz_path = DATASET_FILES[split][dataset_name]
    if not exists(npz_path):
        raise FileNotFoundError(f"Dataset npz not found for '{dataset_name}': {npz_path}")

    data = np.load(npz_path, allow_pickle=True)
    keys = list(data.keys())
    imgnames = data['imgname'] if 'imgname' in data else []
    first_examples = [_imgname_to_str(x) for x in imgnames[:3]]

    prefixes = {}
    for x in imgnames[:50]:
        s = _imgname_to_str(x).replace('\\', '/')
        p = s.split('/')[0] if '/' in s else '.'
        prefixes[p] = prefixes.get(p, 0) + 1

    return {
        'npz_path': npz_path,
        'num_samples': int(len(imgnames)) if 'imgname' in data else 0,
        'available_keys': keys,
        'center_shape': tuple(data['center'].shape) if 'center' in data else None,
        'scale_shape': tuple(data['scale'].shape) if 'scale' in data else None,
        'pose_shape': tuple(data['pose'].shape) if 'pose' in data else None,
        'shape_shape': tuple(data['shape'].shape) if 'shape' in data else None,
        'part_shape': tuple(data['part'].shape) if 'part' in data else None,
        'openpose_shape': tuple(data['openpose'].shape) if 'openpose' in data else None,
        'S_shape': tuple(data['S'].shape) if 'S' in data else None,
        'has_smpl_shape': tuple(data['has_smpl'].shape) if 'has_smpl' in data else None,
        'focal_length_shape': tuple(data['focal_length'].shape) if 'focal_length' in data else None,
        'img_focal_shape': tuple(data['img_focal'].shape) if 'img_focal' in data else None,
        'img_center_shape': tuple(data['img_center'].shape) if 'img_center' in data else None,
        'has_pose': 'pose' in data,
        'has_shape': 'shape' in data,
        'has_smpl_flag': 'has_smpl' in data,
        'has_part': 'part' in data,
        'has_openpose': 'openpose' in data,
        'has_S': 'S' in data,
        'has_gender': 'gender' in data,
        'has_focal_length': 'focal_length' in data,
        'has_img_focal': 'img_focal' in data,
        'has_img_center': 'img_center' in data,
        'first_img_examples': first_examples,
        'first_img_prefixes': prefixes,
    }

_THERMAL_NPZ_DIR = join(DATASET_NPZ_PATH, 'thermal_pgt_camsmplify_v5')

DATASET_FILES = [
    {
        'h36m-p1': join(DATASET_NPZ_PATH, 'h36m_valid_protocol1_local_aligned.npz'),
        'h36m-p2': join(DATASET_NPZ_PATH, 'h36m_valid_protocol2_local_aligned.npz'),
        'lsp': join(DATASET_NPZ_PATH, 'lsp_dataset_test.npz'),
        'mpi-inf-3dhp': join(DATASET_NPZ_PATH, 'mpi_inf_3dhp_valid.npz'),
        '3dpw': join(DATASET_NPZ_PATH, '3dpw_test.npz'),
        # Thermal test splits (for eval and npz_path_override-based validation)
        'llvip': join(_THERMAL_NPZ_DIR, 'llvip_test_smplify_pgt.npz'),
        'llvip-pgt': join(_THERMAL_NPZ_DIR, 'llvip_test_smplify_pgt.npz'),
        'otp2': join(_THERMAL_NPZ_DIR, 'otp2_test_smplify_pgt.npz'),
        'otp2-pgt': join(_THERMAL_NPZ_DIR, 'otp2_test_smplify_pgt.npz'),
    },

    {
        # ReFit-style train datasets
        # h36m: prefer local-aligned NPZ (direct disk paths, no refit remap needed).
        'h36m': _first_existing_npz('h36m_train_local_aligned.npz', 'h36m_train.npz', 'h36m_train_eft.npz'),
        # mpii: prefer ReFit EFT npz, then SPIN-style fallback.
        'mpii': _first_existing_npz('mpii_train_eft.npz', 'mpii_train.npz'),
        # coco: prefer ReFit EFT npz, then SPIN-style fallback.
        'coco': _first_existing_npz('coco_2014_eft.npz', 'coco_2014_train.npz'),
        # mpi-inf-3dhp: prefer ReFit-style mpii3d name, then SPIN-style fallback.
        'mpi-inf-3dhp': _first_existing_npz('mpii3d_train.npz', 'mpi_inf_3dhp_train.npz'),
        # 3dpw: prefer ReFit train npz, then SPIN-style fallback.
        '3dpw': _first_existing_npz('3dpw_train.npz', '3dpw_train_w2d_smpl3d_gender.npz'),
        # agora: preprocessed npz with pose/shape/S/img_focal/img_center
        'agora': join(DATASET_NPZ_PATH, 'agora_train.npz'),

        # Keep SPIN-compatible datasets for backward compatibility
        'lsp-orig': join(DATASET_NPZ_PATH, 'lsp_dataset_original_train.npz'),
        'lspet': join(DATASET_NPZ_PATH, 'hr-lspet_train.npz'),

        # Thermal datasets (pGT from CAM-SMPLify v3 relaxed)
        'llvip': join(_THERMAL_NPZ_DIR, 'llvip_train_smplify_pgt.npz'),
        'llvip-pgt': join(_THERMAL_NPZ_DIR, 'llvip_train_smplify_pgt.npz'),
        # 09.13 valsplit: 90% image-level subset of llvip-pgt's train images,
        # held out 10% -> llvip_val10_smplify_pgt.npz (VALIDATION.LLVIP_NPZ).
        # Additive only -- 'llvip-pgt' above is untouched so every existing
        # config keeps training on the full 18,633-instance set.
        'llvip-pgt-train90': join(_THERMAL_NPZ_DIR, 'llvip_train90_smplify_pgt.npz'),
        'otp2': join(_THERMAL_NPZ_DIR, 'otp2_train_smplify_pgt.npz'),
        'otp2-pgt': join(_THERMAL_NPZ_DIR, 'otp2_train_smplify_pgt.npz'),
    }
]

DATASET_FOLDERS = {
    'h36m': H36M_TRAIN_ROOT,
    'h36m-p1': H36M_VAL_ROOT,
    'h36m-p2': H36M_VAL_ROOT,
    'lsp-orig': LSP_ORIGINAL_ROOT,
    'lsp': LSP_ROOT,
    'lspet': LSPET_ROOT,
    'mpi-inf-3dhp': MPI_INF_3DHP_ROOT,
    'mpii': MPII_ROOT,
    'coco': COCO_ROOT,
    '3dpw': PW3D_ROOT,
    'agora': AGORA_ROOT,
    'upi-s1h': UPI_S1H_ROOT,
    # Thermal datasets
    'llvip': LLVIP_ROOT,
    'llvip-pgt': LLVIP_ROOT,
    'llvip-pgt-train90': LLVIP_ROOT,
    'otp2': OTP2_ROOT,
    'otp2-pgt': OTP2_DATASET_ROOT,
}

CUBE_PARTS_FILE = 'data/cube_parts.npy'
JOINT_REGRESSOR_TRAIN_EXTRA = 'data/J_regressor_extra.npy'
JOINT_REGRESSOR_H36M = 'data/J_regressor_h36m.npy'
VERTEX_TEXTURE_FILE = 'data/vertex_texture.npy'
STATIC_FITS_DIR = 'data/static_fits'
SMPL_MEAN_PARAMS = 'data/smpl_mean_params.npz'
SMPL_MODEL_DIR = 'data/smpl'


def _project_config_yaml(*parts):
    p = os.path.abspath(os.path.join(os.path.dirname(__file__), *parts))
    return p if os.path.isfile(p) else None


def build_train_config_defaults():
    """
    Fallback-only helper for backward compatibility.
    Primary train config should come from configs/train/default.yaml.
    """
    default_yaml = _project_config_yaml('configs', 'train', 'default.yaml')
    if default_yaml is not None:
        cfg = load_yaml_config(default_yaml)
        if isinstance(cfg, dict):
            return cfg
    return {
        "EXPERIMENT": {"NAME": "spin_hrnetw48_spin_smplify", "TIME_TO_RUN": 30000000, "RESUME": False},
        "IO": {"LOG_DIR": "logs", "CHECKPOINT": None, "PRETRAINED_CHECKPOINT": None},
        "MODEL": {"BACKBONE": MODEL_BACKBONE, "IMG_RES": 256, "STRICT_HRNET": True, "PRETRAINED_INIT": True},
        "DATASET": {"LIST": list(DATASET_DEFAULT_LIST), "PARTITION": list(DATASET_DEFAULT_PARTITION)},
        "TRAIN": {
            "NUM_EPOCHS": 50,
            "LR": 5e-5,
            "BATCH_SIZE": 64,
            "WEIGHT_DECAY": 0.0,
            "SAVE_STEP": 50,
            "MAX_STEP": 0,
        },
        "DATALOADER": {"NUM_WORKERS": 8, "PIN_MEMORY": True, "SHUFFLE_TRAIN": True},
        # Validation/checkpoint cadence consolidated to the modern keys
        # (VALIDATION.INTERVAL / TRAIN.SAVE_STEP). Legacy aliases TRAIN.VALID_STEP,
        # LOGGING.TEST_STEPS, LOGGING.CHECKPOINT_STEPS removed so per-concept keys
        # are unique and TrainOptions no longer warns about duplicate legacy keys.
        "LOGGING": {"SUMMARY_STEPS": 200, "RENDER_SUMMARIES": False},
        "AUG": {"USE_AUGMENTATION": True, "ROT_FACTOR": 30.0, "NOISE_FACTOR": 0.2, "SCALE_FACTOR": 0.20, "OCC_PROB": 0.15, "FLIP_PROB": 0.5, "EXTREME_CROP_AUG_RATE": 0.0, "EXTREME_CROP_AUG_LEVEL": 0},
        "LOSS": {
            "SUPERVISION_MODE": "fit_or_gt",
            "SHAPE_LOSS_WEIGHT": 0.1,
            "KEYPOINT_2D_LOSS_WEIGHT": 3.0,
            "KEYPOINT_3D_LOSS_WEIGHT": 5.0,
            "POSE_LOSS_WEIGHT": 1.0,
            "BETA_LOSS_WEIGHT": 0.02,
            "CAM_DEPTH_LOSS_WEIGHT": 1.0,
            "CAM_LOSS_WEIGHT": 0.0,
            "OPENPOSE_TRAIN_WEIGHT": 0.0,
            "GT_TRAIN_WEIGHT": 1.0,
        },
        "SMPLIFY": {"RUN": True, "THRESHOLD": 80.0, "NUM_ITERS": 100, "NUM_CAM_ITERS": 100, "NUM_BODY_ITERS": 200, "STEP_SIZE": 5e-3, "POSE_PRIOR_WEIGHT": 4.78, "SHAPE_PRIOR_WEIGHT": 5.0, "ANGLE_PRIOR_WEIGHT": 15.2, "SIGMA": 100.0, "INIT_POSE_REG_WEIGHT": 0.0, "INIT_BETAS_REG_WEIGHT": 0.0, "INIT_GLOBAL_ORIENT_REG_WEIGHT": 0.0},
        "VALIDATION": {
            "ENABLED": True,
            "INTERVAL": 3000,
            "DATASET": "3dpw",
            "METRIC": "MPJPE",
            "SAVE_BEST": True,
            "BATCH_SIZE": 0,
            "NUM_WORKERS": 4,
            "PW3D_NPZ": "/home/q36131134/HMR_models/SPIN_smplify/data/dataset_extras/3dpw_test.npz",
            "PW3D_SEQUENCE_DIR": "/home/q36131134/HMR_models/dataset/pw3d/sequenceFiles/validation",
        },
        "DEBUG": {"IGNORE_3D": False, "SKIP_DATASET_IMAGE_CHECK": True},
    }


PATH_FIELD_KEYS = {
    "HRNET_PRETRAINED",
    "SMPL_MODEL_DIR",
    "SMPL_MEAN_PARAMS",
    "DATASET_NPZ_PATH",
    "JOINT_REGRESSOR_TRAIN_EXTRA",
    "JOINT_REGRESSOR_H36M",
    "CUBE_PARTS_FILE",
    "VERTEX_TEXTURE_FILE",
    "STATIC_FITS_DIR",
    "PASCAL_OCCLUDERS",
    "OPENPOSE_PATH",
    "SMPLIFY_PRIOR_DIR",
    "H36M_ROOT",
    "H36M_TRAIN_ROOT",
    "H36M_VAL_ROOT",
    "LSP_ROOT",
    "LSP_ORIGINAL_ROOT",
    "LSPET_ROOT",
    "MPII_ROOT",
    "COCO_ROOT",
    "MPI_INF_3DHP_ROOT",
    "PW3D_ROOT",
    "UPI_S1H_ROOT",
}


def build_paths_config_defaults():
    return {
        "DATASET_ROOTS": {
            "H36M_ROOT": H36M_ROOT,
            "H36M_TRAIN_ROOT": H36M_TRAIN_ROOT,
            "H36M_VAL_ROOT": H36M_VAL_ROOT,
            "LSP_ROOT": LSP_ROOT,
            "LSP_ORIGINAL_ROOT": LSP_ORIGINAL_ROOT,
            "LSPET_ROOT": LSPET_ROOT,
            "MPII_ROOT": MPII_ROOT,
            "COCO_ROOT": COCO_ROOT,
            "MPI_INF_3DHP_ROOT": MPI_INF_3DHP_ROOT,
            "PW3D_ROOT": PW3D_ROOT,
            "AGORA_ROOT": AGORA_ROOT,
            "UPI_S1H_ROOT": UPI_S1H_ROOT,
            "LLVIP_ROOT": LLVIP_ROOT,
            "OTP2_DATASET_ROOT": OTP2_DATASET_ROOT,
            "OTP2_ROOT": OTP2_ROOT,
        },
        "PATHS": {
            "DATASET_NPZ_PATH": DATASET_NPZ_PATH,
            "H36M_REFIT_REMAP_FILE": H36M_REFIT_REMAP_FILE,
            "OPENPOSE_PATH": OPENPOSE_PATH,
            "PASCAL_OCCLUDERS": PASCAL_OCCLUDERS,
            "SMPLIFY_PRIOR_DIR": SMPLIFY_PRIOR_DIR,
            "PRETRAIN_DIR": PRETRAIN_DIR,
            "HRNET_PRETRAINED": HRNET_PRETRAINED,
            "CUBE_PARTS_FILE": CUBE_PARTS_FILE,
            "JOINT_REGRESSOR_TRAIN_EXTRA": JOINT_REGRESSOR_TRAIN_EXTRA,
            "JOINT_REGRESSOR_H36M": JOINT_REGRESSOR_H36M,
            "VERTEX_TEXTURE_FILE": VERTEX_TEXTURE_FILE,
            "STATIC_FITS_DIR": STATIC_FITS_DIR,
            "SMPL_MEAN_PARAMS": SMPL_MEAN_PARAMS,
            "SMPL_MODEL_DIR": SMPL_MODEL_DIR,
        },
    }


def load_paths_config(path):
    cfg = load_any_config(path)
    if not isinstance(cfg, dict):
        raise ValueError(f"paths config must be a dict: {path}")
    return cfg


def load_train_params_config(path):
    return load_any_config(path)


def load_demo_params_config(path):
    return load_any_config(path)


def _rebuild_runtime_tables():
    global H36M_REFIT_REMAP_FILE
    global DATASET_FILES, DATASET_FOLDERS
    global _H36M_REFIT_REMAP_CACHE

    H36M_REFIT_REMAP_FILE = _first_existing_path(
        join(DATASET_NPZ_PATH, 'h36m_refit_imgname_to_local.json'),
        os.path.normpath(join(H36M_ROOT, '..', 'h36m_flat', 'h36m_refit_imgname_to_local.json')),
    )
    _H36M_REFIT_REMAP_CACHE = None

    _thermal_npz_dir = join(DATASET_NPZ_PATH, 'thermal_pgt_camsmplify_v5')

    DATASET_FILES = [
        {
            'h36m-p1': join(DATASET_NPZ_PATH, 'h36m_valid_protocol1_local_aligned.npz'),
            'h36m-p2': join(DATASET_NPZ_PATH, 'h36m_valid_protocol2_local_aligned.npz'),
            'lsp': join(DATASET_NPZ_PATH, 'lsp_dataset_test.npz'),
            'mpi-inf-3dhp': join(DATASET_NPZ_PATH, 'mpi_inf_3dhp_valid.npz'),
            '3dpw': join(DATASET_NPZ_PATH, '3dpw_test.npz'),
            'llvip': join(_thermal_npz_dir, 'llvip_test_smplify_pgt.npz'),
            'llvip-pgt': join(_thermal_npz_dir, 'llvip_test_smplify_pgt.npz'),
            'otp2': join(_thermal_npz_dir, 'otp2_test_smplify_pgt.npz'),
            'otp2-pgt': join(_thermal_npz_dir, 'otp2_test_smplify_pgt.npz'),
        },
        {
            'h36m': _first_existing_npz('h36m_train_local_aligned.npz', 'h36m_train.npz', 'h36m_train_eft.npz'),
            'mpii': _first_existing_npz('mpii_train_eft.npz', 'mpii_train.npz'),
            'coco': _first_existing_npz('coco_2014_eft.npz', 'coco_2014_train.npz'),
            'mpi-inf-3dhp': _first_existing_npz('mpii3d_train.npz', 'mpi_inf_3dhp_train.npz'),
            '3dpw': _first_existing_npz('3dpw_train.npz', '3dpw_train_w2d_smpl3d_gender.npz'),
            'agora': join(DATASET_NPZ_PATH, 'agora_train.npz'),
            'lsp-orig': join(DATASET_NPZ_PATH, 'lsp_dataset_original_train.npz'),
            'lspet': join(DATASET_NPZ_PATH, 'hr-lspet_train.npz'),
            'llvip': join(_thermal_npz_dir, 'llvip_train_smplify_pgt.npz'),
            'llvip-pgt': join(_thermal_npz_dir, 'llvip_train_smplify_pgt.npz'),
            # 09.13 valsplit (see module-level DATASET_FILES[1] above for detail).
            'llvip-pgt-train90': join(_thermal_npz_dir, 'llvip_train90_smplify_pgt.npz'),
            'otp2': join(_thermal_npz_dir, 'otp2_train_smplify_pgt.npz'),
            'otp2-pgt': join(_thermal_npz_dir, 'otp2_train_smplify_pgt.npz'),
        }
    ]

    DATASET_FOLDERS = {
        'h36m': H36M_TRAIN_ROOT,
        'h36m-p1': H36M_VAL_ROOT,
        'h36m-p2': H36M_VAL_ROOT,
        'lsp-orig': LSP_ORIGINAL_ROOT,
        'lsp': LSP_ROOT,
        'lspet': LSPET_ROOT,
        'mpi-inf-3dhp': MPI_INF_3DHP_ROOT,
        'mpii': MPII_ROOT,
        'coco': COCO_ROOT,
        '3dpw': PW3D_ROOT,
        'agora': AGORA_ROOT,
        'upi-s1h': UPI_S1H_ROOT,
        'llvip': LLVIP_ROOT,
        'llvip-pgt': LLVIP_ROOT,
        'llvip-pgt-train90': LLVIP_ROOT,
        'otp2': OTP2_ROOT,
        'otp2-pgt': OTP2_DATASET_ROOT,
    }


def apply_paths_config(paths_cfg, verbose=True):
    """
    Apply machine-local path overrides at runtime.
    This mutates config module globals so existing training/demo code can reuse
    current path validation and dataset resolution helpers without refactoring.
    """
    if paths_cfg is None:
        return {}
    if isinstance(paths_cfg, str):
        paths_cfg = load_paths_config(paths_cfg)
    if not isinstance(paths_cfg, dict):
        raise ValueError("paths_cfg must be dict or path string")

    dataset_roots = paths_cfg.get("DATASET_ROOTS", paths_cfg.get("dataset_roots", {}))
    path_overrides = paths_cfg.get("PATHS", paths_cfg.get("paths", {}))

    root_map = {
        "H36M_ROOT": "H36M_ROOT",
        "H36M_TRAIN_ROOT": "H36M_TRAIN_ROOT",
        "H36M_VAL_ROOT": "H36M_VAL_ROOT",
        "LSP_ROOT": "LSP_ROOT",
        "LSP_ORIGINAL_ROOT": "LSP_ORIGINAL_ROOT",
        "LSPET_ROOT": "LSPET_ROOT",
        "MPII_ROOT": "MPII_ROOT",
        "COCO_ROOT": "COCO_ROOT",
        "MPI_INF_3DHP_ROOT": "MPI_INF_3DHP_ROOT",
        "PW3D_ROOT": "PW3D_ROOT",
        "AGORA_ROOT": "AGORA_ROOT",
        "UPI_S1H_ROOT": "UPI_S1H_ROOT",
        "LLVIP_ROOT": "LLVIP_ROOT",
        "OTP2_DATASET_ROOT": "OTP2_DATASET_ROOT",
        "OTP2_ROOT": "OTP2_ROOT",
        "LLVIP_VISIBLE_ROOT": "LLVIP_VISIBLE_ROOT",
    }
    for key, global_name in root_map.items():
        if key in dataset_roots and dataset_roots[key] is not None:
            globals()[global_name] = dataset_roots[key]

    override_keys = [
        "DATASET_NPZ_PATH",
        "H36M_REFIT_REMAP_FILE",
        "OPENPOSE_PATH",
        "PASCAL_OCCLUDERS",
        "SMPLIFY_PRIOR_DIR",
        "PRETRAIN_DIR",
        "HRNET_PRETRAINED",
        "CUBE_PARTS_FILE",
        "JOINT_REGRESSOR_TRAIN_EXTRA",
        "JOINT_REGRESSOR_H36M",
        "VERTEX_TEXTURE_FILE",
        "STATIC_FITS_DIR",
        "SMPL_MEAN_PARAMS",
        "SMPL_MODEL_DIR",
    ]
    for key in override_keys:
        if key in path_overrides and path_overrides[key] is not None:
            globals()[key] = path_overrides[key]

    _rebuild_runtime_tables()

    # Respect explicit remap file override after auto rebuild.
    if "H36M_REFIT_REMAP_FILE" in path_overrides and path_overrides["H36M_REFIT_REMAP_FILE"] is not None:
        globals()["H36M_REFIT_REMAP_FILE"] = path_overrides["H36M_REFIT_REMAP_FILE"]

    if verbose:
        print("[config] Applied paths config overrides.")
    return {
        "DATASET_ROOTS": dataset_roots,
        "PATHS": path_overrides,
    }


def apply_dataset_npz_overrides(overrides):
    """Override per-dataset NPZ filenames in DATASET_FILES[1].

    Call after apply_paths_config (which rebuilds DATASET_FILES via _rebuild_runtime_tables).
    overrides: dict {dataset_name: npz_filename_relative_to_DATASET_NPZ_PATH}
    e.g. {'coco': 'coco_2014_train.npz', 'mpii': 'mpii_train.npz'}
    """
    if not overrides:
        return
    for ds_name, npz_name in overrides.items():
        if npz_name:
            DATASET_FILES[1][ds_name] = os.path.join(DATASET_NPZ_PATH, str(npz_name))


def validate_runtime_train_paths(
    dataset_names,
    require_renderer_files=False,
    require_hrnet=False,
    require_smpl_gender=False,
):
    validate_dataset_roots(dataset_names)
    validate_dataset_npzs(dataset_names, is_train=True)
    validate_dataset_npz_contents(dataset_names, is_train=True)
    validate_smpl_files(require_gender=require_smpl_gender)
    validate_core_data_files()
    if require_renderer_files:
        validate_renderer_data_files()
    if require_hrnet:
        validate_hrnet_pretrained()
    if not os.path.isfile(join(SMPLIFY_PRIOR_DIR, 'gmm_08.pkl')):
        raise FileNotFoundError(
            f"SMPLify prior file not found: {join(SMPLIFY_PRIOR_DIR, 'gmm_08.pkl')}"
        )
    return True


def validate_runtime_demo_paths(require_renderer=False, require_smpl_gender=False):
    if not os.path.isdir(SMPL_MODEL_DIR):
        raise FileNotFoundError(f"SMPL model dir not found: {SMPL_MODEL_DIR}")
    validate_smpl_files(require_gender=require_smpl_gender)
    validate_core_data_files()
    if require_renderer:
        validate_renderer_data_files()
    return True

_warn_if_empty_root('h36m', H36M_ROOT)
_warn_if_empty_root('mpii', MPII_ROOT)
_warn_if_empty_root('coco', COCO_ROOT)
_warn_if_empty_root('mpi-inf-3dhp', MPI_INF_3DHP_ROOT)
_warn_if_empty_root('3dpw', PW3D_ROOT)
_warn_if_empty_root('agora', AGORA_ROOT)
