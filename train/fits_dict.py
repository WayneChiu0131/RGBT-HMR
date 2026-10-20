import os
import cv2
import torch
import numpy as np

import config
import constants
from utils.geometry import batch_rodrigues


class FitsDict:
    """Dictionary keeping track of the best fit per image in the training set.

    Init sources (in priority order): static_fits → zero_init.
    NPZ pose/shape init is disabled to avoid noisy pseudo-GT bootstrapping.
    Companion fits from a previous checkpoint are loaded by BaseTrainer after
    __init__ via load_companion(ckpt_path).
    """

    def __init__(self, options, train_dataset):
        self.options = options
        self.train_dataset = train_dataset
        self.fits_dict = {}
        self.fit_source = {}
        self.verbose = bool(getattr(options, 'fits_dict_verbose', False))

        # Array used to flip SMPL pose parameters
        self.flipped_parts = torch.tensor(
            constants.SMPL_POSE_FLIP_PERM, dtype=torch.int64
        )

        # Load dictionary state
        for ds_name, ds_idx in train_dataset.dataset_dict.items():
            ds_obj = train_dataset.datasets[ds_idx]
            ds_len = len(ds_obj)
            if ds_len <= 0:
                raise ValueError(
                    f"Dataset '{ds_name}' has zero samples; cannot initialize FitsDict"
                )

            infer = self.infer_fit_source_for_dataset(ds_name, ds_len)
            source = infer["source"]
            detail = infer.get("detail", "")
            static_file = infer.get("static_file", "")

            if source.endswith("_invalid"):
                print(
                    f"[FitsDict][WARNING] {source} for dataset '{ds_name}': {detail}. "
                    f"Falling back to zero_init."
                )
                fits = self._zero_init_fits(ds_len)
                self.fits_dict[ds_name] = fits
                self.fit_source[ds_name] = 'zero_init'
                self._log(
                    f"[FitsDict] {ds_name}: source=zero_init (size mismatch fallback), "
                    f"num_samples={ds_len}"
                )
                continue

            if source == "static_fits":
                fits = self._load_fits_file(
                    static_file, ds_name, ds_len, source_name="static_fits"
                )
            elif source == "zero_init":
                fits = self._zero_init_fits(ds_len)
                self._log(
                    f"[FitsDict][WARNING] source=zero_init dataset={ds_name} "
                    f"(no static_fits; NPZ pose/shape init disabled)"
                )
            else:
                raise RuntimeError(
                    f"Unhandled fits source '{source}' for dataset '{ds_name}'"
                )

            if fits is None:
                raise RuntimeError(
                    f"Fits tensor is None for dataset '{ds_name}' from source '{source}'"
                )

            self.fits_dict[ds_name] = fits
            self.fit_source[ds_name] = source
            self._log(
                f"[FitsDict] {ds_name}: source={source}, "
                f"num_samples={ds_len}, tensor_shape={tuple(fits.shape)}"
            )

    @staticmethod
    def _static_fits_file_path(ds_name):
        return os.path.join(config.STATIC_FITS_DIR, ds_name + "_fits.npy")

    @staticmethod
    def _validate_fits_array_shape(arr, ds_len):
        return arr.shape == (ds_len, 82)

    def _log(self, msg):
        if self.verbose:
            print(msg)

    @staticmethod
    def infer_fit_source_for_dataset(ds_name, ds_len):
        """Infer the source of fits for a dataset.

        Sources (in priority order): static_fits → zero_init.
        Checkpoint-dir _fits.npy is intentionally excluded.
        """
        static_file = FitsDict._static_fits_file_path(ds_name)
        npz_path = config.DATASET_FILES[1].get(ds_name, "")

        # 1) static_fits
        if os.path.isfile(static_file):
            arr = np.load(static_file)
            if not FitsDict._validate_fits_array_shape(arr, ds_len):
                return {
                    "source": "static_fits_invalid",
                    "detail": (
                        f"static_fits shape={arr.shape}, "
                        f"expected={(ds_len, 82)}, path={static_file}"
                    ),
                    "static_file": static_file,
                    "npz_path": npz_path,
                }
            return {
                "source": "static_fits",
                "detail": static_file,
                "static_file": static_file,
                "npz_path": npz_path,
            }

        # 2) zero init
        return {
            "source": "zero_init",
            "detail": f"no static_fits for {ds_name}; NPZ pose/shape init disabled",
            "static_file": static_file,
            "npz_path": npz_path,
        }

    def _load_fits_file(self, path, ds_name, ds_len, source_name):
        arr = np.load(path)
        expected = (ds_len, 82)
        if arr.shape != expected:
            raise ValueError(
                f"Invalid {source_name} fits shape for '{ds_name}': "
                f"got {arr.shape}, expected {expected}, path={path}"
            )
        return torch.from_numpy(arr.astype(np.float32))

    def _zero_init_fits(self, ds_len):
        return torch.zeros((ds_len, 82), dtype=torch.float32)

    def __getitem__(self, x):
        """Retrieve dictionary entries."""
        dataset_name, ind, rot, is_flipped = x

        if isinstance(dataset_name, (str, bytes)):
            dataset_name = [dataset_name]
        else:
            dataset_name = list(dataset_name)

        rot_t = torch.as_tensor(rot).float()
        is_flipped_t = torch.as_tensor(is_flipped).bool()
        target_device = rot_t.device
        ind_cpu = torch.as_tensor(ind).long().cpu()

        batch_size = len(dataset_name)
        pose = torch.zeros((batch_size, 72), dtype=torch.float32)
        betas = torch.zeros((batch_size, 10), dtype=torch.float32)

        for n in range(batch_size):
            ds = dataset_name[n]
            if isinstance(ds, bytes):
                ds = ds.decode("utf-8")
            else:
                ds = str(ds)

            i_int = int(ind_cpu[n].item())

            if ds not in self.fits_dict:
                raise KeyError(f"Unknown dataset in FitsDict lookup: {ds}")

            ds_len = self.fits_dict[ds].shape[0]
            if i_int < 0 or i_int >= ds_len:
                raise IndexError(
                    f"Index out of range for dataset '{ds}': "
                    f"index={i_int}, dataset_len={ds_len}"
                )

            params = self.fits_dict[ds][i_int]
            pose[n, :] = params[:72]
            betas[n, :] = params[72:]

        pose = pose.to(target_device).clone()
        betas = betas.to(target_device).clone()
        pose = self.flip_pose(self.rotate_pose(pose, rot_t), is_flipped_t)
        return pose, betas

    def __setitem__(self, x, val):
        """Update dictionary entries."""
        dataset_name, ind, rot, is_flipped, update = x

        if isinstance(dataset_name, (str, bytes)):
            dataset_name = [dataset_name]
        else:
            dataset_name = list(dataset_name)

        pose, betas = val
        target_device = pose.device

        rot_t = torch.as_tensor(rot).float().to(target_device)
        is_flipped_t = torch.as_tensor(is_flipped).bool().to(target_device)
        ind_cpu = torch.as_tensor(ind).long().cpu()
        update_cpu = torch.as_tensor(update).bool().cpu()

        batch_size = len(dataset_name)

        # Undo flipping and rotation before saving back
        pose = self.rotate_pose(self.flip_pose(pose, is_flipped_t), -rot_t)
        params = torch.cat((pose, betas), dim=-1).cpu()

        for n in range(batch_size):
            if not bool(update_cpu[n].item()):
                continue

            ds = dataset_name[n]
            if isinstance(ds, bytes):
                ds = ds.decode("utf-8")
            else:
                ds = str(ds)

            i_int = int(ind_cpu[n].item())

            if ds not in self.fits_dict:
                raise KeyError(f"Unknown dataset in FitsDict update: {ds}")

            ds_len = self.fits_dict[ds].shape[0]
            if i_int < 0 or i_int >= ds_len:
                raise IndexError(
                    f"Index out of range for dataset '{ds}': "
                    f"index={i_int}, dataset_len={ds_len}"
                )

            self.fits_dict[ds][i_int] = params[n]

    def flip_pose(self, pose, is_flipped):
        """Flip SMPL pose parameters."""
        is_flipped = is_flipped.bool().to(pose.device)
        flipped_parts = self.flipped_parts.to(pose.device)
        pose_f = pose.clone()
        pose_f[is_flipped, :] = pose[is_flipped][:, flipped_parts]
        # Negate the 2nd and 3rd axis-angle channels
        pose_f[is_flipped, 1::3] *= -1
        pose_f[is_flipped, 2::3] *= -1
        return pose_f

    @staticmethod
    def _companion_path(ckpt_path: str) -> str:
        stem = ckpt_path
        for ext in ('.pt', '.pth', '.ckpt'):
            if stem.endswith(ext):
                stem = stem[:-len(ext)]
                break
        return stem + '_fits.npy'

    def save_companion(self, ckpt_path: str):
        """Save all dataset fits alongside a checkpoint as <ckpt_stem>_fits.npy."""
        companion_path = self._companion_path(ckpt_path)
        save_dict = {
            ds_name: tensor.cpu().numpy()
            for ds_name, tensor in self.fits_dict.items()
        }
        np.save(companion_path, save_dict)
        self._log(f"[FitsDict] saved companion fits → {companion_path}")

    def load_companion(self, ckpt_path: str) -> bool:
        """Load fits from a companion file saved by save_companion. Returns True if loaded."""
        companion_path = self._companion_path(ckpt_path)
        if not os.path.isfile(companion_path):
            print(f"[FitsDict] no companion fits found at {companion_path}, keeping init state")
            return False
        saved = np.load(companion_path, allow_pickle=True).item()
        loaded_count = 0
        for ds_name, arr in saved.items():
            if ds_name not in self.fits_dict:
                self._log(f"[FitsDict] companion has unknown dataset '{ds_name}', skipping")
                continue
            expected_shape = tuple(self.fits_dict[ds_name].shape)
            if arr.shape != expected_shape:
                print(f"[FitsDict] companion shape mismatch for '{ds_name}': "
                      f"got {arr.shape}, expected {expected_shape}, skipping")
                continue
            self.fits_dict[ds_name] = torch.from_numpy(arr.astype(np.float32))
            loaded_count += 1
        print(f"[FitsDict] loaded companion fits from {companion_path} "
              f"({loaded_count}/{len(self.fits_dict)} datasets)")
        return loaded_count > 0

    def rotate_pose(self, pose, rot):
        """Rotate SMPL pose parameters by rot degrees.

        GPU path: builds the batched Z-rotation R and the global-pose rotation
        matrix on `pose.device` (via `batch_rodrigues`), composes them via
        torch.matmul, and only falls back to CPU+cv2.Rodrigues for the final
        rotmat -> axis-angle step (matches SPIN's CPU fallback there).
        """
        pose = pose.clone()
        device = pose.device
        dtype = pose.dtype

        rot_t = torch.as_tensor(rot, dtype=dtype, device=device).reshape(-1)
        theta = -np.pi * rot_t / 180.0
        cos = torch.cos(theta)
        sin = torch.sin(theta)
        zeros = torch.zeros_like(cos)
        ones = torch.ones_like(cos)
        # Batched [B, 3, 3] Z-rotation matrix on device
        R = torch.stack([
            torch.stack([cos, -sin, zeros], dim=-1),
            torch.stack([sin,  cos, zeros], dim=-1),
            torch.stack([zeros, zeros, ones], dim=-1),
        ], dim=1)

        global_pose = pose[:, :3]
        global_pose_rotmat = batch_rodrigues(global_pose)  # [B, 3, 3] on device
        composed = torch.matmul(R, global_pose_rotmat)     # [B, 3, 3] on device

        # Final rotmat -> axis-angle: SPIN also uses cv2.Rodrigues per-sample
        composed_np = composed.detach().cpu().numpy().astype(np.float32)
        out_pose_np = np.zeros((composed_np.shape[0], 3), dtype=np.float32)
        for i in range(composed_np.shape[0]):
            aa_new, _ = cv2.Rodrigues(composed_np[i])
            out_pose_np[i, :] = aa_new.reshape(3)

        pose[:, :3] = torch.from_numpy(out_pose_np).to(device=device, dtype=dtype)
        return pose