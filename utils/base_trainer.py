from __future__ import division
import sys
import time
import os
import csv
from datetime import datetime

import numpy as np
import torch
from tqdm import tqdm
tqdm.monitor_interval = 0

try:
    from torch.utils.tensorboard import SummaryWriter
except Exception:
    class SummaryWriter:  # fallback no-op writer for environments without tensorboard package
        def __init__(self, *args, **kwargs):
            pass

        def add_scalar(self, *args, **kwargs):
            pass

        def add_image(self, *args, **kwargs):
            pass

        def close(self):
            pass

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from .data_loader import CheckpointDataLoader
from .saver import CheckpointSaver


class BaseTrainer(object):
    """Base class for Trainer objects.
    Handles checkpointing/logging/resuming and step-based validation.
    """

    def __init__(self, options):
        self.options = options
        self.endtime = time.time() + self.options.time_to_run
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        # override this function to define your model, optimizers etc.
        self.init_fn()

        self.saver = CheckpointSaver(save_dir=options.checkpoint_dir)
        self.summary_writer = SummaryWriter(self.options.summary_dir)

        self.checkpoint = None
        if bool(getattr(self.options, 'resume', False)):
            checkpoint_file = getattr(self.options, 'checkpoint', None)
            if checkpoint_file not in [None, ''] and self.saver.exists_checkpoint(checkpoint_file):
                self.checkpoint = self.saver.load_checkpoint(
                    self.models_dict,
                    self.optimizers_dict,
                    checkpoint_file=checkpoint_file
                )
            elif self.saver.exists_checkpoint():
                self.checkpoint = self.saver.load_checkpoint(
                    self.models_dict,
                    self.optimizers_dict,
                    checkpoint_file=None
                )

        self._resumed_checkpoint_path = getattr(self.saver, 'last_loaded_checkpoint_path', None)

        # [M7-companion-fits] load fits from companion file after checkpoint is known
        if self._resumed_checkpoint_path and hasattr(self, 'fits_dict') and self.fits_dict is not None:
            self.fits_dict.load_companion(self._resumed_checkpoint_path)

        if self.checkpoint is None:
            self.epoch_count = 0
            self.step_count = 0
        else:
            self.epoch_count = int(self.checkpoint.get('epoch', 0))
            self.step_count = int(self.checkpoint.get('total_step_count', 0))

        self.best_txt_path = os.path.join(self.options.log_dir, 'best.txt')
        self.validation_metric_name = str(getattr(self.options, 'validation_metric', 'MPJPE'))
        self.performance_type = str(self.checkpoint.get('performance_type', 'min')) if self.checkpoint else 'min'
        self.best_performance = self.checkpoint.get('best_performance', None) if self.checkpoint else None
        if self.best_performance is None:
            self.best_performance = self._load_best_from_txt()

        # Track per-metric best values for emitting metric-specific checkpoints
        # (e.g. checkpoint_best_mpjpe.pt and checkpoint_best_reproj2d.pt). All
        # tracked metrics use 'min' (lower is better).
        # Plan 06.28: in multi-dataset validation mode, metric keys arrive in
        # `<dataset>/<metric>` form (e.g. `3dpw/MPJPE`, `llvip/2D-MPJPE`,
        # `composite/thermal_projection`); we also track those automatically.
        self.tracked_best_metrics = (
            'MPJPE', 'REPROJ_2D',
            '3dpw/MPJPE', '3dpw/PA-MPJPE', '3dpw/REPROJ_2D',
            'h36m-p1/MPJPE', 'h36m-p2/MPJPE',
            'llvip/2D-MPJPE', 'llvip-pgt/2D-MPJPE',
            'otp2/2D-MPJPE', 'otp2-pgt/2D-MPJPE',
            'composite/thermal_projection',
        )
        self.best_metric_values = {}
        if self.checkpoint and isinstance(self.checkpoint.get('best_metric_values', None), dict):
            for k, v in self.checkpoint['best_metric_values'].items():
                try:
                    self.best_metric_values[str(k).upper()] = float(v)
                except Exception:
                    continue

        self.valid_step = int(getattr(self.options, 'valid_step', 0) or 0)
        self.save_step = int(getattr(self.options, 'save_step', getattr(self.options, 'checkpoint_steps', 0)) or 0)
        self.max_step = int(getattr(self.options, 'max_step', 0) or 0)
        self.validation_enabled = bool(getattr(self.options, 'validation_enabled', False))
        self.validation_save_best = bool(getattr(self.options, 'validation_save_best', True))
        if self.valid_step <= 0:
            self.validation_enabled = False

        # -----------------------------
        # step-0 baseline + delta / early-stop guard (ablation diagnostics)
        # See docs/mu_hmr2_root_cause_ablation_plan.md §2. Read from the optional
        # BASELINE config section (namespace tree, absent -> all defaults off).
        # -----------------------------
        _bl = getattr(self.options, 'BASELINE', None)
        self.baseline_run_step0 = bool(getattr(_bl, 'RUN_STEP0_VALIDATION', False))
        self.baseline_compare = bool(getattr(_bl, 'COMPARE_TO_STEP0', False))
        self.baseline_early_stop = bool(getattr(_bl, 'EARLY_STOP_ENABLED', False))
        self.baseline_es_start = int(getattr(_bl, 'EARLY_STOP_START_STEP', 0) or 0)
        self.baseline_es_end = int(getattr(_bl, 'EARLY_STOP_END_STEP', 0) or 0)
        self.baseline_rgbir_margin = float(getattr(_bl, 'RGBIR_FAIL_MARGIN_PX', 5.0) or 5.0)
        self.baseline_ir_margin = float(getattr(_bl, 'IR_FAIL_MARGIN_PX', 8.0) or 8.0)
        self.baseline_gate_collapse_thr = float(getattr(_bl, 'GATE_IR_COLLAPSE_THRESHOLD', 0.95) or 0.95)
        # If EARLY_STOP_ABORT is true, a failing guard raises (halts training).
        # Default false: log a prominent WARN but keep running so the full
        # diagnostic trajectory (e.g. E4 5k) is captured.
        self.baseline_es_abort = bool(getattr(_bl, 'EARLY_STOP_ABORT', False))
        # Long-run (E4-long) guard extensions — all default to the previous
        # single-shot behavior so existing configs are unaffected:
        #  * RGBIR_FAIL_CONSECUTIVE: require N consecutive rgbir violations (>=1).
        #  * IR_DRIFT_MARGIN_PX: |ir - step0_ir| beyond this => freeze/routing bug
        #    (0/None disables the check).
        #  * NAN_GUARD: abort on non-finite training loss (default on).
        self.baseline_rgbir_consecutive = max(1, int(getattr(_bl, 'RGBIR_FAIL_CONSECUTIVE', 1) or 1))
        _ir_drift = getattr(_bl, 'IR_DRIFT_MARGIN_PX', None)
        self.baseline_ir_drift_margin = float(_ir_drift) if _ir_drift not in (None, '', 0, 0.0) else None
        self.baseline_nan_guard = bool(getattr(_bl, 'NAN_GUARD', True))
        self._rgbir_violation_streak = 0
        self.step0_metrics = None
        self._early_stop_flagged = False

        # Disk-safety: by default _check_and_save_best keeps a timestamped
        # best_<ts>.pt (2.6G each) on every improvement — over a long run these
        # accumulate and can fill the disk. VALIDATION.SAVE_BEST_TIMESTAMPED can
        # turn that off (keeping only the overwriting checkpoint_best.pt).
        _val_cfg = getattr(self.options, 'VALIDATION', None)
        self.save_best_timestamped = bool(getattr(_val_cfg, 'SAVE_BEST_TIMESTAMPED', True)) \
            if _val_cfg is not None else True

        # -----------------------------
        # loss/validation monitor setup
        # -----------------------------
        self.loss_record_steps = 50
        self.loss_plot_steps = 500

        self.loss_monitor_dir = os.path.join(self.options.log_dir, 'train_monitor')
        os.makedirs(self.loss_monitor_dir, exist_ok=True)

        self.loss_csv_path = os.path.join(self.loss_monitor_dir, 'train_loss_log.csv')
        self.validation_csv_path = os.path.join(self.loss_monitor_dir, 'validation_log.csv')
        self.total_loss_plot_path = os.path.join(self.loss_monitor_dir, 'total_loss_vs_iteration.png')
        self.weighted_loss_plot_path = os.path.join(self.loss_monitor_dir, 'weighted_losses_vs_iteration.png')
        self.validation_plot_path = os.path.join(self.loss_monitor_dir, 'validation_metrics_vs_iteration.png')

        self.loss_history = []
        self._load_existing_loss_history()
        self.validation_history = []
        self._load_existing_validation_history()

    def _load_best_from_txt(self):
        if not os.path.isfile(self.best_txt_path):
            return None
        try:
            with open(self.best_txt_path, 'r', encoding='utf-8') as f:
                text = f.read().strip()
            if text == '':
                return None
            return float(text)
        except Exception:
            return None

    def _load_existing_loss_history(self):
        """Load existing CSV so resumed training continues the same plots.

        [safe-guard] After loading, drop any row whose global_step is ahead of
        the resumed self.step_count. This prevents plot time-axis reversal or
        overlap when resuming from an earlier checkpoint while a newer CSV
        (from a later, now-discarded run) still sits on disk. We only mutate
        the in-memory self.loss_history; the CSV file is left untouched so
        future appends continue the same file.
        """
        # 防止未來 __init__ 被重排而導致 step_count 尚未設定時靜默 no-op
        assert hasattr(self, 'step_count'), (
            '_load_existing_loss_history must be called after '
            'self.step_count is set; otherwise the resume-point '
            'truncation will silently no-op.'
        )
        if not os.path.isfile(self.loss_csv_path):
            return

        try:
            with open(self.loss_csv_path, 'r', newline='') as f:
                reader = csv.DictReader(f)
                for row in reader:
                    parsed = {}
                    for k, v in row.items():
                        if k == 'timestamp':
                            parsed[k] = v
                        else:
                            try:
                                if k in ['epoch', 'iter_in_epoch', 'global_step']:
                                    parsed[k] = int(float(v))
                                else:
                                    parsed[k] = float(v)
                            except Exception:
                                parsed[k] = v
                    self.loss_history.append(parsed)
        except Exception as e:
            print(f'[LossMonitor] Warning: failed to load existing loss csv: {e}')
            self.loss_history = []
            return

        n_loaded = len(self.loss_history)
        resume_step = int(getattr(self, 'step_count', 0) or 0)
        # Rows without a usable global_step are kept (cannot compare); rows with
        # global_step > resume_step are dropped.
        self.loss_history = [
            row for row in self.loss_history
            if not isinstance(row.get('global_step'), int) or row['global_step'] <= resume_step
        ]
        n_kept = len(self.loss_history)
        print(f'[LossMonitor] loaded {n_loaded} rows, kept {n_kept} rows after '
              f'truncation to step={resume_step}')

    def _load_existing_validation_history(self):
        if not os.path.isfile(self.validation_csv_path):
            return
        try:
            with open(self.validation_csv_path, 'r', newline='') as f:
                reader = csv.DictReader(f)
                for row in reader:
                    parsed = {}
                    for k, v in row.items():
                        if k == 'timestamp':
                            parsed[k] = v
                        else:
                            try:
                                parsed[k] = int(float(v)) if k in ('epoch', 'iter_in_epoch', 'global_step') else float(v)
                            except Exception:
                                parsed[k] = v
                    self.validation_history.append(parsed)
        except Exception as e:
            print(f'[ValMonitor] Warning: failed to load existing validation csv: {e}')
            self.validation_history = []
            return

        resume_step = int(getattr(self, 'step_count', 0) or 0)
        self.validation_history = [
            row for row in self.validation_history
            if not isinstance(row.get('global_step'), int) or row['global_step'] <= resume_step
        ]
        print(f'[ValMonitor] loaded {len(self.validation_history)} validation rows')

    @staticmethod
    def _append_csv_row(path, row):
        file_exists = os.path.isfile(path)
        need_header = (not file_exists) or os.path.getsize(path) == 0
        with open(path, 'a', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=list(row.keys()))
            if need_header:
                writer.writeheader()
            writer.writerow(row)

    def _build_loss_row(self, epoch, step, losses):
        lr = 0.0
        if hasattr(self, 'optimizer') and self.optimizer is not None:
            lr = float(self.optimizer.param_groups[0]['lr'])

        row = {
            'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            'epoch': int(epoch),
            'iter_in_epoch': int(step),
            'global_step': int(self.step_count),
            'lr': float(lr),
        }

        for k, v in losses.items():
            try:
                row[k] = float(v)
            except Exception:
                row[k] = v
        return row

    def _maybe_print_losses(self, epoch, step, losses):
        return

    def _maybe_record_losses(self, epoch, step, losses):
        if self.loss_record_steps <= 0:
            return
        if self.step_count % self.loss_record_steps != 0:
            return

        row = self._build_loss_row(epoch, step, losses)
        self._append_csv_row(self.loss_csv_path, row)
        self.loss_history.append(row)

        if self.loss_plot_steps > 0 and self.step_count % self.loss_plot_steps == 0:
            self._update_loss_plots()

    def _update_loss_plots(self):
        if len(self.loss_history) == 0:
            return

        steps = [row.get('global_step', i + 1) for i, row in enumerate(self.loss_history)]
        total_loss = [row.get('loss', float('nan')) for row in self.loss_history]

        plt.figure(figsize=(10, 5))
        plt.plot(steps, total_loss, linewidth=1.5)
        plt.xlabel('Iteration')
        plt.ylabel('Loss')
        plt.title('Total Loss vs Iteration')
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(self.total_loss_plot_path, dpi=150)
        plt.close()

        weighted_keys = [
            'weighted_loss_keypoints_2d',
            'weighted_loss_keypoints_3d',
            'weighted_loss_regr_pose',
            'weighted_loss_regr_betas',
            'weighted_loss_shape',
            'weighted_loss_depth',
            'weighted_loss_cam',
            'weighted_loss_cam_full',
        ]

        plt.figure(figsize=(11, 6))
        plotted_any = False
        for key in weighted_keys:
            vals = [row.get(key, float('nan')) for row in self.loss_history]
            has_valid = any([(v == v) for v in vals])  # NaN check
            if has_valid:
                plt.plot(steps, vals, linewidth=1.2, label=key)
                plotted_any = True

        plt.xlabel('Iteration')
        plt.ylabel('Loss')
        plt.title('Weighted Loss Components vs Iteration')
        plt.grid(True, alpha=0.3)
        if plotted_any:
            plt.legend()
        plt.tight_layout()
        plt.savefig(self.weighted_loss_plot_path, dpi=150)
        plt.close()

    def _update_validation_plots(self):
        if len(self.validation_history) == 0:
            return

        steps = [row.get('global_step', i + 1) for i, row in enumerate(self.validation_history)]

        # Collect all numeric metric columns (exclude bookkeeping columns)
        _skip = {'timestamp', 'epoch', 'iter_in_epoch', 'global_step', 'performance', 'performance_type'}
        all_keys = []
        for row in self.validation_history:
            for k in row:
                if k not in _skip and k not in all_keys:
                    try:
                        float(row[k])
                        all_keys.append(k)
                    except Exception:
                        pass

        if not all_keys:
            return

        # Group by metric suffix (e.g., 'MPJPE', 'PA-MPJPE', '2D-MPJPE', 'REPROJ_2D')
        groups = {}
        for key in all_keys:
            if '/' in key:
                suffix = key.split('/', 1)[1]
            else:
                suffix = key
            groups.setdefault(suffix, []).append(key)

        n_groups = len(groups)
        if n_groups == 0:
            return

        fig, axes = plt.subplots(n_groups, 1, figsize=(11, 4 * n_groups), squeeze=False)

        primary = getattr(self.options, 'primary_metric', None) or getattr(
            getattr(self.options, 'VALIDATION', None), 'PRIMARY_METRIC', None)

        for ax, (suffix, keys) in zip(axes[:, 0], sorted(groups.items())):
            plotted_any = False
            for key in sorted(keys):
                vals = []
                for row in self.validation_history:
                    try:
                        vals.append(float(row[key]))
                    except Exception:
                        vals.append(float('nan'))
                has_valid = any(v == v for v in vals)
                if not has_valid:
                    continue
                lw = 2.0 if key == primary else 1.2
                ls = '--' if key == primary else '-'
                ax.plot(steps, vals, linewidth=lw, linestyle=ls, label=key)
                plotted_any = True
            ax.set_title(suffix)
            ax.set_xlabel('Iteration')
            ax.set_ylabel('mm' if 'MPJPE' in suffix.upper() else 'px')
            ax.grid(True, alpha=0.3)
            if plotted_any:
                ax.legend(fontsize=8)

        fig.suptitle('Validation Metrics vs Iteration', fontsize=12)
        plt.tight_layout()
        plt.savefig(self.validation_plot_path, dpi=150)
        plt.close()

    def _save_checkpoint(self, epoch, batch_idx, dataset_perm=None, checkpoint_file=None):
        self.finalize()
        extra_state = {
            'best_performance': self.best_performance,
            'performance_type': self.performance_type,
            'validation_metric_name': self.validation_metric_name,
            'best_metric_values': getattr(self, 'best_metric_values', {}),
        }
        # [M7-companion-fits] 接收實際儲存路徑，並存 companion fits
        actual_ckpt_path = self.saver.save_checkpoint(
            self.models_dict,
            self.optimizers_dict,
            epoch,
            batch_idx,
            self.options.batch_size,
            dataset_perm,
            self.step_count,
            extra_state=extra_state,
            checkpoint_file=checkpoint_file,
        )
        if actual_ckpt_path and hasattr(self, 'fits_dict') and self.fits_dict is not None:
            self.fits_dict.save_companion(actual_ckpt_path)

    def _extract_validation(self, validation_result):
        if validation_result is None:
            return None, {}
        if isinstance(validation_result, dict):
            metrics = dict(validation_result)
            performance = metrics.get('performance', None)
            if performance is None:
                if self.validation_metric_name in metrics:
                    performance = metrics[self.validation_metric_name]
                elif self.validation_metric_name.upper() in metrics:
                    performance = metrics[self.validation_metric_name.upper()]
                elif self.validation_metric_name.lower() in metrics:
                    performance = metrics[self.validation_metric_name.lower()]
            if 'performance_type' in metrics and metrics['performance_type'] is not None:
                self.performance_type = str(metrics['performance_type'])
            return performance, metrics
        try:
            performance = float(validation_result)
            return performance, {self.validation_metric_name: performance}
        except Exception:
            return None, {}

    def _record_validation_metrics(self, epoch, step, metrics):
        if not isinstance(metrics, dict) or len(metrics) == 0:
            return

        row = {
            'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            'epoch': int(epoch),
            'iter_in_epoch': int(step),
            'global_step': int(self.step_count),
        }
        for k, v in metrics.items():
            if k in ('performance_type',):
                continue
            try:
                val = float(v)
            except Exception:
                continue
            row[k] = val
            self.summary_writer.add_scalar(f'Validation/{k}', val, self.step_count)

        if len(row) > 4:
            self._append_csv_row(self.validation_csv_path, row)
            self.validation_history.append(row)
            self._update_validation_plots()

    def _check_and_save_best(self, performance, epoch, step, dataset_perm=None, metrics=None):
        # Always refresh checkpoint_last.pt after each validation, regardless of
        # whether performance improved. This guarantees a recoverable snapshot
        # even when training crashes between checkpoint_steps boundaries.
        os.makedirs(self.options.checkpoint_dir, exist_ok=True)
        last_ckpt = os.path.join(self.options.checkpoint_dir, 'checkpoint_last.pt')
        self._save_checkpoint(
            epoch=epoch,
            batch_idx=step + 1,
            dataset_perm=dataset_perm,
            checkpoint_file=last_ckpt,
        )

        # Per-metric "best" checkpoints — one file per tracked metric.
        # All tracked metrics use 'min' (lower is better).
        if isinstance(metrics, dict):
            # Build case-insensitive lookup so 'mpjpe', 'MPJPE', '3dpw/mpjpe' all hit.
            metrics_upper = {str(k).upper(): v for k, v in metrics.items()}
            for raw_name in self.tracked_best_metrics:
                key_u = str(raw_name).upper()
                if key_u not in metrics_upper:
                    continue
                try:
                    val = float(metrics_upper[key_u])
                except Exception:
                    continue
                if not np.isfinite(val):
                    continue
                prev = self.best_metric_values.get(key_u, None)
                if prev is None or val < prev:
                    self.best_metric_values[key_u] = val
                    # Sanitize key for filesystem: '3dpw/MPJPE' -> '3dpw_mpjpe'
                    safe = key_u.lower().replace('/', '_').replace(' ', '_').replace('-', '_')
                    metric_ckpt = os.path.join(
                        self.options.checkpoint_dir,
                        f'checkpoint_best_{safe}.pt',
                    )
                    print(f'[Validation] New best {raw_name}: {val:.6f} -> {metric_ckpt}')
                    self._save_checkpoint(
                        epoch=epoch,
                        batch_idx=step + 1,
                        dataset_perm=dataset_perm,
                        checkpoint_file=metric_ckpt,
                    )

        if performance is None:
            return False

        performance = float(performance)
        is_best = False
        if self.performance_type == 'max':
            if self.best_performance is None or performance > float(self.best_performance):
                is_best = True
        else:
            if self.best_performance is None or performance < float(self.best_performance):
                is_best = True

        if not is_best:
            return False

        self.best_performance = performance
        os.makedirs(self.options.log_dir, exist_ok=True)
        with open(self.best_txt_path, 'w', encoding='utf-8') as f:
            f.write(str(float(performance)))

        print(f'[Validation] New best {self.validation_metric_name}: {performance:.6f}')
        if self.validation_save_best:
            best_ckpt = os.path.join(self.options.checkpoint_dir, 'checkpoint_best.pt')
            self._save_checkpoint(
                epoch=epoch,
                batch_idx=step + 1,
                dataset_perm=dataset_perm,
                checkpoint_file=best_ckpt,
            )
            # Preserve every best refresh as a timestamped snapshot so later evaluation
            # can pick across multiple "best" points instead of only the latest one.
            # Disabled via VALIDATION.SAVE_BEST_TIMESTAMPED=false to avoid the 2.6G
            # -per-improvement disk accumulation on long runs.
            if self.save_best_timestamped:
                ts = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
                ts_ckpt = os.path.join(self.options.checkpoint_dir, f'best_{ts}.pt')
                self._save_checkpoint(
                    epoch=epoch,
                    batch_idx=step + 1,
                    dataset_perm=dataset_perm,
                    checkpoint_file=ts_ckpt,
                )
        return True

    def load_pretrained(self, checkpoint_file=None):
        """Load full HMR state dict from a pretrained checkpoint.

        This runs AFTER the backbone (ViTPose / HRNet) is built and its own
        pretrained weights are loaded inside the model constructor, so the
        Stage-1 checkpoint here overrides the entire HMR — backbone, regression
        head, and camera head all at once. Missing / unexpected keys are logged
        so silent mismatches are caught early.
        """
        if checkpoint_file is None:
            return

        checkpoint = torch.load(checkpoint_file, map_location=self.device)
        loaded_any = False

        for model in self.models_dict:
            if model in checkpoint:
                src_sd = checkpoint[model]
                load_result = self.models_dict[model].load_state_dict(src_sd, strict=False)
                missing = list(getattr(load_result, 'missing_keys', []) or [])
                unexpected = list(getattr(load_result, 'unexpected_keys', []) or [])
                n_src = len(src_sd) if hasattr(src_sd, '__len__') else -1
                print(
                    f"[Init] pretrained '{model}' loaded from {checkpoint_file} | "
                    f"ckpt_keys={n_src} | missing={len(missing)} | unexpected={len(unexpected)}"
                )
                if missing:
                    print(f"  [Init] missing[:10] = {missing[:10]}")
                if unexpected:
                    print(f"  [Init] unexpected[:10] = {unexpected[:10]}")
                loaded_any = True

        if loaded_any:
            print(f"[Init] pretrained_checkpoint_loaded={checkpoint_file}")
        else:
            print(f"[Init][WARN] pretrained_checkpoint specified but no matching key "
                  f"in checkpoint dict (expected keys: {list(self.models_dict.keys())}). "
                  f"File={checkpoint_file}")

    def _maybe_run_step0_baseline(self):
        """Run a no-train validation before step 1 and cache the metrics.

        Stores the flat metric dict on ``self.step0_metrics`` so later
        validations can report delta-vs-step0 and the early-stop guard can
        compare against the untrained prior. Safe no-op when disabled or when
        validation is not wired for this run.
        """
        if not self.baseline_run_step0 or not self.validation_enabled:
            return
        if self.step0_metrics is not None:  # e.g. resumed run
            return
        print('[Step0Baseline] running no-train validation before training ...')
        try:
            result = self.validate()
        except Exception as e:  # never let baseline probing kill the run
            print(f'[Step0Baseline][WARN] validation failed, skipping baseline: {e}')
            return
        if not isinstance(result, dict):
            print('[Step0Baseline][WARN] validate() did not return a dict; skipping.')
            return
        self.step0_metrics = {
            k: float(v) for k, v in result.items()
            if isinstance(v, (int, float)) and k not in ('performance', 'performance_type')
        }
        for key in ('llvip-pgt/rgb/2D-MPJPE', 'llvip-pgt/ir/2D-MPJPE',
                    'llvip-pgt/rgbir/2D-MPJPE'):
            if key in self.step0_metrics:
                print(f'[Step0Baseline] {key}={self.step0_metrics[key]:.2f}')

    def _annotate_and_guard_step0(self, metrics):
        """Add delta-vs-step0 keys to ``metrics`` and evaluate the early-stop guard.

        2D-MPJPE is lower-is-better, so ``delta = current - step0``; a POSITIVE
        delta means the model got WORSE than the untrained prior. Returns True
        if the guard fired (only aborts when EARLY_STOP_ABORT is set).
        """
        if not self.baseline_compare or not self.step0_metrics or not isinstance(metrics, dict):
            return False
        for key, val in list(metrics.items()):
            if key.endswith('/2D-MPJPE') and key in self.step0_metrics:
                try:
                    metrics[key + '_delta_vs_step0'] = float(val) - self.step0_metrics[key]
                except (TypeError, ValueError):
                    continue
        rgbir_key = 'llvip-pgt/rgbir/2D-MPJPE'
        ir_key = 'llvip-pgt/ir/2D-MPJPE'
        gate_key = 'llvip-pgt/rgbir/gate_ir_mean'
        d_rgbir = metrics.get(rgbir_key + '_delta_vs_step0')
        d_ir = metrics.get(ir_key + '_delta_vs_step0')
        gate_ir = metrics.get(gate_key, None)
        if d_rgbir is not None:
            print(f'[Validation] step={self.step_count} | '
                  f'{rgbir_key}={metrics.get(rgbir_key, float("nan")):.2f} | '
                  f'delta_vs_step0={d_rgbir:+.2f} px')
        if not self.baseline_early_stop:
            return False
        in_window = (self.baseline_es_start <= self.step_count <= self.baseline_es_end
                     if self.baseline_es_end > 0 else self.step_count >= self.baseline_es_start)
        if not in_window:
            return False

        # Track consecutive rgbir violations (condition 2: needs N in a row).
        rgbir_over = d_rgbir is not None and d_rgbir > self.baseline_rgbir_margin
        self._rgbir_violation_streak = (self._rgbir_violation_streak + 1) if rgbir_over else 0

        reasons = []
        # (2) rgbir worse than step0+margin for N consecutive validations.
        if rgbir_over and self._rgbir_violation_streak >= self.baseline_rgbir_consecutive:
            reasons.append(
                f'rgbir delta {d_rgbir:+.2f}px > margin {self.baseline_rgbir_margin:.1f} '
                f'for {self._rgbir_violation_streak} consecutive validations')
        # (3) gate collapsed to IR AND rgbir degraded.
        if (gate_ir is not None and float(gate_ir) > self.baseline_gate_collapse_thr
                and d_rgbir is not None and d_rgbir > 0):
            reasons.append(
                f'gate_ir {float(gate_ir):.3f} > {self.baseline_gate_collapse_thr:.2f} '
                f'AND rgbir degraded (delta {d_rgbir:+.2f}px)')
        # (4) IR-only drifted from step0_ir (freeze / validation-routing bug).
        if self.baseline_ir_drift_margin is not None and d_ir is not None \
                and abs(d_ir) > self.baseline_ir_drift_margin:
            reasons.append(
                f'ir-only drifted {d_ir:+.2f}px from step-0 (|delta| > '
                f'{self.baseline_ir_drift_margin:.1f}) — adapter-freeze or routing bug')

        if not reasons:
            return False
        msg = (f'[EarlyStop] step={self.step_count}: ' + '; '.join(reasons) +
               f' | rgbir={metrics.get(rgbir_key, float("nan")):.2f}, '
               f'ir={metrics.get(ir_key, float("nan")):.2f}, '
               f'gate_ir={float(gate_ir) if gate_ir is not None else float("nan"):.3f}.')
        self._early_stop_flagged = True
        if self.baseline_es_abort:
            raise RuntimeError(msg)
        print('[EarlyStop][WARN] ' + msg +
              ' Continuing (EARLY_STOP_ABORT=false).')
        return True

    def train(self):
        stop_training = False
        last_epoch = self.epoch_count

        # Optional no-train baseline before any optimizer step (ablation runs).
        self._maybe_run_step0_baseline()

        for epoch in range(self.epoch_count, self.options.num_epochs):
            last_epoch = epoch
            train_data_loader = CheckpointDataLoader(
                self.train_ds,
                checkpoint=self.checkpoint,
                batch_size=self.options.batch_size,
                num_workers=self.options.num_workers,
                pin_memory=self.options.pin_memory,
                shuffle=self.options.shuffle_train
            )

            if epoch == self.epoch_count:
                steps_per_epoch = len(train_data_loader)
                expected_total = steps_per_epoch * self.options.num_epochs
                print(f'[LR] steps_per_epoch={steps_per_epoch} | '
                      f'num_epochs={self.options.num_epochs} | '
                      f'expected_total={expected_total} | '
                      f'max_step={self.max_step}')
                if self.max_step > 0 and abs(self.max_step - expected_total) / max(expected_total, 1) > 0.1:
                    print(f'[LR][WARN] max_step={self.max_step} differs from '
                          f'expected_total={expected_total} by >10%; '
                          f'cosine schedule may not reach ~0 at training end')

            pbar = tqdm(
                train_data_loader,
                desc=f'Epoch {epoch}',
                total=max(
                    len(train_data_loader),
                    int(len(train_data_loader.sampler.dataset_perm) // self.options.batch_size)
                ),
                initial=train_data_loader.checkpoint_batch_idx,
                dynamic_ncols=True,
                leave=True,
                mininterval=1.0
            )

            for step, batch in enumerate(pbar, train_data_loader.checkpoint_batch_idx):
                if time.time() >= self.endtime:
                    self._update_loss_plots()
                    self._save_checkpoint(
                        epoch=epoch,
                        batch_idx=step,
                        dataset_perm=train_data_loader.sampler.dataset_perm,
                    )
                    self.summary_writer.close()
                    sys.exit(0)

                batch = {
                    k: v.to(self.device, non_blocking=True) if isinstance(v, torch.Tensor) else v
                    for k, v in batch.items()
                }

                output, losses = self.train_step(batch)
                self.step_count += 1

                # (1) NaN/Inf training-loss guard: save a snapshot and abort so a
                # long unattended run does not silently continue on garbage.
                if getattr(self, 'baseline_nan_guard', True):
                    _lv = losses.get('loss') if isinstance(losses, dict) else None
                    if _lv is not None and not np.isfinite(float(_lv)):
                        print(f'[EarlyStop][NaN] non-finite training loss={_lv} at '
                              f'step={self.step_count}; saving checkpoint and aborting.')
                        self._save_checkpoint(
                            epoch=epoch, batch_idx=step + 1,
                            dataset_perm=train_data_loader.sampler.dataset_perm,
                        )
                        self._early_stop_flagged = True
                        raise RuntimeError(
                            f'[EarlyStop][NaN] non-finite training loss at step={self.step_count}')

                if hasattr(self, 'lr_scheduler') and self.lr_scheduler is not None:
                    self.lr_scheduler.step()

                self._maybe_record_losses(epoch, step, losses)
                self._maybe_print_losses(epoch, step, losses)

                if self.options.summary_steps > 0 and self.step_count % self.options.summary_steps == 0:
                    self.train_summaries(batch, output, losses)

                if self.validation_enabled and self.valid_step > 0 and self.step_count % self.valid_step == 0:
                    validation_result = self.validate()
                    performance, metrics = self._extract_validation(validation_result)
                    self._annotate_and_guard_step0(metrics)
                    self._record_validation_metrics(epoch, step, metrics)
                    self._check_and_save_best(
                        performance=performance,
                        epoch=epoch,
                        step=step,
                        dataset_perm=train_data_loader.sampler.dataset_perm,
                        metrics=metrics,
                    )

                if self.save_step > 0 and self.step_count % self.save_step == 0:
                    self._save_checkpoint(
                        epoch=epoch,
                        batch_idx=step + 1,
                        dataset_perm=train_data_loader.sampler.dataset_perm,
                    )

                # [maintainability] self.test() is a no-op in BaseTrainer and not
                # overridden; validation is driven exclusively by VALIDATION.INTERVAL
                # (self.valid_step) above. LOGGING.TEST_STEPS is kept only as a
                # legacy fallback for VALIDATION.INTERVAL inside train_options.py.

                if self.max_step > 0 and self.step_count >= self.max_step:
                    stop_training = True
                    break

            self.checkpoint = None
            self._update_loss_plots()
            if stop_training:
                break

        self.finalize()
        self._update_loss_plots()
        self._save_checkpoint(
            epoch=last_epoch + 1,
            batch_idx=0,
            dataset_perm=None,
        )

        self.summary_writer.close()
        return

    # The following methods (with the possible exception of test) have to be implemented in the derived classes
    def init_fn(self):
        raise NotImplementedError('You need to provide an _init_fn method')

    def train_step(self, input_batch):
        raise NotImplementedError('You need to provide a _train_step method')

    def train_summaries(self, input_batch):
        raise NotImplementedError('You need to provide a _train_summaries method')

    def validate(self):
        return None

    def test(self):
        pass
