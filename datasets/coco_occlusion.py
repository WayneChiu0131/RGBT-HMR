import os
import random

import cv2
import joblib
import numpy as np


def resize_by_factor(im, factor):
    h, w = im.shape[:2]
    new_w = max(1, int(round(w * factor)))
    new_h = max(1, int(round(h * factor)))
    return cv2.resize(im, (new_w, new_h), interpolation=cv2.INTER_LINEAR)


def paste_over(im_src, im_dst, center):
    """
    Alpha paste RGBA source over RGB destination.
    center: (x, y) on destination image.
    """
    if im_src.shape[-1] != 4:
        alpha = np.ones((*im_src.shape[:2], 1), dtype=im_src.dtype) * 255
        im_src = np.concatenate([im_src, alpha], axis=-1)

    src_h, src_w = im_src.shape[:2]
    dst_h, dst_w = im_dst.shape[:2]
    cx, cy = int(center[0]), int(center[1])

    x1 = cx - src_w // 2
    y1 = cy - src_h // 2
    x2 = x1 + src_w
    y2 = y1 + src_h

    clip_x1 = max(0, x1)
    clip_y1 = max(0, y1)
    clip_x2 = min(dst_w, x2)
    clip_y2 = min(dst_h, y2)
    if clip_x1 >= clip_x2 or clip_y1 >= clip_y2:
        return im_dst

    src_x1 = clip_x1 - x1
    src_y1 = clip_y1 - y1
    src_x2 = src_x1 + (clip_x2 - clip_x1)
    src_y2 = src_y1 + (clip_y2 - clip_y1)

    src_roi = im_src[src_y1:src_y2, src_x1:src_x2]
    dst_roi = im_dst[clip_y1:clip_y2, clip_x1:clip_x2]
    if src_roi.size == 0 or dst_roi.size == 0:
        return im_dst

    src_rgb = src_roi[..., :3].astype(np.float32)
    src_alpha = (src_roi[..., 3:4].astype(np.float32) / 255.0)
    if float(src_alpha.max()) <= 0.0:
        return im_dst
    dst_rgb = dst_roi.astype(np.float32)
    out = src_alpha * src_rgb + (1.0 - src_alpha) * dst_rgb
    im_dst[clip_y1:clip_y2, clip_x1:clip_x2] = out.astype(np.uint8)
    return im_dst


def load_pascal_occluders(path=None, default=None):
    """
    Load precomputed Pascal occluders from pkl.
    """
    if path is not None and os.path.exists(path):
        try:
            return joblib.load(path)
        except Exception:
            return [] if default is None else default
    return [] if default is None else default


def occlude_with_pascal_objects(im, occluders):
    """
    Apply synthetic occlusion to an HWC uint8 image using preloaded occluders.
    """
    if im.dtype != np.uint8:
        im = np.clip(im, 0, 255).astype(np.uint8)
    if occluders is None or len(occluders) == 0:
        return im

    out = im.copy()
    num_occ = random.randint(1, 4)
    h, w = out.shape[:2]
    base = 0.5 * (h + w)

    for _ in range(num_occ):
        occ = random.choice(occluders)
        if occ is None:
            continue
        occ_arr = np.asarray(occ)
        if occ_arr.ndim != 3 or occ_arr.shape[2] not in (3, 4):
            continue
        if occ_arr.shape[2] == 3:
            alpha = np.ones((*occ_arr.shape[:2], 1), dtype=occ_arr.dtype) * 255
            occ_arr = np.concatenate([occ_arr, alpha], axis=-1)

        factor = random.uniform(0.15, 0.6) * (base / max(1.0, max(occ_arr.shape[:2])))
        occ_resized = resize_by_factor(occ_arr, factor)
        if occ_resized.shape[0] < 2 or occ_resized.shape[1] < 2:
            continue
        center = (random.randint(0, max(0, w - 1)), random.randint(0, max(0, h - 1)))
        out = paste_over(occ_resized, out, center)

    return out.astype(np.uint8)
