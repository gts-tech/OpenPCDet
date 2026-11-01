"""
Rotated-IoU AP evaluator for 3D detection -- a numba-free alternative metric.

Why this exists
---------------
OpenPCDet's kitti_object_eval_python computes rotated-box IoU with a numba.cuda
kernel (rotate_iou.py). On NVIDIA Blackwell (sm_120, e.g. the RTX 50-series) that
kernel does not build: numba >= 0.68 (needed for sm_120) rejects the eager-signature
device functions, while numba <= 0.58 (which accepts them) emits PTX for an arch it
cannot target and segfaults on the first kernel. Net effect: the packaged AP metric
is unavailable on current-gen hardware.

AP does not depend on that kernel. It is greedy, score-ordered prediction<->GT
matching at an IoU threshold, a precision-recall curve, and an integration rule. This
module computes AP directly and takes the rotated IoU from a swappable backend,
defaulting to OpenPCDet's own compiled CUDA op (pcdet.ops.iou3d_nms, C++/CUDA -- not
numba), so the metric runs on any architecture the rest of the library already builds
for, Blackwell included. A dependency-free numpy backend (exact Sutherland-Hodgman
convex-rect clipping) is included for CPU cross-checks and CI.

Reports BEV and 3D AP per class (all-point or 11-point integration). Drop-in on the
(det_annos, gt_annos) lists OpenPCDet already assembles for evaluation.

Box convention everywhere: [x, y, z, dx, dy, dz, heading], z = CENTER height
(matches boxes_iou3d_gpu: zmax=z+dz/2, zmin=z-dz/2).

Inputs:
  det_annos : list of per-frame prediction dicts (OpenPCDet result.pkl format):
      {'name':(N,)U, 'score':(N,)f, 'boxes_lidar':(N,7)f, 'pred_labels':(N,)i, 'frame_id':str}
  gt_annos  : list of per-frame GT dicts (OpenPCDet 'annos' format):
      {'name':(M,)U, 'gt_boxes_lidar':(M,7)f}   # gt box key name is configurable
Frames are aligned by list position (det_annos[i] <-> gt_annos[i]); the companion
runner's build_gt() aligns prediction frames to GT infos by frame_id.

No dependency on numba. IoU backend is pluggable (see IOU_BACKENDS).
"""

import numpy as np


# --------------------------------------------------------------------------- #
# IoU backends -- each returns an (N, M) IoU matrix for boxes_a (N,7), boxes_b (M,7)
# --------------------------------------------------------------------------- #

def _rect_corners(boxes):
    """(K,7)->(K,4,2) BEV corners, CCW. Uses dx along heading, dy perpendicular."""
    x, y, dx, dy, yaw = boxes[:, 0], boxes[:, 1], boxes[:, 3], boxes[:, 4], boxes[:, 6]
    c, s = np.cos(yaw), np.sin(yaw)
    # local corners (half extents), CCW: (+,+)->(-,+)->(-,-)->(+,-)
    lx = np.stack([dx / 2, -dx / 2, -dx / 2, dx / 2], axis=1)  # (K,4)
    ly = np.stack([dy / 2, dy / 2, -dy / 2, -dy / 2], axis=1)
    cx = x[:, None] + lx * c[:, None] - ly * s[:, None]
    cy = y[:, None] + lx * s[:, None] + ly * c[:, None]
    return np.stack([cx, cy], axis=2)  # (K,4,2)


def _poly_area(poly):
    """Shoelace area of a polygon given as (P,2)."""
    if len(poly) < 3:
        return 0.0
    x = poly[:, 0]; y = poly[:, 1]
    return 0.5 * abs(np.dot(x, np.roll(y, -1)) - np.dot(y, np.roll(x, -1)))


def _clip_poly(subject, clip):
    """Sutherland-Hodgman: intersection polygon of convex `subject` by convex `clip`.
    Both given as (P,2) CCW. Returns (Q,2) array (possibly empty)."""
    def inside(p, a, b):
        return (b[0] - a[0]) * (p[1] - a[1]) - (b[1] - a[1]) * (p[0] - a[0]) >= -1e-12

    def inter(p1, p2, a, b):
        x1, y1 = p1; x2, y2 = p2; x3, y3 = a; x4, y4 = b
        den = (x1 - x2) * (y3 - y4) - (y1 - y2) * (x3 - x4)
        if abs(den) < 1e-12:
            return p2
        t = ((x1 - x3) * (y3 - y4) - (y1 - y3) * (x3 - x4)) / den
        return np.array([x1 + t * (x2 - x1), y1 + t * (y2 - y1)])

    output = list(subject)
    n = len(clip)
    for i in range(n):
        a, b = clip[i], clip[(i + 1) % n]
        inp = output
        output = []
        if not inp:
            break
        s = inp[-1]
        for e in inp:
            if inside(e, a, b):
                if not inside(s, a, b):
                    output.append(inter(s, e, a, b))
                output.append(e)
            elif inside(s, a, b):
                output.append(inter(s, e, a, b))
            s = e
    return np.array(output) if output else np.zeros((0, 2))


def iou_numpy(boxes_a, boxes_b, mode='3d'):
    """numba-free rotated IoU via Sutherland-Hodgman. mode in {'bev','3d'}.
    Reference backend: exact for convex rects, validates the AP core without a GPU."""
    boxes_a = np.asarray(boxes_a, dtype=np.float64)
    boxes_b = np.asarray(boxes_b, dtype=np.float64)
    N, M = boxes_a.shape[0], boxes_b.shape[0]
    out = np.zeros((N, M), dtype=np.float64)
    if N == 0 or M == 0:
        return out
    ca = _rect_corners(boxes_a); cb = _rect_corners(boxes_b)
    area_a = boxes_a[:, 3] * boxes_a[:, 4]
    area_b = boxes_b[:, 3] * boxes_b[:, 4]
    za1 = boxes_a[:, 2] - boxes_a[:, 5] / 2; za2 = boxes_a[:, 2] + boxes_a[:, 5] / 2
    zb1 = boxes_b[:, 2] - boxes_b[:, 5] / 2; zb2 = boxes_b[:, 2] + boxes_b[:, 5] / 2
    for i in range(N):
        for j in range(M):
            inter_poly = _clip_poly(ca[i], cb[j])
            inter_area = _poly_area(inter_poly)
            if inter_area <= 0:
                continue
            if mode == 'bev':
                union = area_a[i] + area_b[j] - inter_area
                out[i, j] = inter_area / max(union, 1e-6)
            else:  # 3d
                h = max(0.0, min(za2[i], zb2[j]) - max(za1[i], zb1[j]))
                inter_vol = inter_area * h
                vol_a = area_a[i] * boxes_a[i, 5]
                vol_b = area_b[j] * boxes_b[j, 5]
                out[i, j] = inter_vol / max(vol_a + vol_b - inter_vol, 1e-6)
    return out


def iou_pcdet_gpu(boxes_a, boxes_b, mode='3d'):
    """Container backend: reuse the compiled CUDA ops (NO numba). Requires a GPU + pcdet.
       mode='3d'  -> iou3d_nms_utils.boxes_iou3d_gpu
       mode='bev' -> iou3d_nms_utils.boxes_iou_bev
    """
    import torch
    from pcdet.ops.iou3d_nms import iou3d_nms_utils
    a = torch.from_numpy(np.asarray(boxes_a, np.float32)).cuda()
    b = torch.from_numpy(np.asarray(boxes_b, np.float32)).cuda()
    if a.shape[0] == 0 or b.shape[0] == 0:
        return np.zeros((a.shape[0], b.shape[0]), np.float32)
    fn = iou3d_nms_utils.boxes_iou3d_gpu if mode == '3d' else iou3d_nms_utils.boxes_iou_bev
    return fn(a, b).cpu().numpy()


IOU_BACKENDS = {'numpy': iou_numpy, 'pcdet_gpu': iou_pcdet_gpu}


# --------------------------------------------------------------------------- #
# AP core
# --------------------------------------------------------------------------- #

def _ap_from_pr(rec, prec, method='all_point'):
    """VOC-style AP. method='all_point' (COCO/VOC2010+ continuous) or '11_point'."""
    if method == '11_point':
        ap = 0.0
        for t in np.linspace(0, 1, 11):
            p = prec[rec >= t]
            ap += (p.max() if p.size else 0.0) / 11.0
        return ap
    # all-point: envelope then integrate
    mrec = np.concatenate(([0.0], rec, [1.0]))
    mpre = np.concatenate(([0.0], prec, [0.0]))
    for i in range(len(mpre) - 2, -1, -1):
        mpre[i] = max(mpre[i], mpre[i + 1])
    idx = np.where(mrec[1:] != mrec[:-1])[0]
    return float(np.sum((mrec[idx + 1] - mrec[idx]) * mpre[idx + 1]))


def eval_class(det_annos, gt_annos, class_name, iou_thr, iou_fn, mode='3d',
               ap_method='all_point'):
    """AP for one class at one IoU threshold. Greedy, score-ordered, per-frame matching."""
    preds = []          # (score, frame_idx, box)
    n_gt = 0
    gt_by_frame = {}    # frame_idx -> (boxes(M,7), matched[M])
    for fi, (det, gt) in enumerate(zip(det_annos, gt_annos)):
        gmask = (gt['name'] == class_name)
        gboxes = gt['gt_boxes_lidar'][gmask] if gmask.any() else np.zeros((0, 7))
        gt_by_frame[fi] = [gboxes, np.zeros(len(gboxes), dtype=bool)]
        n_gt += len(gboxes)
        dmask = (det['name'] == class_name)
        if dmask.any():
            for s, b in zip(det['score'][dmask], det['boxes_lidar'][dmask]):
                preds.append((float(s), fi, b))

    if n_gt == 0:
        return float('nan'), 0, len(preds)
    if not preds:
        return 0.0, n_gt, 0

    preds.sort(key=lambda t: -t[0])
    tp = np.zeros(len(preds)); fp = np.zeros(len(preds))
    for k, (s, fi, box) in enumerate(preds):
        gboxes, matched = gt_by_frame[fi]
        if len(gboxes) == 0:
            fp[k] = 1; continue
        ious = iou_fn(box[None, :], gboxes, mode=mode)[0]  # (M,)
        j = int(np.argmax(ious))
        if ious[j] >= iou_thr and not matched[j]:
            tp[k] = 1; matched[j] = True
        else:
            fp[k] = 1

    tp_c = np.cumsum(tp); fp_c = np.cumsum(fp)
    rec = tp_c / n_gt
    prec = tp_c / np.maximum(tp_c + fp_c, 1e-12)
    return _ap_from_pr(rec, prec, ap_method), n_gt, len(preds)


def evaluate(det_annos, gt_annos, class_names, iou_thresholds,
             iou_backend='numpy', modes=('bev', '3d'), ap_method='all_point',
             gt_box_key='gt_boxes_lidar'):
    """Full evaluation. iou_thresholds: dict class->thr, or a single float for all.
    Returns nested dict: result[mode][class] = {'AP':..,'n_gt':..,'n_pred':..}, plus mAP."""
    iou_fn = IOU_BACKENDS[iou_backend]
    # normalize GT key
    gt_annos = [{'name': g['name'], 'gt_boxes_lidar': g[gt_box_key]} for g in gt_annos]
    if isinstance(iou_thresholds, (int, float)):
        iou_thresholds = {c: float(iou_thresholds) for c in class_names}

    result = {}
    for mode in modes:
        result[mode] = {}
        aps = []
        for c in class_names:
            ap, n_gt, n_pred = eval_class(det_annos, gt_annos, c, iou_thresholds[c],
                                          iou_fn, mode=mode, ap_method=ap_method)
            result[mode][c] = {'AP': ap, 'n_gt': n_gt, 'n_pred': n_pred,
                               'iou_thr': iou_thresholds[c]}
            if not np.isnan(ap):
                aps.append(ap)
        result[mode]['mAP'] = float(np.mean(aps)) if aps else float('nan')
    return result


def format_result(result):
    lines = []
    for mode, md in result.items():
        lines.append(f'=== {mode.upper()} AP ({"all-point"}) ===')
        for c, v in md.items():
            if c == 'mAP':
                continue
            ap = v['AP']
            aps = 'nan' if np.isnan(ap) else f'{ap*100:6.2f}'
            lines.append(f'  {c:12s} AP={aps}  (IoU>={v["iou_thr"]}, '
                         f'n_gt={v["n_gt"]}, n_pred={v["n_pred"]})')
        m = md['mAP']
        lines.append(f'  {"mAP":12s}    {"nan" if np.isnan(m) else f"{m*100:6.2f}"}')
    return '\n'.join(lines)


# --------------------------------------------------------------------------- #
# OpenPCDet evaluation() contract wrapper
# --------------------------------------------------------------------------- #

def evaluate_openpcdet(det_annos, gt_annos, class_names, iou_thresholds=None,
                       modes=('bev', '3d'), ap_method='all_point',
                       iou_backend='pcdet_gpu'):
    """Adapter for CustomDataset.evaluation(): returns (ap_result_str, ap_dict).

    Args:
        det_annos : result.pkl-style list -- {name, score, boxes_lidar, pred_labels, frame_id}
        gt_annos  : list of OpenPCDet 'annos' dicts (name + gt_boxes_lidar),
                    positionally aligned with det_annos (same contract as the kitti path).
        class_names : evaluated classes, native names (no KITTI remap needed -- we match
                      predictions to GT by name directly).
        iou_thresholds : dict class->IoU, a single float, or None for the default
                         (Vehicle/Car 0.7, else 0.5).
    Returns:
        (ap_result_str, ap_dict) -- the printable string and the scalar dict the
        OpenPCDet eval loop logs. ap_dict values may be nan for a class with no GT
        (absent class -> undefined AP, excluded from mAP), by design.
    """
    if iou_thresholds is None:
        _d = {'Vehicle': 0.7, 'Car': 0.7, 'Pedestrian': 0.5, 'Cyclist': 0.5}
        iou_thresholds = {c: _d.get(c, 0.5) for c in class_names}
    res = evaluate(det_annos, gt_annos, class_names, iou_thresholds,
                   iou_backend=iou_backend, modes=modes, ap_method=ap_method)
    ap_result_str = format_result(res)
    ap_dict = {}
    for mode in modes:
        for c in class_names:
            ap_dict[f'{c}_{mode}/AP'] = float(res[mode][c]['AP'])  # may be nan (no GT)
        ap_dict[f'mAP_{mode}'] = float(res[mode]['mAP'])
    return ap_result_str, ap_dict
