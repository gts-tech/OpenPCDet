"""
Runner for custom_ap_eval -- OpenPCDet evaluation via the numba-free AP metric.

Reuses the compiled CUDA IoU op (pcdet.ops.iou3d_nms), so it runs on Blackwell
(sm_120) where the stock kitti_object_eval_python numba.cuda kernel does not build.
See custom_ap_eval.py for the full rationale.

Usage (from the repo root, with torch + pcdet importable, both files side by side):
    python run_custom_eval.py \\
        --result output/.../eval/.../result.pkl \\
        --infos  data/custom/custom_infos_val.pkl \\
        --backend pcdet_gpu --iou 0.7
Use --backend numpy for a GPU-free, numba-free cross-check.
"""
import argparse, pickle
import numpy as np
from pcdet.datasets.custom.custom_ap_eval import evaluate, format_result


def load(p):
    with open(p, 'rb') as f:
        return pickle.load(f)


def build_gt(infos, det_annos, gt_box_key='gt_boxes_lidar', frame_key_path=('point_cloud', 'lidar_idx')):
    """Align GT to det order by frame_id. Introspects structure first (no assumptions)."""
    print('=== GT introspection (verify before trusting) ===')
    i0 = infos[0]
    print('info[0] keys     :', list(i0.keys()))
    assert 'annos' in i0, "expected 'annos' in each info -- inspect and adjust"
    print('info[0][annos]   :', list(i0['annos'].keys()))
    # frame id of each info
    def fid(info):
        v = info
        for k in frame_key_path:
            v = v[k]
        return str(v)
    gt_by_fid = {}
    for info in infos:
        an = info['annos']
        assert gt_box_key in an, f"'{gt_box_key}' not in annos keys {list(an.keys())}"
        gt_by_fid[fid(info)] = {'name': np.asarray(an['name']),
                                'gt_boxes_lidar': np.asarray(an[gt_box_key], dtype=np.float32)}
    print(f'GT frames indexed: {len(gt_by_fid)}')
    # align to det order
    gt_annos = []
    miss = 0
    for d in det_annos:
        f = str(d['frame_id'])
        if f not in gt_by_fid:
            miss += 1
            gt_annos.append({'name': np.array([], dtype='<U10'),
                             'gt_boxes_lidar': np.zeros((0, 7), np.float32)})
        else:
            gt_annos.append(gt_by_fid[f])
    if miss:
        print(f'WARNING: {miss} det frame_id(s) had no matching GT info (check frame_key_path)')
    return gt_annos


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--result', required=True)
    ap.add_argument('--infos', required=True)
    ap.add_argument('--backend', default='pcdet_gpu', choices=['pcdet_gpu', 'numpy'])
    ap.add_argument('--iou', type=float, default=0.7,
                    help='single IoU thr for all classes; edit for per-class')
    ap.add_argument('--classes', nargs='+', default=['Vehicle', 'Pedestrian', 'Cyclist'])
    ap.add_argument('--gt-box-key', default='gt_boxes_lidar')
    args = ap.parse_args()

    det = load(args.result)
    infos = load(args.infos)
    print(f'loaded {len(det)} det frames, {len(infos)} gt infos')
    gt = build_gt(infos, det, gt_box_key=args.gt_box_key)

    # Per-class IoU thresholds: KITTI/Waymo convention is 0.7 Vehicle, 0.5 Ped/Cyc.
    # Single --iou applies to all; swap to the dict below for the conventional split.
    thr = {'Vehicle': args.iou, 'Pedestrian': args.iou, 'Cyclist': args.iou}
    # thr = {'Vehicle': 0.7, 'Pedestrian': 0.5, 'Cyclist': 0.5}

    res = evaluate(det, gt, args.classes, thr,
                   iou_backend=args.backend, modes=('bev', '3d'),
                   ap_method='all_point', gt_box_key='gt_boxes_lidar')
    print('\n' + format_result(res))


if __name__ == '__main__':
    main()
