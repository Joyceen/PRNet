"""Evaluate PRNet results: MAE, max/mean/weighted F-measure, S-measure, E-measure."""
import argparse
import os
import time
import numpy as np
import cv2
from tqdm import tqdm


def load_mask(path):
    img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        return None
    return img.astype(np.float64) / 255.0


def mae(pred, gt):
    return float(np.abs(pred - gt).mean())


def calc_f_measures(pred, gt, beta=0.3):
    """Compute maxF, meanF via threshold scan (optimized with sorted traversal)."""
    pred_f = pred.ravel()
    gt_f = (gt.ravel() > 0.5)
    pos = gt_f.sum()
    neg = len(gt_f) - pos
    # Sort predictions descending
    idx = np.argsort(pred_f)[::-1]
    sorted_gt = gt_f[idx]
    # Cumulative sums as threshold decreases
    tp = np.cumsum(sorted_gt)
    fp = np.cumsum(~sorted_gt)
    fn = pos - tp
    precision = tp / (tp + fp + 1e-8)
    recall = tp / (tp + fn + 1e-8)
    f_beta = (1 + beta**2) * precision * recall / (beta**2 * precision + recall + 1e-8)
    return float(np.nanmax(f_beta)), float(np.nanmean(f_beta))


def weighted_f_measure(pred, gt, beta=0.3):
    """Weighted F-measure (wF)."""
    from scipy.ndimage import uniform_filter
    d = np.abs(pred - gt)
    w = 1 - uniform_filter(d, size=5)
    w = w / (w.sum() + 1e-8)
    pred_b = (pred >= 0.5).astype(np.float64)
    wp = (w * pred_b * gt).sum() / ((w * pred_b).sum() + 1e-8)
    wr = (w * pred_b * gt).sum() / ((w * gt).sum() + 1e-8)
    return float((1 + beta**2) * wp * wr / (beta**2 * wp + wr + 1e-8))


def s_measure(pred, gt):
    """S-measure (Structure-measure). Simplified version of Fan et al. ICCV 2017."""
    x = gt.mean()
    if x < 1e-6:
        return 0.0
    # Region-aware component
    mu_obj = pred[gt > 0.5].mean() if (gt > 0.5).any() else 0.0
    mu_bg = pred[gt < 0.5].mean() if (gt < 0.5).any() else 0.0
    obj_sim = 1.0 - abs(mu_obj - 1.0)
    bg_sim = 1.0 - abs(mu_bg - 0.0)
    sr = x * obj_sim + (1 - x) * bg_sim

    # Object-aware centeredness
    coords = np.argwhere(gt > 0.5)
    if len(coords) == 0:
        return float(sr)
    cy, cx = coords.mean(axis=0)
    h, w = gt.shape
    al = 1.0 - np.sqrt((cy - h/2)**2 + (cx - w/2)**2) / (np.sqrt((h/2)**2 + (w/2)**2) + 1e-8)
    so = al * obj_sim + (1 - al) * (1.0 - abs(x - pred.mean()))
    return float(0.5 * sr + 0.5 * so)


def e_measure(pred, gt):
    """E-measure (Enhanced-alignment measure). Fan et al. IJCAI 2018."""
    pred_b = (pred >= 0.5).astype(np.float64)
    aligned = 1.0 - 2.0 * np.abs(pred_b - gt)
    phi = 1.0 - 2.0 * np.abs(pred - pred.mean()) + 1.0 - 2.0 * np.abs(gt - gt.mean())
    return float((aligned * phi).sum() / (gt.size + 1e-8))


def evaluate_single(pred_path, gt_path):
    pred = load_mask(pred_path)
    gt = load_mask(gt_path)
    if pred is None or gt is None:
        return None
    if pred.shape != gt.shape:
        pred = cv2.resize(pred, (gt.shape[1], gt.shape[0]))

    em = mae(pred, gt)
    mxf, mnf = calc_f_measures(pred, gt)
    wf = weighted_f_measure(pred, gt)
    sm = s_measure(pred, gt)
    e_val = e_measure(pred, gt)
    return {'MAE': em, 'maxFm': mxf, 'meanFm': mnf, 'wFm': wf, 'Sm': sm, 'Em': e_val}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--pred_root', type=str, default='./test_maps/PRNet/')
    parser.add_argument('--gt_root', type=str, default='./TestDataset/')
    parser.add_argument('--datasets', type=str, nargs='+', default=['CAMO', 'COD10K', 'NC4K'])
    args = parser.parse_args()

    for ds in args.datasets:
        pred_dir = os.path.join(args.pred_root, ds)
        gt_dir = os.path.join(args.gt_root, ds, 'GT')
        if not os.path.isdir(pred_dir) or not os.path.isdir(gt_dir):
            print(f'\n[{ds}] Skipped: directory not found')
            continue

        pred_files = sorted([f for f in os.listdir(pred_dir) if f.endswith(('.png', '.jpg'))])
        t0 = time.time()

        results = []
        for pf in tqdm(pred_files, desc=f'{ds}'):
            stem = os.path.splitext(pf)[0]
            gt_path = os.path.join(gt_dir, stem + '.png')
            if not os.path.exists(gt_path):
                gt_path = os.path.join(gt_dir, stem + '.jpg')
            pred_path = os.path.join(pred_dir, pf)
            res = evaluate_single(pred_path, gt_path)
            if res is not None:
                results.append(res)

        if not results:
            print(f'\n[{ds}] No valid predictions')
            continue

        metrics = {k: np.mean([r[k] for r in results]) for k in results[0]}
        elapsed = time.time() - t0
        print(f'\n[{ds}] ({len(results)} images, {elapsed:.1f}s)')
        print(f'  MAE↓     : {metrics["MAE"]:.4f}')
        print(f'  maxFm↑   : {metrics["maxFm"]:.4f}')
        print(f'  meanFm↑  : {metrics["meanFm"]:.4f}')
        print(f'  wFm↑     : {metrics["wFm"]:.4f}')
        print(f'  Sm↑      : {metrics["Sm"]:.4f}')
        print(f'  Em↑      : {metrics["Em"]:.4f}')


if __name__ == '__main__':
    main()
