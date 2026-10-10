#!/usr/bin/env python3
"""Render traceable PNG metrics for completed V4.1 5-model experiments.

Only aggregate training CSV is used. DO NOT manufacture a confusion matrix,
true confidence-threshold F1 curve, or unseen evaluation results.
"""
from __future__ import annotations
import csv
import math
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]
MODELS = [
    ('model1_yolo26n_synthetic', 'Model 1 | YOLO26n | Synthetic', 'results.csv'),
    ('model2_yolo26n_hybrid', 'Model 2 | YOLO26n | Hybrid', 'results.csv'),
    ('model3_yolo26n_p2_synthetic', 'Model 3 | YOLO26n-P2 | Synthetic', 'results.csv'),
    ('model4_yolo26n_p2_hybrid', 'Model 4 | YOLO26n-P2 | Hybrid', 'results.csv'),
    ('model5_rfdetr_n_synthetic', 'Model 5 | RF-DETR Nano | Synthetic', 'metrics.csv'),
]

def num(x):
    try:
        f = float(x)
        return f if math.isfinite(f) else None
    except (ValueError, TypeError):
        return None

def load_csv(path):
    with path.open(encoding='utf-8-sig', newline='') as f:
        return list(csv.DictReader(f))

def point_map(rows, model_num):
    found = {}
    for row in rows:
        epoch = num(row.get('epoch'))
        if epoch is None:
            continue
        # RF-DETR CSV logger has zero-based epochs; training completion is +1.
        e = int(epoch) + (1 if model_num == 5 else 0)
        p = found.setdefault(e, {'epoch': e})
        if model_num != 5:
            fields = {
                'precision':'metrics/precision(B)', 'recall':'metrics/recall(B)',
                'map50':'metrics/mAP50(B)', 'map5095':'metrics/mAP50-95(B)',
                'trainbox':'train/box_loss', 'valbox':'val/box_loss',
                'traincls':'train/cls_loss', 'valcls':'val/cls_loss',
            }
        else:
            fields = {
                'precision':'val/precision', 'recall':'val/recall',
                'map50':'val/mAP_50', 'map5095':'val/mAP_50_95',
                'f1':'val/F1', 'trainloss':'train/loss',
                'ema_map5095':'val/ema_mAP_50_95',
            }
        for name, key in fields.items():
            value = num(row.get(key))
            if value is not None:
                p[name] = value
    if model_num != 5:
        for p in found.values():
            prec, rec = p.get('precision'), p.get('recall')
            if prec is not None and rec is not None:
                p['f1'] = 2*prec*rec/(prec+rec) if (prec+rec)>0 else 0.0
    return [found[k] for k in sorted(found)]

def figure(path, title, subtitle, points, series, ylimit=None, ylabel='Metric value'):
    fig, ax = plt.subplots(figsize=(12, 5.6))
    plotted = 0
    for field, name in series:
        xy = [(row['epoch'], row[field]) for row in points if field in row]
        if len(xy) < 2:
            continue
        ax.plot([x for x, _ in xy], [y for _, y in xy],
                label=name, linewidth=1.7)
        plotted += 1
    if not plotted:
        plt.close(fig)
        return False
    fig.suptitle(title, fontsize=15, weight='bold', y=0.98)
    ax.set_title(subtitle, fontsize=9, loc='left', pad=12)
    ax.set_xlabel('Completed epoch')
    ax.set_ylabel(ylabel)
    ax.grid(True, alpha=0.23)
    if ylimit:
        ax.set_ylim(*ylimit)
    ax.legend(loc='best')
    fig.tight_layout(rect=[0, 0, 1, 0.93])
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, format='png', dpi=170)
    plt.close(fig)
    return True

def main():
    count = 0
    for index, (slug, title, csv_name) in enumerate(MODELS, start=1):
        folder = ROOT / 'models_8run' / slug / 'results'
        src = folder / csv_name
        if not src.is_file():
            raise SystemExit(f'Expected authoritative CSV missing: {src}')
        points = point_map(load_csv(src), index)
        if len(points) < 2:
            raise SystemExit(f'Not enough metrics for {slug}')
        out = folder / 'png'
        meta = ('Validation metrics from original trainer CSV. Not sealed-test evaluation.'
                + (' Epoch 196 is last logged in Model 1 CSV.' if index == 1 else ''))
        count += figure(out/'performance.png', title+' - Precision / Recall / mAP', meta,
                        points, [('precision','Precision'),('recall','Recall'),
                                 ('map50','mAP50'),('map5095','mAP50-95')], (0,1))
        f1_subtitle = ('Recorded val/F1 from RF-DETR CSV.' if index == 5 else
                       'Derived epoch F1 = 2*P*R/(P+R); NOT native F1-confidence sweep.')
        count += figure(out/'f1.png', title+' - F1 by epoch', f1_subtitle,
                        points, [('f1','Recorded validation F1' if index == 5 else
                                       'Derived F1 (harmonic mean)')], (0,1), 'F1 score')
        loss = ([('trainloss','Train loss')] if index == 5 else
                [('trainbox','Train box'),('valbox','Validation box'),
                 ('traincls','Train class'),('valcls','Validation class')])
        count += figure(out/'loss.png', title+' - Training loss',
                        'Logged loss as available (not cross-model comparable).',
                        points, loss, None, 'Loss')
        if index == 5:
            count += figure(out/'ema_map.png', title+' - RF-DETR EMA mAP50-95',
                            'Recorded non-EMA and EMA validation mAP50-95.',
                            points, [('map5095','mAP50-95'),
                                     ('ema_map5095','EMA mAP50-95')], (0,1))
        print(f'{slug}: last logged completed epoch {points[-1]["epoch"]}')
    print(f'Generated {count} PNG charts from training CSVs')
    if count != 16:
        raise SystemExit(f'Unexpected chart count: {count}, expected 16')

if __name__ == '__main__':
    main()
