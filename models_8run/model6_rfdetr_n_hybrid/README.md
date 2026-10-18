# Model 6 — RF-DETR Nano · V4.2 Hybrid · Run06

**Training completed:** E200 according to original Johny E200 audit.

- Dataset: V4.2 Hybrid (1,425 train / 399 validation), six leaf classes.
- Training: RF-DETR Nano; AdamW; 200 epochs; batch 4, gradient accumulation 4; resolution 384.
- `weights/checkpoint_best_ema.pth`: best EMA weight **from the final Colab runtime**.
- `weights/checkpoint_best_total.pth`: best non-EMA/total checkpoint artifact from final runtime.
- `weights/last_ema.pth`: final EMA inference weight.
- RF-DETR uses `.pth`, **not YOLO `.pt`** weight format.
- `results/metrics_E196_E200.csv`: only the FINAL 5-epoch segment (CSV epoch values 195–199 are zero-based).
- `results/png/*_E196_E200.png`: genuine plots derived only from those saved metrics.
- The original E1–E195 metrics and historical global best cannot be independently
  verified from these final Drive files. Do not cite the last-runtime best as
  all-time E1–E200 best without recovering the historical metrics/weights.
- `results/JOHNY_E200_AUDIT.json`: original finish receipt and SHA256 manifest.

Source: Kent Master Drive Run06 final_e200; upload separately verified by SHA256.
