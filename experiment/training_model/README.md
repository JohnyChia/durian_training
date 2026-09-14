# Durian leaf model training

Three reproducible experiment entry points are provided. Run the checks first:

```bash
cd /home/johny/durian_ws
python3 experiment/training_model/train_original.py --check-only
python3 experiment/training_model/train_baseline.py --check-only
python3 experiment/training_model/train_advanced.py --check-only
```

Train in order when CUDA is available:

```bash
python3 experiment/training_model/train_original.py
python3 experiment/training_model/train_baseline.py
python3 experiment/training_model/train_advanced.py
```

Each command refuses to replace an existing final model. Use `--overwrite` only when an intentional retrain should replace it.

Final outputs:

- `/home/johny/durian_ws/models/durian_leaf/original.pt`: standard six-class YOLO26n checkpoint, trained on Gazebo only.
- `/home/johny/durian_ws/models/durian_leaf/baseline.pt`: standard six-class YOLO26n checkpoint, trained on balanced real/Gazebo data.
- `/home/johny/durian_ws/models/durian_leaf/advanced.pt`: standard six-class YOLO checkpoint with an EfficientNet-B0 backbone and YOLO26 detection head, trained on the same balanced data as Baseline.

All three final files can be loaded directly with `YOLO(path)`. Baseline and Advanced use identical data and training settings so the backbone comparison is meaningful.

Training artifacts and evaluation reports are stored under `experiment/outputs/training_model/`; the three final deployable files are stored under `models/durian_leaf/`.
