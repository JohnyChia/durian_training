# Durian leaf model training

The project uses one self-contained dataset at `experiment/hybrid_dataset`.

- `original/` is the Gazebo-only view for Original.
- `hybrid/` is the image- and training-box-balanced real/Gazebo view shared by Baseline and Advanced.
- `evaluation/` contains separate real and Gazebo evaluation configurations.
- `audit.json` must pass before any training command is allowed to run.

The Git repository stores the dataset as a Git LFS archive. In a fresh Google Colab runtime:

```bash
git clone https://github.com/JohnyChia/durian_training.git
cd durian_training
git lfs pull
tar -xzf dataset/hybrid_dataset.tar.gz -C experiment
pip install -r experiment/training_model/requirements.txt
```

After extraction, `experiment/hybrid_dataset/audit.json` must exist. The rebuild script records how the dataset was produced, but rebuilding from source requires reacquiring the retired source datasets.

Three reproducible experiment entry points are provided. Run the checks first:

```bash
cd durian_training
python3 experiment/training_model/train_original.py --check-only
python3 experiment/training_model/train_baseline.py --check-only
python3 experiment/training_model/train_advanced.py --check-only
```

Train in order when CUDA is available:

```bash
python3 experiment/training_model/train_original.py --overwrite
python3 experiment/training_model/train_baseline.py --overwrite
python3 experiment/training_model/train_advanced.py --overwrite
```

Each command refuses to replace an existing final model. Use `--overwrite` only when an intentional retrain should replace it.

Final outputs:

- `models/durian_leaf/original.pt`: standard six-class YOLO26n checkpoint, trained on Gazebo only.
- `models/durian_leaf/baseline.pt`: standard six-class YOLO26n checkpoint, trained on balanced real/Gazebo data.
- `models/durian_leaf/advanced.pt`: standard six-class YOLO checkpoint with an EfficientNet-B0 backbone and YOLO26 detection head, trained on the same balanced data as Baseline.

All three final files can be loaded directly with `YOLO(path)`. Baseline and Advanced use the same audited hybrid data. Existing checkpoints were trained on the retired dataset, so all three models must be retrained before the new experiment.

Training artifacts and evaluation reports are stored under `experiment/outputs/training_model/`; the three final deployable files are stored under `models/durian_leaf/`.
