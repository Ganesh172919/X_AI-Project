# Installation and Setup

## Python Version

Recommended:

- Python 3.11+

The project has also been exercised in Python 3.13.

## Install Dependencies

```bash
pip install -r requirements.txt
```

## Main Dependencies

- `torch`
- `numpy`
- `pandas`
- `scikit-learn`
- `shap`
- `matplotlib`
- `seaborn`
- `PyYAML`
- `ucimlrepo`
- `tensorboard`
- `nbformat`

## Environment Setup

From the project directory:

```bash
cd instashap_project
python main.py --dataset all --model all --fast-dev-run
```

## Optional GPU

If PyTorch detects CUDA, the project can use it automatically when:

- `global.device: auto`

To force CPU, change `config.yaml`:

```yaml
global:
  device: cpu
```

