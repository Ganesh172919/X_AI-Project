# Experiment Reproduction

## Supported Reproduction Runs

### All datasets

```bash
python main.py --dataset all --model all
```

### Fast validation run

```bash
python main.py --dataset all --model all --fast-dev-run
```

### Individual datasets

```bash
python main.py --dataset bike --model all
python main.py --dataset covertype --model all
python main.py --dataset adult --model all
```

## What "Reproduction" Means Here

This repository reproduces:

- the experiment framing
- the masked additive methodology
- dataset-specific interaction analysis
- artifact generation and comparison tables

It does **not** claim exact parity with the original authors' undisclosed engineering details.

## Paper Comparison Tables

Each dataset writes a table comparing reproduced values to paper-reported values:

- Bike: black-box, GAM-1, GAM-2 normalized error
- Covertype: black-box, GAM-1, GAM-2 accuracy
- Adult: GAM-1 and InstaSHAP accuracy

## Reproduction Strategy

If you want closer paper numbers:

1. Disable `--fast-dev-run`
2. Increase epochs in `config.yaml`
3. Increase `max_rows` for Covertype
4. Tune hidden sizes, regularization, and learning rate
5. Inspect whether additional interactions should be included
