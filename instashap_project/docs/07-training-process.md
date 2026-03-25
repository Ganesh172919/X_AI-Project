# Training Process

## Training Stages

The project trains models in a staged workflow:

1. Train the black-box baseline on labels
2. Train GAM-1 and optionally GAM-2 on labels
3. Train the masked surrogate on black-box masked outputs
4. Train InstaSHAP against masked surrogate outputs
5. Run SHAP and InstaSHAP explanations on evaluation samples

## Loss Functions

### Regression

- Black-box and GAM models: mean squared error
- Surrogate: mean squared error on masked outputs
- InstaSHAP: mean squared error on masked additive outputs

### Classification

- Black-box and GAM models: cross-entropy
- Explanations are compared on raw logits / log-probability style outputs
- Surrogate and InstaSHAP: mean squared error on surrogate output vectors

## Optimization

The main neural models use:

- `AdamW`
- configurable learning rate
- configurable weight decay
- early stopping via validation loss

## Hyperparameters

Hyperparameters are centrally defined in:

- `config.yaml`

Important groups:

- `training.blackbox`
- `training.gam`
- `training.surrogate`
- `training.instashap`

## Fast Development Mode

`--fast-dev-run` reduces runtime by:

- shrinking dataset size where configured
- capping epochs to a small number

This is useful for:

- smoke tests
- CI-like validation
- checking that artifact generation works

## Hardware Requirements

### Minimum

- CPU-only is enough for fast-dev runs
- 8+ GB RAM recommended

### Better for full experiments

- GPU optional but helpful
- More RAM is useful, especially for Covertype

