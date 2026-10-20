import os
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch
from torch import nn

from data.preprocessing import FeatureGroup
from masking.config import MaskingConfig
from models.blackbox_model import MaskedSurrogateMLP
from training import train
from training.evaluate import predict_probabilities
from utils.metrics import convergence_epoch
from xai.shap_wrapper import compute_shap_values


@pytest.mark.parametrize("background_samples,curriculum", [(1, False), (3, False), (3, True)])
def test_background_training_pairs_match(background_samples, curriculum, monkeypatch):
    torch.manual_seed(0)
    groups = {name: FeatureGroup(name, "numeric", index, index + 1)
              for index, name in enumerate(["a", "b"])}
    preprocessor = SimpleNamespace(feature_order=["a", "b"], num_original_features=2,
                                   group=groups.__getitem__)
    blackbox = nn.Linear(2, 1, bias=False)
    with torch.no_grad():
        blackbox.weight.copy_(torch.tensor([[1.0, 10.0]]))

    class RecordingSurrogate(MaskedSurrogateMLP):
        def forward(self, inputs, feature_mask):
            self.last_inputs = inputs.detach()
            assert inputs.shape[0] == feature_mask.shape[0]
            return super().forward(inputs, feature_mask)

    surrogate = RecordingSurrogate(2, 2, 1, [4])
    checked_pairs = []

    class CheckingLoss(nn.MSELoss):
        def forward(self, predictions, targets):
            with torch.no_grad():
                torch.testing.assert_close(targets, blackbox(surrogate.last_inputs))
            checked_pairs.append(len(targets))
            return super().forward(predictions, targets)

    monkeypatch.setattr(train.nn, "MSELoss", CheckingLoss)
    values = np.arange(8, dtype=np.float32).reshape(4, 2)
    background = np.arange(20, 220, dtype=np.float32).reshape(100, 2)
    _, history = train.train_masked_surrogate(
        surrogate, blackbox, values, values[:2], preprocessor, torch.device("cpu"),
        {"epochs": 1, "batch_size": 4, "lr": 0.001, "weight_decay": 0.0,
         "patience": 2, "masks_per_sample": 2, "edge_mask_probability": 0.0},
        MaskingConfig(strategy="empirical_background", background_samples_train=background_samples,
                      use_curriculum=curriculum),
        background,
    )
    assert checked_pairs == [8 * background_samples, 2]
    assert np.isfinite(history[0]["train_loss"])
    assert np.isfinite(history[0]["val_loss"])


def test_binary_logit_probabilities():
    model = nn.Linear(1, 1, bias=False)
    with torch.no_grad():
        model.weight.fill_(1.0)
    logits = np.array([[-2.0], [0.0], [2.0]], dtype=np.float32)
    probabilities = predict_probabilities(model, logits, torch.device("cpu"))
    positive = 1.0 / (1.0 + np.exp(-logits[:, 0]))
    np.testing.assert_allclose(probabilities, np.column_stack([1.0 - positive, positive]), atol=1e-7)


def test_multiclass_probabilities_control():
    logits = np.array([[1.0, 2.0, 3.0]], dtype=np.float32)
    actual = predict_probabilities(nn.Identity(), logits, torch.device("cpu"))
    np.testing.assert_allclose(actual, torch.softmax(torch.from_numpy(logits), dim=1).numpy())


def test_permutation_shap_accepts_small_budget():
    groups = {name: FeatureGroup(name, "numeric", index, index + 1)
              for index, name in enumerate(["a", "b", "c"])}
    preprocessor = SimpleNamespace(feature_order=list(groups), feature_groups=groups,
                                   num_original_features=3)
    background = np.array([[0.0, 0.0, 0.0], [1.0, 2.0, 3.0]])
    explained = np.array([[3.0, 6.0, 9.0]])
    result = compute_shap_values(lambda rows: rows.sum(axis=1), background, explained,
                                 preprocessor, max_evals=1)
    assert result.grouped_values.shape == (1, 3, 1)
    np.testing.assert_allclose(result.grouped_values.sum() + result.base_values[0], 18.0)


def test_cli_log_level(tmp_path):
    script = Path(__file__).resolve().parents[1] / "main.py"
    result = subprocess.run([sys.executable, str(script), "--log-level", "ERROR", "--config", "missing.yaml"],
                            cwd=tmp_path, capture_output=True, text=True, encoding="utf-8", timeout=30,
                            env=dict(os.environ, PYTHONIOENCODING="utf-8"))
    assert result.returncode == 1
    assert "Config not found" in result.stdout
    assert "INFO" not in result.stdout


def test_convergence_threshold_control():
    assert convergence_epoch([8.0, 4.0, 2.1, 2.0], threshold_ratio=0.95) == 2
