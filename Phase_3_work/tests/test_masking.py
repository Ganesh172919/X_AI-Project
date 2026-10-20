from __future__ import annotations

import unittest

import numpy as np
import pandas as pd

from instashap_project.data.loaders import DatasetMetadata
from instashap_project.data.preprocessing import TabularPreprocessor
from instashap_project.masking import build_masked_batch


class MaskingTests(unittest.TestCase):
    def setUp(self) -> None:
        self.frame = pd.DataFrame(
            {
                "num": [1.0, 2.0, 3.0],
                "cat": pd.Categorical(["a", "b", "a"]),
            }
        )
        metadata = DatasetMetadata(
            name="toy",
            task="classification",
            target_name="target",
            numeric_features=["num"],
            categorical_features=["cat"],
        )
        self.preprocessor = TabularPreprocessor(metadata).fit(self.frame)
        self.transformed = self.preprocessor.transform(self.frame)
        self.feature_mask = np.asarray([[1.0, 0.0]], dtype=np.float32)
        self.background_bank = self.transformed[1:]

    def test_zero_mask_preserves_visible_features(self) -> None:
        masked = build_masked_batch(
            preprocessor=self.preprocessor,
            transformed_inputs=self.transformed[:1],
            feature_mask=self.feature_mask,
            strategy="zero_mask",
            rng=np.random.default_rng(42),
            background_bank=None,
            background_samples=1,
        )
        np.testing.assert_allclose(masked[0, 0, 0], self.transformed[0, 0])
        categorical_slice = self.preprocessor.group("cat").indices
        np.testing.assert_allclose(masked[0, 0, categorical_slice], 0.0)

    def test_empirical_background_keeps_one_hot_valid(self) -> None:
        masked = build_masked_batch(
            preprocessor=self.preprocessor,
            transformed_inputs=self.transformed[:1],
            feature_mask=self.feature_mask,
            strategy="empirical_background",
            rng=np.random.default_rng(42),
            background_bank=self.background_bank,
            background_samples=2,
        )
        categorical_slice = self.preprocessor.group("cat").indices
        for sample_index in range(masked.shape[1]):
            self.assertAlmostEqual(float(masked[0, sample_index, categorical_slice].sum()), 1.0, places=6)
        np.testing.assert_allclose(masked[0, :, 0], self.transformed[0, 0])


if __name__ == "__main__":
    unittest.main()
