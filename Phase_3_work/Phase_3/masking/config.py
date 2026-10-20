"""Masking configuration dataclass."""

from __future__ import annotations
from dataclasses import dataclass


@dataclass
class MaskingConfig:
    """Configuration for feature masking strategies."""
    strategy: str = "zero"                    # "zero" | "empirical_background"
    background_bank_size: int = 256
    background_samples_train: int = 1
    background_samples_eval: int = 4
    use_curriculum: bool = False
    curriculum_warmup_frac: float = 0.25
    curriculum_standard_frac: float = 0.40
    ensemble_size: int = 1
    seed: int = 42

    @classmethod
    def from_config(cls, config: dict, seed: int = 42) -> "MaskingConfig":
        masking_cfg = config.get("masking", {})
        return cls(
            background_bank_size=int(masking_cfg.get("background_bank_size", 256)),
            background_samples_train=int(masking_cfg.get("background_samples_train", 1)),
            background_samples_eval=int(masking_cfg.get("background_samples_eval", 4)),
            curriculum_warmup_frac=float(masking_cfg.get("curriculum_warmup_frac", 0.25)),
            curriculum_standard_frac=float(masking_cfg.get("curriculum_standard_frac", 0.40)),
            ensemble_size=int(masking_cfg.get("ensemble_size", 3)),
            seed=seed,
        )

    def for_zero(self) -> "MaskingConfig":
        return MaskingConfig(strategy="zero", seed=self.seed)

    def for_background(self) -> "MaskingConfig":
        cfg = MaskingConfig(
            strategy="empirical_background",
            background_bank_size=self.background_bank_size,
            background_samples_train=self.background_samples_train,
            background_samples_eval=self.background_samples_eval,
            seed=self.seed,
        )
        return cfg

    def for_curriculum(self) -> "MaskingConfig":
        cfg = self.for_background()
        cfg.use_curriculum = True
        cfg.curriculum_warmup_frac = self.curriculum_warmup_frac
        cfg.curriculum_standard_frac = self.curriculum_standard_frac
        return cfg

    def for_ensemble(self) -> "MaskingConfig":
        cfg = self.for_curriculum()
        cfg.ensemble_size = self.ensemble_size
        return cfg
