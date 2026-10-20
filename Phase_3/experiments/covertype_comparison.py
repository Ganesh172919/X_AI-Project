"""Covertype comparison experiment — 4-variant ablation study with 3 innovations.

Variants:
  instashap_zero       — Baseline (Phase 2 reproduction)
  instashap_bg         — Innovation 1: empirical-background masking
  instashap_curriculum — Innovation 1 + 2: + curriculum training
  instashap_full       — Innovation 1 + 2 + 3: + surrogate ensemble
"""

from __future__ import annotations

import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from data.loaders import load_dataset
from data.preprocessing import TabularPreprocessor, make_splits
from masking.config import MaskingConfig
from models.blackbox_model import MaskedSurrogateMLP, TabularMLP
from models.gam import GAMModel
from models.instashap import InstaSHAPModel
from training.train import (
    train_blackbox, train_gam, train_masked_surrogate,
    train_instashap_model, train_surrogate_ensemble,
)
from training.evaluate import predict_classes, predict_probabilities, predict_raw_outputs
from xai.instashap_explainer import explain_instashap
from xai.shap_wrapper import compute_shap_values
from utils import metrics as M
from utils.reproducibility import set_global_seed, resolve_device, ensure_dir, save_json
from utils.logging_utils import get_logger
from utils import visualization as viz

log = get_logger("experiment")


def _run_single_seed(
    seed: int,
    config: dict,
    masking_base: MaskingConfig,
    variant: str,
    output_root: Path,
    device: torch.device,
    fast_dev_run: bool = False,
) -> dict:
    """Execute experiment for a single seed."""
    set_global_seed(seed)
    log.info(f"═══ Seed {seed} ═══")

    ds_cfg = config["datasets"]["covertype"]
    max_rows = 4000 if fast_dev_run else ds_cfg.get("max_rows")

    # ── Load + split ──────────────────────────────────────────────────
    bundle = load_dataset("covertype", max_rows=max_rows, seed=seed)
    splits = make_splits(bundle, test_size=ds_cfg["test_size"], val_size=ds_cfg["val_size"], seed=seed)

    preprocessor = TabularPreprocessor(bundle.metadata)
    X_train = preprocessor.fit_transform(splits.X_train)
    X_val = preprocessor.transform(splits.X_val)
    X_test = preprocessor.transform(splits.X_test)
    y_train = splits.y_train.values
    y_val = splits.y_val.values
    y_test = splits.y_test.values

    n_classes = int(bundle.target.nunique())
    n_features = preprocessor.num_original_features
    feature_dim = preprocessor.input_dim
    feature_names = list(preprocessor.feature_order)
    interactions = [tuple(p) for p in ds_cfg.get("interaction_pairs", [])]

    bb_cfg = config["training"]["blackbox"]
    gam_cfg = config["training"]["gam"]
    surr_cfg = {**config["training"]["surrogate"], "output_dim": n_classes}
    isha_cfg = config["training"]["instashap"]

    if fast_dev_run:
        for c in [bb_cfg, gam_cfg, surr_cfg, isha_cfg]:
            c["epochs"] = min(c.get("epochs", 4), 4)
            c["patience"] = min(c.get("patience", 3), 3)

    results: dict = {"seed": seed, "variant": variant, "n_train": len(X_train),
                     "n_val": len(X_val), "n_test": len(X_test)}
    histories: dict = {}

    # ── Shared: Train black-box ───────────────────────────────────────
    log.info("Training black-box MLP...")
    t0 = time.time()
    blackbox = TabularMLP(feature_dim, n_classes, bb_cfg["hidden_dims"], bb_cfg["dropout"])
    blackbox, bb_hist = train_blackbox(blackbox, X_train, y_train, X_val, y_val, device, "classification", bb_cfg)
    results["blackbox_time"] = time.time() - t0
    histories["blackbox"] = bb_hist

    bb_preds = predict_classes(blackbox, X_test, device)
    bb_probs = predict_probabilities(blackbox, X_test, device)
    results["blackbox_accuracy"] = M.accuracy(y_test, bb_preds)
    results["blackbox_logloss"] = M.log_loss(y_test, bb_probs)
    log.info(f"  Black-box accuracy: {results['blackbox_accuracy']:.4f}")

    # ── Shared: Train GAMs ────────────────────────────────────────────
    log.info("Training GAM-1...")
    gam1 = GAMModel(preprocessor, n_classes, gam_cfg["hidden_dims"], interactions=[], dropout=gam_cfg["dropout"])
    gam1, gam1_hist = train_gam(gam1, X_train, y_train, X_val, y_val, device, "classification", gam_cfg)
    gam1_preds = predict_classes(gam1, X_test, device)
    gam1_probs = predict_probabilities(gam1, X_test, device)
    results["gam1_accuracy"] = M.accuracy(y_test, gam1_preds)
    results["gam1_logloss"] = M.log_loss(y_test, gam1_probs)
    histories["gam1"] = gam1_hist
    log.info(f"  GAM-1 accuracy: {results['gam1_accuracy']:.4f}")

    log.info("Training GAM-2...")
    gam2 = GAMModel(preprocessor, n_classes, gam_cfg["hidden_dims"], interactions=interactions, dropout=gam_cfg["dropout"])
    gam2, gam2_hist = train_gam(gam2, X_train, y_train, X_val, y_val, device, "classification", gam_cfg)
    gam2_preds = predict_classes(gam2, X_test, device)
    gam2_probs = predict_probabilities(gam2, X_test, device)
    results["gam2_accuracy"] = M.accuracy(y_test, gam2_preds)
    results["gam2_logloss"] = M.log_loss(y_test, gam2_probs)
    histories["gam2"] = gam2_hist
    log.info(f"  GAM-2 accuracy: {results['gam2_accuracy']:.4f}")

    # ── Build background bank ─────────────────────────────────────────
    rng = np.random.default_rng(seed)
    bg_bank = preprocessor.build_background_bank(X_train, masking_base.background_bank_size, rng)

    # ── Variant branches ──────────────────────────────────────────────
    variant_keys = []
    if variant in ("baseline", "compare"):
        variant_keys.append("instashap_zero")
    if variant in ("improved", "compare"):
        variant_keys.extend(["instashap_bg", "instashap_curriculum", "instashap_full"])

    surrogate_val_histories: dict[str, list[float]] = {}

    for vk in variant_keys:
        log.info(f"── Training variant: {vk} ──")
        t0 = time.time()

        if vk == "instashap_zero":
            mc = masking_base.for_zero()
            mc.seed = seed
            surr = MaskedSurrogateMLP(feature_dim, n_features, n_classes, surr_cfg["hidden_dims"], surr_cfg["dropout"])
            surr, surr_hist = train_masked_surrogate(
                surr, blackbox, X_train, X_val, preprocessor, device, surr_cfg, mc, None)
            surrogate_val_histories[vk] = [h["val_loss"] for h in surr_hist]
            histories[f"surrogate_{vk}"] = surr_hist
            active_surrogate = surr

        elif vk == "instashap_bg":
            mc = masking_base.for_background()
            mc.seed = seed
            surr = MaskedSurrogateMLP(feature_dim, n_features, n_classes, surr_cfg["hidden_dims"], surr_cfg["dropout"])
            surr, surr_hist = train_masked_surrogate(
                surr, blackbox, X_train, X_val, preprocessor, device, surr_cfg, mc, bg_bank)
            surrogate_val_histories[vk] = [h["val_loss"] for h in surr_hist]
            histories[f"surrogate_{vk}"] = surr_hist
            active_surrogate = surr

        elif vk == "instashap_curriculum":
            mc = masking_base.for_curriculum()
            mc.seed = seed
            surr = MaskedSurrogateMLP(feature_dim, n_features, n_classes, surr_cfg["hidden_dims"], surr_cfg["dropout"])
            surr, surr_hist = train_masked_surrogate(
                surr, blackbox, X_train, X_val, preprocessor, device, surr_cfg, mc, bg_bank)
            surrogate_val_histories[vk] = [h["val_loss"] for h in surr_hist]
            histories[f"surrogate_{vk}"] = surr_hist
            active_surrogate = surr

        elif vk == "instashap_full":
            mc = masking_base.for_ensemble()
            mc.seed = seed
            ensemble, ens_hists = train_surrogate_ensemble(
                blackbox, X_train, X_val, preprocessor, device, surr_cfg,
                mc, bg_bank, ensemble_size=masking_base.ensemble_size, base_seed=seed)
            avg_hist = []
            for ep in range(len(ens_hists[0])):
                avg_vl = np.mean([h[ep]["val_loss"] for h in ens_hists if ep < len(h)])
                avg_hist.append(avg_vl)
            surrogate_val_histories[vk] = avg_hist
            histories[f"surrogate_{vk}"] = ens_hists[0]  # store first for viz
            active_surrogate = ensemble

        results[f"{vk}_surrogate_time"] = time.time() - t0

        # Train InstaSHAP against this surrogate
        t0 = time.time()
        isha = InstaSHAPModel(preprocessor, n_classes, isha_cfg["hidden_dims"],
                              interactions=interactions, dropout=isha_cfg["dropout"])
        mc_isha = mc
        isha, isha_hist = train_instashap_model(
            isha, active_surrogate, X_train, X_val, preprocessor, device, isha_cfg,
            mc_isha, bg_bank if mc.strategy == "empirical_background" else None)
        results[f"{vk}_instashap_time"] = time.time() - t0
        histories[f"instashap_{vk}"] = isha_hist

        # Evaluate predictive performance
        isha_preds = predict_classes(isha, X_test, device)
        isha_probs = predict_probabilities(isha, X_test, device)
        results[f"{vk}_accuracy"] = M.accuracy(y_test, isha_preds)
        results[f"{vk}_logloss"] = M.log_loss(y_test, isha_probs)
        log.info(f"  {vk} accuracy: {results[f'{vk}_accuracy']:.4f}")

        # Convergence speed (Innovation 2)
        if vk in surrogate_val_histories:
            results[f"{vk}_convergence_epoch"] = M.convergence_epoch(surrogate_val_histories[vk])

        # Generate InstaSHAP explanations
        isha_expl = explain_instashap(isha, X_test[:ds_cfg.get("shap_sample_size", 24)],
                                      device, feature_names)
        results[f"_instashap_values_{vk}"] = isha_expl.grouped_values

    # ── Permutation SHAP (ground truth) ───────────────────────────────
    log.info("Computing Permutation SHAP (ground truth)...")
    shap_sample_size = ds_cfg.get("shap_sample_size", 24)
    bg_size = config["global"].get("shap_background_size", 64)
    max_evals = config["global"].get("shap_max_evals", 256)

    bg_idx = rng.choice(len(X_train), size=min(bg_size, len(X_train)), replace=False)
    X_background = X_train[bg_idx]
    X_explain = X_test[:shap_sample_size]

    def bb_fn(x: np.ndarray) -> np.ndarray:
        return predict_raw_outputs(blackbox, x, device)

    t0 = time.time()
    shap_result = compute_shap_values(bb_fn, X_background, X_explain, preprocessor, max_evals)
    results["shap_time"] = time.time() - t0

    # Select predicted class attributions for comparison
    bb_pred_classes = predict_classes(blackbox, X_explain, device)
    shap_for_pred_class = np.zeros((shap_sample_size, n_features), dtype=np.float32)
    for i in range(shap_sample_size):
        pc = int(bb_pred_classes[i])
        if pc < shap_result.grouped_values.shape[2]:
            shap_for_pred_class[i] = shap_result.grouped_values[i, :, pc]

    # ── Compare each variant's explanations vs SHAP ───────────────────
    for vk in variant_keys:
        key = f"_instashap_values_{vk}"
        if key not in results:
            continue
        isha_vals = results[key]
        isha_for_pred_class = np.zeros((shap_sample_size, n_features), dtype=np.float32)
        for i in range(min(shap_sample_size, len(isha_vals))):
            pc = int(bb_pred_classes[i])
            if pc < isha_vals.shape[2]:
                isha_for_pred_class[i] = isha_vals[i, :, pc]

        results[f"{vk}_explanation_mse"] = M.explanation_mse(shap_for_pred_class, isha_for_pred_class)
        results[f"{vk}_explanation_mae"] = M.explanation_mae(shap_for_pred_class, isha_for_pred_class)
        results[f"{vk}_spearman_rho"] = M.spearman_rank_correlation(shap_for_pred_class, isha_for_pred_class)
        log.info(f"  {vk} vs SHAP — MSE={results[f'{vk}_explanation_mse']:.4f}, "
                 f"MAE={results[f'{vk}_explanation_mae']:.4f}, ρ={results[f'{vk}_spearman_rho']:.4f}")

    # Clean up non-serializable values
    clean_results = {k: v for k, v in results.items() if not k.startswith("_")}

    # ── Save plots for this seed ──────────────────────────────────────
    seed_dir = output_root / "plots" / "covertype" / f"seed_{seed}"
    ensure_dir(seed_dir)

    # Training curves
    for hname, hdata in histories.items():
        if hdata:
            viz.plot_training_curves({hname: hdata}, f"{hname} (seed={seed})", seed_dir / f"{hname}_curves.png")

    # Convergence comparison
    if len(surrogate_val_histories) > 1:
        viz.plot_convergence_comparison(surrogate_val_histories, seed_dir / "convergence_comparison.png")

    # Explanation scatter
    variant_vals_for_plot: dict[str, np.ndarray] = {}
    for vk in variant_keys:
        key = f"_instashap_values_{vk}"
        if key in results:
            vals = results[key]
            pc_vals = np.zeros((shap_sample_size, n_features), dtype=np.float32)
            for i in range(min(shap_sample_size, len(vals))):
                pc = int(bb_pred_classes[i])
                if pc < vals.shape[2]:
                    pc_vals[i] = vals[i, :, pc]
            variant_vals_for_plot[vk] = pc_vals

    if variant_vals_for_plot:
        viz.plot_explanation_scatter_multi(
            shap_for_pred_class, variant_vals_for_plot,
            feature_names[:4], seed_dir / "explanation_scatter.png"
        )

    # Shape functions
    try:
        viz.plot_shape_functions(
            preprocessor, gam2, ["elevation", "slope"], X_train, device,
            seed_dir / "gam2_shape_functions.png", "GAM-2 Shape Functions")
    except Exception:
        pass

    return clean_results


def run_comparison(
    config: dict,
    variant: str = "compare",
    fast_dev_run: bool = False,
) -> dict:
    """Run full 3-seed comparison experiment."""
    device = resolve_device(config["global"].get("device", "auto"))
    output_root = Path(config["global"].get("output_root", "results"))
    ensure_dir(output_root / "tables")
    ensure_dir(output_root / "plots" / "covertype")
    ensure_dir(output_root / "artifacts" / "covertype")

    seeds = config["global"].get("seeds", [42, 123, 7])
    if fast_dev_run:
        seeds = seeds[:1]

    masking_base = MaskingConfig.from_config(config, seed=seeds[0])

    all_results: list[dict] = []
    for seed in seeds:
        result = _run_single_seed(seed, config, masking_base, variant, output_root, device, fast_dev_run)
        all_results.append(result)

    # ── Aggregate across seeds ────────────────────────────────────────
    aggregated: dict = {"n_seeds": len(seeds), "seeds": seeds, "variant": variant}

    numeric_keys = [k for k in all_results[0] if isinstance(all_results[0][k], (int, float))]
    for k in numeric_keys:
        vals = [r[k] for r in all_results if k in r]
        if vals:
            aggregated[f"{k}_mean"] = float(np.mean(vals))
            aggregated[f"{k}_std"] = float(np.std(vals))

    # ── Save tables ───────────────────────────────────────────────────
    # Model comparison table
    models_data = []
    for model_name in ["blackbox", "gam1", "gam2"]:
        row = {"model": model_name}
        for metric in ["accuracy", "logloss"]:
            key = f"{model_name}_{metric}"
            row[f"{metric}_mean"] = aggregated.get(f"{key}_mean", "")
            row[f"{metric}_std"] = aggregated.get(f"{key}_std", "")
        models_data.append(row)

    variant_keys = []
    if variant in ("baseline", "compare"):
        variant_keys.append("instashap_zero")
    if variant in ("improved", "compare"):
        variant_keys.extend(["instashap_bg", "instashap_curriculum", "instashap_full"])

    for vk in variant_keys:
        row = {"model": vk}
        for metric in ["accuracy", "logloss"]:
            key = f"{vk}_{metric}"
            row[f"{metric}_mean"] = aggregated.get(f"{key}_mean", "")
            row[f"{metric}_std"] = aggregated.get(f"{key}_std", "")
        models_data.append(row)

    pd.DataFrame(models_data).to_csv(output_root / "tables" / "covertype_model_metrics.csv", index=False)

    # Explanation comparison table
    expl_data = []
    for vk in variant_keys:
        row = {"variant": vk}
        for metric in ["explanation_mse", "explanation_mae", "spearman_rho", "convergence_epoch"]:
            key = f"{vk}_{metric}"
            row[f"{metric}_mean"] = aggregated.get(f"{key}_mean", "")
            row[f"{metric}_std"] = aggregated.get(f"{key}_std", "")
        expl_data.append(row)

    pd.DataFrame(expl_data).to_csv(output_root / "tables" / "covertype_explanation_comparison.csv", index=False)

    # ── Aggregated plots ──────────────────────────────────────────────
    plot_dir = output_root / "plots" / "covertype"

    # Innovation comparison bars — accuracy
    if len(variant_keys) > 1:
        acc_metrics = {}
        for vk in variant_keys:
            acc_metrics[vk] = {
                "accuracy": aggregated.get(f"{vk}_accuracy_mean", 0),
                "accuracy_std": aggregated.get(f"{vk}_accuracy_std", 0),
            }
        viz.plot_innovation_comparison_bars(
            acc_metrics, "accuracy", "InstaSHAP Accuracy — Innovation Ablation",
            "Test Accuracy", plot_dir / "innovation_accuracy_bars.png", higher_is_better=True)

        # Explanation MSE bars
        mse_metrics = {}
        for vk in variant_keys:
            mse_metrics[vk] = {
                "explanation_mse": aggregated.get(f"{vk}_explanation_mse_mean", 0),
                "explanation_mse_std": aggregated.get(f"{vk}_explanation_mse_std", 0),
            }
        viz.plot_innovation_comparison_bars(
            mse_metrics, "explanation_mse", "Explanation MSE vs SHAP — Innovation Ablation",
            "Explanation MSE (lower=better)", plot_dir / "innovation_mse_bars.png", higher_is_better=False)

        # Spearman rho bars
        rho_metrics = {}
        for vk in variant_keys:
            rho_metrics[vk] = {
                "spearman_rho": aggregated.get(f"{vk}_spearman_rho_mean", 0),
                "spearman_rho_std": aggregated.get(f"{vk}_spearman_rho_std", 0),
            }
        viz.plot_innovation_comparison_bars(
            rho_metrics, "spearman_rho", "Spearman ρ vs SHAP — Innovation Ablation",
            "Spearman Rank Correlation", plot_dir / "innovation_rho_bars.png", higher_is_better=True)

        # Model accuracy bar chart (all models)
        all_model_accs = {}
        for m in ["blackbox", "gam1", "gam2"] + variant_keys:
            all_model_accs[m] = aggregated.get(f"{m}_accuracy_mean", 0)
        viz.plot_metric_bar_chart(
            all_model_accs, "Accuracy", "All Models — Test Accuracy",
            plot_dir / "all_models_accuracy.png",
            paper_benchmarks={"blackbox": 0.804, "gam1": 0.724, "gam2": 0.822})

        # Radar chart
        radar_metrics = {}
        radar_keys = ["accuracy", "spearman_rho"]
        for vk in variant_keys:
            radar_metrics[vk] = {}
            for mk in radar_keys:
                key = f"{vk}_{mk}_mean"
                radar_metrics[vk][mk] = aggregated.get(key, 0)
            # Invert MSE for radar (higher = better)
            mse_val = aggregated.get(f"{vk}_explanation_mse_mean", 1)
            radar_metrics[vk]["fidelity"] = max(0, 1 - mse_val * 10)
        viz.plot_summary_radar(radar_metrics, ["accuracy", "spearman_rho", "fidelity"],
                               plot_dir / "innovation_radar.png")

    # ── Save JSON summary ─────────────────────────────────────────────
    save_json(aggregated, output_root / "artifacts" / "covertype" / "covertype_summary.json")
    save_json(all_results, output_root / "artifacts" / "covertype" / "per_seed_results.json")

    log.info("═══ Experiment Complete ═══")
    log.info(f"Results saved to: {output_root}")

    # Print summary table
    print("\n" + "=" * 80)
    print("PHASE 3 RESULTS SUMMARY — Covertype")
    print("=" * 80)
    print(f"{'Variant':<25} {'Accuracy':>10} {'Expl MSE':>12} {'Expl MAE':>12} {'Spearman ρ':>12}")
    print("-" * 80)
    for vk in variant_keys:
        acc = aggregated.get(f"{vk}_accuracy_mean", 0)
        mse = aggregated.get(f"{vk}_explanation_mse_mean", 0)
        mae = aggregated.get(f"{vk}_explanation_mae_mean", 0)
        rho = aggregated.get(f"{vk}_spearman_rho_mean", 0)
        print(f"{vk:<25} {acc:>10.4f} {mse:>12.6f} {mae:>12.6f} {rho:>12.4f}")
    print("=" * 80 + "\n")

    return aggregated
