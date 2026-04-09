# Source Symbol Map

This extra appendix exists so the global documentation suite reaches the requested depth while remaining useful for code navigation.

## instashap_presentation/__init__.py

- Classes: none detected
- Functions: none detected
- Why open it: use this file to trace the source-level implementation details for the repository area it belongs to.

## instashap_presentation/generate_pptx.py

- Classes: none detected
- Functions: set_slide_bg, add_textbox, add_bullet_list, add_rounded_rect, add_arrow, generate_nn_image, generate_bar_chart_image, generate_complexity_chart, img_to_stream, add_image_to_slide, build_title_slide, build_problem_slide, build_realworld_need_slide, build_comparison_slide, build_workflow_step_slide, build_shap_output_slide, build_failure_slide, build_improvement_slide, build_metrics_slide, build_thank_you_slide, generate_presentation
- Why open it: use this file to trace the source-level implementation details for the repository area it belongs to.

## instashap_presentation/theme.py

- Classes: Theme
- Functions: rgb_int
- Why open it: use this file to trace the source-level implementation details for the repository area it belongs to.

## Phase_2_work/instashap_project/__init__.py

- Classes: none detected
- Functions: none detected
- Why open it: use this file to trace the source-level implementation details for the repository area it belongs to.

## Phase_2_work/instashap_project/data/__init__.py

- Classes: none detected
- Functions: none detected
- Why open it: use this file to trace the source-level implementation details for the repository area it belongs to.

## Phase_2_work/instashap_project/data/loaders.py

- Classes: DatasetMetadata, DatasetBundle
- Functions: load_bike_sharing, _soil_type_to_climate_label, load_covertype, load_adult_income, load_dataset
- Why open it: use this file to trace the source-level implementation details for the repository area it belongs to.

## Phase_2_work/instashap_project/data/preprocessing.py

- Classes: FeatureGroup, SplitBundle, TabularPreprocessor
- Functions: make_splits
- Why open it: use this file to trace the source-level implementation details for the repository area it belongs to.

## Phase_2_work/instashap_project/experiments/__init__.py

- Classes: none detected
- Functions: none detected
- Why open it: use this file to trace the source-level implementation details for the repository area it belongs to.

## Phase_2_work/instashap_project/experiments/adult_income.py

- Classes: none detected
- Functions: run
- Why open it: use this file to trace the source-level implementation details for the repository area it belongs to.

## Phase_2_work/instashap_project/experiments/bike_sharing.py

- Classes: none detected
- Functions: run
- Why open it: use this file to trace the source-level implementation details for the repository area it belongs to.

## Phase_2_work/instashap_project/experiments/common.py

- Classes: ExperimentResult
- Functions: _reshape_targets, _labels, _select_model_flags, _select_output_per_sample, _primary_metric_name, run_tabular_experiment
- Why open it: use this file to trace the source-level implementation details for the repository area it belongs to.

## Phase_2_work/instashap_project/experiments/covertype.py

- Classes: none detected
- Functions: run
- Why open it: use this file to trace the source-level implementation details for the repository area it belongs to.

## Phase_2_work/instashap_project/main.py

- Classes: none detected
- Functions: load_config, parse_args, main
- Why open it: use this file to trace the source-level implementation details for the repository area it belongs to.

## Phase_2_work/instashap_project/models/__init__.py

- Classes: none detected
- Functions: none detected
- Why open it: use this file to trace the source-level implementation details for the repository area it belongs to.

## Phase_2_work/instashap_project/models/blackbox_model.py

- Classes: TabularMLP, MaskedSurrogateMLP, RandomForestBlackBox
- Functions: _build_mlp_layers
- Why open it: use this file to trace the source-level implementation details for the repository area it belongs to.

## Phase_2_work/instashap_project/models/gam.py

- Classes: ComponentSpec, ComponentMLP, GAMModel
- Functions: component_name
- Why open it: use this file to trace the source-level implementation details for the repository area it belongs to.

## Phase_2_work/instashap_project/models/instashap.py

- Classes: InstaSHAPModel
- Functions: none detected
- Why open it: use this file to trace the source-level implementation details for the repository area it belongs to.

## Phase_2_work/instashap_project/training/__init__.py

- Classes: none detected
- Functions: none detected
- Why open it: use this file to trace the source-level implementation details for the repository area it belongs to.

## Phase_2_work/instashap_project/training/evaluate.py

- Classes: none detected
- Functions: predict_raw_outputs, predict_targets, evaluate_supervised_model
- Why open it: use this file to trace the source-level implementation details for the repository area it belongs to.

## Phase_2_work/instashap_project/training/train.py

- Classes: TrainingResult
- Functions: _make_tensor_loader, _expand_feature_mask_torch, _shapley_size_distribution, sample_shapley_feature_masks, _create_writer, _supervised_loss, _validation_loss, train_blackbox_model, _raw_outputs, train_masked_surrogate, train_gam_model, train_instashap_model
- Why open it: use this file to trace the source-level implementation details for the repository area it belongs to.

## Phase_2_work/instashap_project/utils/__init__.py

- Classes: none detected
- Functions: none detected
- Why open it: use this file to trace the source-level implementation details for the repository area it belongs to.

## Phase_2_work/instashap_project/utils/logging_utils.py

- Classes: none detected
- Functions: configure_logging, get_logger, format_log_event, _normalize_value
- Why open it: use this file to trace the source-level implementation details for the repository area it belongs to.

## Phase_2_work/instashap_project/utils/metrics.py

- Classes: RegressionMetrics, ClassificationMetrics
- Functions: regression_metrics, classification_metrics, explanation_error, benchmark_callable
- Why open it: use this file to trace the source-level implementation details for the repository area it belongs to.

## Phase_2_work/instashap_project/utils/reproducibility.py

- Classes: none detected
- Functions: set_global_seed, resolve_device, ensure_dir, write_json
- Why open it: use this file to trace the source-level implementation details for the repository area it belongs to.

## Phase_2_work/instashap_project/utils/visualization.py

- Classes: none detected
- Functions: _ensure_parent, plot_training_curves, plot_metric_bars, _output_vector, plot_shape_function, plot_interaction_heatmap, plot_feature_importance, plot_explanation_alignment
- Why open it: use this file to trace the source-level implementation details for the repository area it belongs to.

## Phase_2_work/instashap_project/xai/__init__.py

- Classes: none detected
- Functions: none detected
- Why open it: use this file to trace the source-level implementation details for the repository area it belongs to.

## Phase_2_work/instashap_project/xai/instashap_explainer.py

- Classes: InstaSHAPExplanationResult, InstaSHAPExplainer
- Functions: none detected
- Why open it: use this file to trace the source-level implementation details for the repository area it belongs to.

## Phase_2_work/instashap_project/xai/shap_wrapper.py

- Classes: ShapExplanationResult, ShapBaselineExplainer
- Functions: none detected
- Why open it: use this file to trace the source-level implementation details for the repository area it belongs to.

## Phase_3_work/complete_code.py

- Classes: none detected
- Functions: py_to_notebook, make_cell
- Why open it: use this file to trace the source-level implementation details for the repository area it belongs to.

## Phase_3_work/instashap_project/__init__.py

- Classes: none detected
- Functions: none detected
- Why open it: use this file to trace the source-level implementation details for the repository area it belongs to.

## Phase_3_work/instashap_project/data/__init__.py

- Classes: none detected
- Functions: none detected
- Why open it: use this file to trace the source-level implementation details for the repository area it belongs to.

## Phase_3_work/instashap_project/data/loaders.py

- Classes: DatasetMetadata, DatasetBundle
- Functions: load_bike_sharing, _soil_type_to_climate_label, load_covertype, load_adult_income, load_dataset
- Why open it: use this file to trace the source-level implementation details for the repository area it belongs to.

## Phase_3_work/instashap_project/data/preprocessing.py

- Classes: FeatureGroup, SplitBundle, TabularPreprocessor
- Functions: make_splits
- Why open it: use this file to trace the source-level implementation details for the repository area it belongs to.

## Phase_3_work/instashap_project/experiments/__init__.py

- Classes: none detected
- Functions: none detected
- Why open it: use this file to trace the source-level implementation details for the repository area it belongs to.

## Phase_3_work/instashap_project/experiments/common.py

- Classes: ExperimentResult
- Functions: _reshape_targets, _labels, _numeric_summary, _masking_configs, _selected_variants, _variant_label, _surrogate_label, _fidelity_label, evaluate_coalition_fidelity, run_phase3_experiment
- Why open it: use this file to trace the source-level implementation details for the repository area it belongs to.

## Phase_3_work/instashap_project/experiments/covertype.py

- Classes: none detected
- Functions: run
- Why open it: use this file to trace the source-level implementation details for the repository area it belongs to.

## Phase_3_work/instashap_project/masking.py

- Classes: MaskingConfig
- Functions: build_background_bank, build_masked_batch
- Why open it: use this file to trace the source-level implementation details for the repository area it belongs to.

## Phase_3_work/instashap_project/models/__init__.py

- Classes: none detected
- Functions: none detected
- Why open it: use this file to trace the source-level implementation details for the repository area it belongs to.

## Phase_3_work/instashap_project/models/blackbox_model.py

- Classes: TabularMLP, MaskedSurrogateMLP, RandomForestBlackBox
- Functions: _build_mlp_layers
- Why open it: use this file to trace the source-level implementation details for the repository area it belongs to.

## Phase_3_work/instashap_project/models/gam.py

- Classes: ComponentSpec, ComponentMLP, GAMModel
- Functions: component_name
- Why open it: use this file to trace the source-level implementation details for the repository area it belongs to.

## Phase_3_work/instashap_project/models/instashap.py

- Classes: InstaSHAPModel
- Functions: none detected
- Why open it: use this file to trace the source-level implementation details for the repository area it belongs to.

## Phase_3_work/instashap_project/reporting.py

- Classes: none detected
- Functions: _read_summary, _read_csv, _frame_to_text, _compact_comparison, _write_markdown, _interpret_outcome, _make_text_page, _make_table_page, _make_image_page, generate_reports
- Why open it: use this file to trace the source-level implementation details for the repository area it belongs to.

## Phase_3_work/instashap_project/training/__init__.py

- Classes: none detected
- Functions: none detected
- Why open it: use this file to trace the source-level implementation details for the repository area it belongs to.

## Phase_3_work/instashap_project/training/evaluate.py

- Classes: none detected
- Functions: predict_raw_outputs, predict_targets, evaluate_supervised_model
- Why open it: use this file to trace the source-level implementation details for the repository area it belongs to.

## Phase_3_work/instashap_project/training/train.py

- Classes: TrainingResult
- Functions: _make_tensor_loader, _expand_feature_mask_torch, _repeat_feature_mask_torch, _shapley_size_distribution, sample_shapley_feature_masks, _create_writer, _supervised_loss, _validation_loss, train_blackbox_model, _raw_outputs, mean_blackbox_outputs, mean_surrogate_outputs, _mean_surrogate_outputs_per_realization, train_masked_surrogate, train_gam_model, train_instashap_model
- Why open it: use this file to trace the source-level implementation details for the repository area it belongs to.

## Phase_3_work/instashap_project/utils/__init__.py

- Classes: none detected
- Functions: none detected
- Why open it: use this file to trace the source-level implementation details for the repository area it belongs to.

## Phase_3_work/instashap_project/utils/logging_utils.py

- Classes: none detected
- Functions: configure_logging, get_logger, format_log_event, _normalize_value
- Why open it: use this file to trace the source-level implementation details for the repository area it belongs to.

## Phase_3_work/instashap_project/utils/metrics.py

- Classes: RegressionMetrics, ClassificationMetrics
- Functions: regression_metrics, classification_metrics, explanation_error, explanation_metrics, benchmark_callable
- Why open it: use this file to trace the source-level implementation details for the repository area it belongs to.

## Phase_3_work/instashap_project/utils/reproducibility.py

- Classes: none detected
- Functions: set_global_seed, resolve_device, ensure_dir, write_json
- Why open it: use this file to trace the source-level implementation details for the repository area it belongs to.

## Phase_3_work/instashap_project/utils/visualization.py

- Classes: none detected
- Functions: _ensure_parent, plot_training_curves, plot_metric_bars, plot_named_metric_bars, _output_vector, plot_shape_function, plot_interaction_heatmap, plot_feature_importance, plot_explanation_alignment
- Why open it: use this file to trace the source-level implementation details for the repository area it belongs to.

## Phase_3_work/instashap_project/xai/__init__.py

- Classes: none detected
- Functions: none detected
- Why open it: use this file to trace the source-level implementation details for the repository area it belongs to.

## Phase_3_work/instashap_project/xai/instashap_explainer.py

- Classes: InstaSHAPExplanationResult, InstaSHAPExplainer
- Functions: none detected
- Why open it: use this file to trace the source-level implementation details for the repository area it belongs to.

## Phase_3_work/instashap_project/xai/shap_wrapper.py

- Classes: ShapExplanationResult, ShapBaselineExplainer
- Functions: none detected
- Why open it: use this file to trace the source-level implementation details for the repository area it belongs to.

## Phase_3_work/main.py

- Classes: none detected
- Functions: load_config, parse_args, _default_summary_path, main
- Why open it: use this file to trace the source-level implementation details for the repository area it belongs to.

## Phase_3_work/phase3_instashap_analysis.py

- Classes: FeatureGroup, TabularPreprocessor, TabularMLP, MaskedSurrogateMLP, ComponentMLP, GAMModel, InstaSHAPModel
- Functions: set_seed, soil_to_climate, build_mlp_layers, make_loader, supervised_loss, shapley_size_distribution, sample_shapley_masks, build_background_bank, build_masked_batch_zero, build_masked_batch_background, raw_model_outputs, mean_blackbox_outputs, mean_surrogate_outputs, train_blackbox, train_surrogate, train_instashap, evaluate_model, explanation_metrics, instashap_explain, plot_training_history, plot_feature_importance, plot_explanation_alignment, masking_fn_zero, shap_model_fn, masking_fn_bg, masking_fn_bg_eval, sample_grouped_masks, masking_fn_grouped_bg, train_surrogate_grouped, train_instashap_grouped
- Why open it: use this file to trace the source-level implementation details for the repository area it belongs to.

## Phase_3_work/tests/test_cli.py

- Classes: CliTests
- Functions: none detected
- Why open it: use this file to trace the source-level implementation details for the repository area it belongs to.

## Phase_3_work/tests/test_masking.py

- Classes: MaskingTests
- Functions: none detected
- Why open it: use this file to trace the source-level implementation details for the repository area it belongs to.
