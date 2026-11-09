# Graph Report - graph.html  (2026-11-09)

## Corpus Check
- cluster-only mode — file stats not available

## Summary
- 2378 nodes · 6679 edges · 93 communities (89 shown, 4 thin omitted)
- Extraction: 97% EXTRACTED · 3% INFERRED · 0% AMBIGUOUS · INFERRED: 214 edges (avg confidence: 0.85)
- Token cost: 0 input · 0 output

## Graph Freshness
- Built from commit: `c8bfbdac`
- Run `git rev-parse HEAD` and compare to check if the graph is stale.
- Run `graphify update .` after code changes (no API cost).

## Community Hubs (Navigation)
- integration.rs
- Ui
- tui/app.rs
- gguf.rs
- autotune.rs
- logits_io.rs
- tools.rs
- cli.rs
- logits_metrics.rs
- src/app.rs
- tui/mod.rs
- backup.rs
- hooked_qwen2.rs
- shift.rs
- server.rs
- score.rs
- loop_detector.rs
- generator.rs
- BenchError
- extract/mod.rs
- budget.rs
- deps
- tui_app
- command.rs
- weights.rs
- worker.rs
- AppDeps
- .args
- runner.rs
- schema.rs
- ExtractError
- dump.rs
- ppl_runner.rs
- engine/mod.rs
- extract_answer.rs
- model
- pathbuf
- mmap_file.rs
- ProgressEvent
- ui_deps
- report.rs
- validate.rs
- Event
- OperationKind
- engine/stats.rs
- executor.rs
- gram.rs
- steering_panel.rs
- result
- TensorStats
- Axis
- EngineError
- .event
- PatchError
- svd.rs
- method.rs
- bench/mod.rs
- tui/ui.rs
- instant
- extract/stats.rs
- extract/prompts.rs
- DType
- model_size.rs
- Site
- invalid
- path
- dispatch
- LinalgError
- dataset.rs
- core_apply
- jacobi.rs
- Value
- ppl_parse.rs
- leace.rs
- super
- Pooling
- ExtractEvent
- Template
- BroadcastAxis
- matrix_panel.rs
- CommandOutput
- Theme
- eval/prompts.rs
- .send
- LayerSelection
- config.rs
- HardwareError
- Method
- deps_purity.rs
- BrokenPipe
- LineRead
- weight-surgeon

## God Nodes (most connected - your core abstractions)
1. `Value` - 62 edges
2. `Ui` - 59 edges
3. `DType` - 57 edges
4. `deps()` - 57 edges
5. `s()` - 50 edges
6. `App` - 48 edges
7. `AppDeps` - 46 edges
8. `ExtractError` - 40 edges
9. `BenchError` - 35 edges
10. `HardwareProbe` - 35 edges

## Surprising Connections (you probably didn't know these)
- `bf16_model()` --calls--> `encode()`  [INFERRED]
  tests/integration.rs → src/engine/test_support.rs
- `autotune_patch_is_valid_deterministic_and_guarded()` --calls--> `parse_patch()`  [INFERRED]
  tests/integration.rs → src/patch/schema.rs
- `cancellation_mid_apply_rolls_back()` --calls--> `core_apply()`  [INFERRED]
  tests/integration.rs → src/app.rs
- `cancellation_mid_apply_rolls_back()` --calls--> `parse_patch()`  [INFERRED]
  tests/integration.rs → src/patch/schema.rs
- `ppl_deps()` --references--> `ColorMode`  [EXTRACTED]
  tests/integration.rs → src/ui.rs

## Import Cycles
- 1-file cycle: `src/mcp/tools.rs -> src/mcp/tools.rs`
- 1-file cycle: `src/engine/index.rs -> src/engine/index.rs`
- 2-file cycle: `src/tui/mod.rs -> src/tui/terminal.rs -> src/tui/mod.rs`
- 2-file cycle: `src/tui/app.rs -> src/tui/mod.rs -> src/tui/app.rs`
- 2-file cycle: `src/tui/mod.rs -> src/tui/worker.rs -> src/tui/mod.rs`
- 2-file cycle: `src/app.rs -> src/mcp/tools.rs -> src/app.rs`
- 2-file cycle: `src/mcp/server.rs -> src/mcp/tools.rs -> src/mcp/server.rs`
- 2-file cycle: `src/eval/loop_detector.rs -> src/eval/mod.rs -> src/eval/loop_detector.rs`
- 2-file cycle: `src/eval/mod.rs -> src/eval/runner.rs -> src/eval/mod.rs`
- 2-file cycle: `src/bench/logits_io.rs -> src/bench/logits_metrics.rs -> src/bench/logits_io.rs`
- 2-file cycle: `src/patch/backup.rs -> src/patch/executor.rs -> src/patch/backup.rs`
- 2-file cycle: `src/cli.rs -> src/patch/schema.rs -> src/cli.rs`
- 2-file cycle: `src/engine/shift.rs -> src/engine/test_support.rs -> src/engine/shift.rs`
- 3-file cycle: `src/tui/app.rs -> src/tui/mod.rs -> src/tui/terminal.rs -> src/tui/app.rs`
- 3-file cycle: `src/tui/mod.rs -> src/tui/terminal.rs -> src/tui/worker.rs -> src/tui/mod.rs`
- 3-file cycle: `src/tui/app.rs -> src/tui/worker.rs -> src/tui/mod.rs -> src/tui/app.rs`
- 3-file cycle: `src/tui/app.rs -> src/tui/mod.rs -> src/tui/ui.rs -> src/tui/app.rs`
- 3-file cycle: `src/app.rs -> src/mcp/tools.rs -> src/mcp/server.rs -> src/app.rs`
- 3-file cycle: `src/eval/loop_detector.rs -> src/eval/mod.rs -> src/eval/runner.rs -> src/eval/loop_detector.rs`
- 3-file cycle: `src/eval/gguf.rs -> src/eval/runner.rs -> src/eval/mod.rs -> src/eval/gguf.rs`

## Communities (93 total, 4 thin omitted)

### Community 0 - "integration.rs"
Cohesion: 0.04
Nodes (36): bench_dump_before_and_after_an_apply_on_a_copy(), bench_dump_refuses_before_computing_and_leaves_nothing(), bench_dump_writes_a_complete_file_and_the_same_model_compares_to_zero(), bench_logits_cli_json_thresholds_and_exit_codes(), copy_into(), DOWN, dump_args(), error_code() (+28 more)

### Community 1 - "Ui"
Cohesion: 0.06
Nodes (50): ALERT_RED, accent(), all_modes(), badge(), BADGE_WIDTH, banner(), banner_is_the_exact_logo_and_aligned(), BLUE (+42 more)

### Community 2 - "tui/app.rs"
Cohesion: 0.08
Nodes (34): accepted(), App, app(), BLINK_TICKS, Companion, CompanionState, Invalid, Loaded (+26 more)

### Community 3 - "gguf.rs"
Cohesion: 0.08
Nodes (43): verdict_label(), run_one(), CommandRunner, last_lines(), StreamControl, SystemCommandRunner, ALLOWED_QUANTS, convert_then_quantize_in_order_with_exact_args() (+35 more)

### Community 4 - "autotune.rs"
Cohesion: 0.08
Nodes (45): apply(), auto_grid(), AUTO_GRID_POINTS, InvalidArgument, AutotuneEvent, Forward, Planned, Solve (+37 more)

### Community 5 - "logits_io.rs"
Cohesion: 0.09
Nodes (44): check_pair(), check_ram(), crc32(), Data, Mapped, Owned, dtypes_mixed_and_three_dimensional_dumps(), DumpFormat (+36 more)

### Community 6 - "tools.rs"
Cohesion: 0.06
Nodes (37): McpArgs, SchemaArgs, SchemaKind, Patch, Tools, run_stdio(), write_schema(), SUPPORTED_PROTOCOL_VERSIONS (+29 more)

### Community 7 - "cli.rs"
Cohesion: 0.06
Nodes (35): command_name(), real_generator(), apply_and_eval_defaults(), ApplyArgs, AutotuneArgs, bench_subcommands_defaults_requirements_and_conflicts(), BenchArgs, BenchCommand (+27 more)

### Community 8 - "logits_metrics.rs"
Cohesion: 0.09
Nodes (40): shared_offsets(), Accumulator, analytic_kl_and_identity(), argmax(), BadSide, After, Before, BLOCK_POSITIONS (+32 more)

### Community 9 - "src/app.rs"
Cohesion: 0.06
Nodes (29): autotune_text(), BAR_WIDTH, BENCH_FAILED, budget_text(), CRITICAL, Envelope, envelope_of(), ErrorBody (+21 more)

### Community 10 - "tui/mod.rs"
Cohesion: 0.10
Nodes (27): TuiArgs, EventSource, EXIT_NOT_INTERACTIVE, interactive_session(), load(), parse_size(), report(), run_cli() (+19 more)

### Community 11 - "backup.rs"
Cohesion: 0.10
Nodes (20): backup_set_over_three_shards(), BackupGuard, BackupSet, copy_synced(), corrupted_or_missing_backup_never_touches_the_original(), create_makes_identical_copy_next_to_original(), create_unique(), drop_without_commit_restores() (+12 more)

### Community 12 - "hooked_qwen2.rs"
Cohesion: 0.14
Nodes (15): Attention, cfg(), DecoderLayer, early_exit_is_bit_identical_to_the_full_forward_layer(), forward_logits_matches_the_reference_at_every_position(), full_forward_matches_candle_reference_logits(), HookedQwen2, ids() (+7 more)

### Community 13 - "shift.rs"
Cohesion: 0.11
Nodes (32): decode(), encode(), every_check_runs_before_the_first_write(), full_strength_erases_the_read_direction_and_is_idempotent(), identical_with_any_thread_count(), matches_an_independent_reference_bit_for_bit_in_every_dtype(), merge(), project() (+24 more)

### Community 14 - "server.rs"
Cohesion: 0.10
Nodes (24): classify(), Incoming, Invalid, Notification, Request, Response, INTERNAL_ERROR, INVALID_PARAMS (+16 more)

### Community 15 - "score.rs"
Cohesion: 0.11
Nodes (33): ItemResult, PairedComparison, answers_match(), broken_generator_stops_and_inputs_are_checked(), close(), EvalRequest, evaluate(), EXCERPT_CHARS (+25 more)

### Community 16 - "loop_detector.rs"
Cohesion: 0.11
Nodes (27): char_periodicity_ignores_layout_separators(), feed(), HEALTHY_FR, healthy_french_text_is_not_a_loop_whatever_the_chunking(), identical_lines(), LoopDetector, LoopDetectorConfig, LoopReason (+19 more)

### Community 17 - "generator.rs"
Cohesion: 0.09
Nodes (21): break_kills_an_endless_child(), CHANNEL_CHUNKS, Decoding, executable_names(), failing_child_reports_stderr_tail(), find_binary(), GenerationOutcome, llama_cli_arguments() (+13 more)

### Community 18 - "BenchError"
Cohesion: 0.07
Nodes (34): BenchError, BinaryNotFound, Cancelled, Compute, Conversion, Generation, Input, InsufficientResources (+26 more)

### Community 19 - "extract/mod.rs"
Cohesion: 0.14
Nodes (29): MethodOptions, COMPANION_SCHEMA, ExtractObserver, ExtractReport, ExtractRequest, file_metadata(), method_metadata(), MIN_RECOMMENDED_PROMPTS (+21 more)

### Community 20 - "budget.rs"
Cohesion: 0.12
Nodes (29): BudgetPolicy, can_load_model(), can_load_model_with(), edge_cases_zero_and_u64_max_do_not_overflow(), eval(), evaluate_budget(), evaluate_with_size(), fits_in_vram_picks_gpu_with_most_free_memory() (+21 more)

### Community 21 - "deps"
Cohesion: 0.14
Nodes (36): apply_shifts_exactly_and_keeps_a_backup(), backups_in(), bench_eval_scores_pairs_and_gates_on_min_accuracy(), bench_ppl_baselines_rounded_delta_and_exact_arguments(), bench_ppl_failures_are_typed_and_the_temporary_gguf_is_removed(), bf16_model(), check_verdict_drives_the_exit_code(), closed_pipe_never_panics_and_keeps_the_exit_code() (+28 more)

### Community 22 - "tui_app"
Cohesion: 0.09
Nodes (24): companion_json(), f32_le(), find_cell(), IdleEvents, PanickingEvents, request(), SlowProvider, spy() (+16 more)

### Community 23 - "command.rs"
Cohesion: 0.10
Nodes (18): CAPTURE_TAIL_BYTES, CHILD_MODE, child_run(), ChildGuard, default_run_streaming_replays_the_output(), drain_tail(), LineSplitter, missing_program_is_binary_not_found() (+10 more)

### Community 24 - "weights.rs"
Cohesion: 0.12
Nodes (17): Qwen2Config, candle_dtype(), cfg(), check_tensors(), Fake, fake(), load_tensors(), loading_reads_only_required_names_and_can_be_cancelled() (+9 more)

### Community 25 - "worker.rs"
Cohesion: 0.12
Nodes (11): Job, Lru, lru_evicts_the_least_recent(), Lru<K, V>, ModelStatsProvider, StatsKey, StatsProvider, StatsReply (+3 more)

### Community 26 - "AppDeps"
Cohesion: 0.20
Nodes (27): AppDeps, autotune_cli(), autotune_failure(), bench_cli(), bench_failure(), bench_outcome(), core_autotune(), core_bench_dump() (+19 more)

### Community 27 - ".args"
Cohesion: 0.13
Nodes (29): autotune_fixture(), autotune_objectives_and_per_prompt_targets(), autotune_patch_is_valid_deterministic_and_guarded(), autotune_response_matches_patching_each_layer_on_disk(), AutotuneFixture, bench_json_on_the_binary_is_one_clean_document_even_with_forced_colors(), binary(), binary_smoke() (+21 more)

### Community 28 - "runner.rs"
Cohesion: 0.15
Nodes (25): Behavior, Endless, Fail, Say, check_expectations(), empty_prompt_list_is_refused(), endless_wait_stops_early_with_loop_detected(), EvalConfig (+17 more)

### Community 29 - "schema.rs"
Cohesion: 0.09
Nodes (17): MlpProj, Schema, InspectToolArgs, proj_name(), FULL, LAYER_SPEC_PATTERN, LayerRange, LayersField (+9 more)

### Community 30 - "ExtractError"
Cohesion: 0.08
Nodes (26): ExtractError, Cancelled, Compute, Config, DegenerateDirection, InsufficientResources, InvalidArgument, Io (+18 more)

### Community 31 - "dump.rs"
Cohesion: 0.13
Nodes (23): check_output(), DumpRequest, header(), header_is_aligned_sorted_and_parseable(), io_error(), OutDtype, F16, F32 (+15 more)

### Community 32 - "ppl_runner.rs"
Cohesion: 0.12
Nodes (24): arguments_are_structured_and_absolute(), build_args(), delta(), delta_is_rounded_before_the_strict_comparison(), est(), llama_args_whitelist(), locate_perplexity(), NUMERIC_FLAGS (+16 more)

### Community 33 - "engine/mod.rs"
Cohesion: 0.13
Nodes (20): ModelIndex, TensorInfo, load_direction_vector(), ReadOnlyModel, TensorView, find_mlp_tensors(), index(), LayerSpec (+12 more)

### Community 34 - "extract_answer.rs"
Cohesion: 0.13
Nodes (26): balanced(), bold(), bold_re(), cached(), END_MARKERS, end_markers_are_not_part_of_the_answer(), extract(), Extracted (+18 more)

### Community 35 - "model"
Cohesion: 0.26
Nodes (29): McpConfig, call(), cancellation_mid_apply_rolls_back(), envelope(), init(), mcp_apply_is_exact_and_always_backed_up(), mcp_bench_progress_and_cancellation_kill_the_process(), mcp_bench_runs_every_mode_and_guards_its_arguments() (+21 more)

### Community 36 - "pathbuf"
Cohesion: 0.11
Nodes (14): Result, Result, Result, Result, inside_paths_resolve_and_outside_paths_are_refused(), resolve_lenient(), same_component(), Sandbox (+6 more)

### Community 37 - "mmap_file.rs"
Cohesion: 0.12
Nodes (17): parse_index(), bf16_weights(), bit_exact_bf16_shift_through_file_leaves_everything_else_intact(), corrupt_files_give_distinct_errors(), direction_vector_from_safetensors_and_json(), DOWN, four_tensor_file(), GATE (+9 more)

### Community 38 - "ProgressEvent"
Cohesion: 0.08
Nodes (23): NullSink, ProgressEvent, ApplyStart, AutotuneForward, AutotuneSolve, Backup, BenchDumpPrompt, BenchEvalQuestion (+15 more)

### Community 39 - "ui_deps"
Cohesion: 0.11
Nodes (26): strip_ansi(), apply_badges_follow_the_real_steps(), apply_case(), autotune_human_output_and_strict_json_on_the_binary(), bench_human_output_golden_in_truecolor(), Calls, check_golden_in_truecolor(), closed_pipe_in_color_mode_never_panics() (+18 more)

### Community 40 - "report.rs"
Cohesion: 0.16
Nodes (22): BenchDumpReport, BenchEvalReport, BenchLogitsReport, BenchRun, dump_text(), DumpInfo, DumpTiming, eval_text() (+14 more)

### Community 41 - "validate.rs"
Cohesion: 0.20
Nodes (23): ApplyReport, OperationReport, TensorReport, full_patch_deserializes_with_defaults(), load_patch(), Operation, parse_patch(), PatchFile (+15 more)

### Community 42 - "Event"
Cohesion: 0.10
Nodes (8): arg_after(), main(), Event, Message, ParseError, TooLong, ScriptedEventSource, TerminalEvents

### Community 43 - "OperationKind"
Cohesion: 0.13
Nodes (10): backend_error(), EngineBackend, ObservedBackend<'a, B>, ObservedBackend<'_, B>, OpStats, shift_op(), OperationKind, Add (+2 more)

### Community 44 - "engine/stats.rs"
Cohesion: 0.17
Nodes (21): block_size_does_not_change_the_result(), cancellation_returns_no_partial_result(), Decode, decoder(), DEFAULT_BLOCK, degenerate_tensors(), f32_bytes(), Histogram (+13 more)

### Community 45 - "executor.rs"
Cohesion: 0.24
Nodes (16): abort(), apply_patch(), dry_run_never_shifts(), failed_rollback_is_critical(), failure_on_second_op_restores_the_original(), failure_without_backup_is_reported_as_left_modified(), model(), nan_or_inf_triggers_rollback() (+8 more)

### Community 46 - "gram.rs"
Cohesion: 0.21
Nodes (14): gram_rank_tolerance(), large_d_never_forms_a_d_by_d_matrix(), matches_a_direct_eigen_decomposition_and_power_iteration(), planted(), rank_deficiency_orthogonality_and_thread_invariance(), thin_svd(), ThinSvd, wide_and_tall_recover_planted_singular_values() (+6 more)

### Community 47 - "steering_panel.rs"
Cohesion: 0.20
Nodes (11): VectorView, render_gauge(), component_rows(), components_around_zero(), D_PRIME_SCALE, kv(), render_components(), render_separation() (+3 more)

### Community 48 - "result"
Cohesion: 0.11
Nodes (9): PLENTY, Ram, amd_gpus_in(), amd_sysfs_parsing_keeps_only_readable_amd_cards(), card(), MemoryInfo, system_probe_never_fails_without_gpu(), SystemProbe (+1 more)

### Community 49 - "TensorStats"
Cohesion: 0.13
Nodes (14): InspectedTensor, TensorStats, Msg, Key, Resize, StatsReady, Tick, alert_text() (+6 more)

### Community 50 - "Axis"
Cohesion: 0.11
Nodes (11): BroadcastAxis, MockBackend, OpStats, PatchBackend, Axis, Cols, Rows, ValidatedOperation (+3 more)

### Community 51 - "EngineError"
Cohesion: 0.10
Nodes (18): EngineError, ByteLengthMismatch, DataSizeMismatch, DirectionLengthMismatch, FileTooSmall, InvalidAlpha, InvalidDirectionSource, InvalidHeaderJson (+10 more)

### Community 52 - ".event"
Cohesion: 0.15
Nodes (7): CATCHING, CliSink, CliSink<'a, 'io>, ExtractView, Interruptible, Io<'a>, SinkObserver

### Community 53 - "PatchError"
Cohesion: 0.12
Nodes (16): CRITICAL, EXECUTION_FAILED, i32, INVALID_PATCH, OK, PatchError, Backend, Backup (+8 more)

### Community 54 - "svd.rs"
Cohesion: 0.23
Nodes (17): Lcg, centering_loses_a_constant_delta_axis(), dot(), Leading, leading(), mean_rows(), MIN_RECOMMENDED_PAIRS, noisy_isotropic_deltas_warn_about_instability() (+9 more)

### Community 55 - "method.rs"
Cohesion: 0.17
Nodes (16): LeaceDiagnostics, compute(), DEFAULT_LEACE_RCOND, DEFAULT_LEACE_SHRINKAGE, holdout_mask(), holdout_mask_is_even_and_deterministic(), MethodDiagnostics, Leace (+8 more)

### Community 56 - "bench/mod.rs"
Cohesion: 0.12
Nodes (12): DEFAULT_CTX, DEFAULT_DUMP_MAX_TOKENS, DEFAULT_LIMIT, DEFAULT_MAX_NEW_TOKENS, DEFAULT_MAX_TOTAL_POSITIONS, DEFAULT_TOP_K, DEFAULT_WARN_DELTA, DEFAULT_WORST (+4 more)

### Community 57 - "tui/ui.rs"
Cohesion: 0.19
Nodes (9): MIN_HEIGHT, MIN_WIDTH, Tab, Inspector, Steering, help_overlay(), status_bar(), view() (+1 more)

### Community 58 - "instant"
Cohesion: 0.19
Nodes (11): cosine(), Data, dot(), generate(), halves(), Lcg, main(), Method (+3 more)

### Community 59 - "extract/stats.rs"
Cohesion: 0.26
Nodes (15): best_threshold(), check_sets(), direction_matches_an_independent_reference(), dot(), finish(), mean(), MIN_DIFF_NORM, Separation (+7 more)

### Community 60 - "extract/prompts.rs"
Cohesion: 0.19
Nodes (10): Tokenizer, chatml_keeps_special_tokens_whole_and_truncation_is_reported(), check_texts(), every_prompt_file_format(), load_prompt_texts(), load_tokenizer(), tokenize(), Tokenized (+2 more)

### Community 61 - "DType"
Cohesion: 0.15
Nodes (7): DType, BF16, F16, F32, Other, encode(), Lcg

### Community 62 - "model_size.rs"
Cohesion: 0.23
Nodes (13): compute_model_size(), directory_without_index_sums_top_level_files(), distinct_errors_for_empty_dir_bad_extension_missing_path(), Index, INDEX_FILE, is_safetensors(), list_safetensors(), missing_shard_is_typed_error() (+5 more)

### Community 63 - "Site"
Cohesion: 0.13
Nodes (12): Site, MlpOut, ResidPost, ChatTemplate, Chatml, NoTemplate, AutotuneToolArgs, ExtractToolArgs (+4 more)

### Community 64 - "invalid"
Cohesion: 0.39
Nodes (15): apply(), autotune(), bench(), bench_command(), call(), check(), eval(), extract_direction() (+7 more)

### Community 65 - "path"
Cohesion: 0.22
Nodes (10): check_readme(), invocations(), long_options(), lowercase_word(), nested_bench_commands_and_mcp_tool_count_are_checked(), read(), readme_commands_options_and_links_are_real(), SHELL_BLOCKS (+2 more)

### Community 66 - "dispatch"
Cohesion: 0.27
Nodes (11): apply(), dispatch(), emit(), error_text(), Io, parse_cli(), parse_cli_with(), PLAIN_UI (+3 more)

### Community 67 - "LinalgError"
Cohesion: 0.16
Nodes (11): Cholesky, indefinite_and_ill_conditioned_matrices_are_refused(), KAPPA_LIMIT, solves_a_known_system(), LinalgError, IllConditioned, NoConvergence, NonFinite (+3 more)

### Community 68 - "dataset.rs"
Cohesion: 0.23
Nodes (12): DatasetFormat, Auto, Gsm8k, Math, file(), formats_limit_and_order(), gold_answer(), Item (+4 more)

### Community 69 - "core_apply"
Cohesion: 0.14
Nodes (12): apply_outcome(), ApplyRequest, core_apply(), ObservedBackend, Rollback, Warn, rollback_line(), rollback_outcome() (+4 more)

### Community 70 - "jacobi.rs"
Cohesion: 0.24
Nodes (11): diagonal_and_analytic_two_by_two(), Eigen, MAX_N, MAX_SWEEPS, offdiag_norm(), orthogonality_error(), random_symmetric_matrices_reconstruct_sorted_and_orthonormal(), reconstruction_error() (+3 more)

### Community 71 - "Value"
Cohesion: 0.36
Nodes (6): Value, error(), result(), Session<'_, W>, bench_issues(), one_op_patch()

### Community 72 - "ppl_parse.rs"
Cohesion: 0.23
Nodes (6): final_re(), number(), parse_final(), PplEstimate, progress(), progress_re()

### Community 73 - "leace.rs"
Cohesion: 0.40
Nodes (10): classes(), cross_cov_norm(), erased_stats(), few_samples_need_shrinkage_and_overfit_out_of_sample(), fit(), invalid_and_degenerate_inputs(), LeaceFit, least_squares_slopes() (+2 more)

### Community 74 - "super"
Cohesion: 0.28
Nodes (7): histogram_rows(), render_histogram(), rows_are_bottom_anchored_and_log_lifts_small_bins(), bar_glyph(), BLOCKS, eighths(), resample()

### Community 75 - "Pooling"
Cohesion: 0.26
Nodes (7): activation(), last_and_mean_pooling_exact_values(), non_finite_activation_names_the_prompt(), pool(), Pooling, Last, Mean

### Community 76 - "ExtractEvent"
Cohesion: 0.18
Nodes (9): ExtractEvent, LoadingWeights, Planned, Prompt, Writing, PromptSet, Neg, Pos (+1 more)

### Community 77 - "Template"
Cohesion: 0.26
Nodes (5): sha256_hex(), DEFAULT_TEMPLATE, Template, VARIABLE, variable_is_required_and_replaced_everywhere()

### Community 78 - "BroadcastAxis"
Cohesion: 0.24
Nodes (9): TensorShifter, WritableModel, ALL, BroadcastAxis, Cols, Rows, ShiftOp, Add (+1 more)

### Community 79 - "matrix_panel.rs"
Cohesion: 0.36
Nodes (6): PROJECTIONS, kv(), LIST_WIDTH, render_inspector(), render_layers(), render_metrics()

### Community 80 - "CommandOutput"
Cohesion: 0.22
Nodes (5): NoRunner, CommandOutput, CancelRunner, ppl_output(), PplRunner

### Community 82 - "eval/prompts.rs"
Cohesion: 0.38
Nodes (8): EvalPrompt, invalid_files_are_clear_errors(), json_array(), jsonl_with_bom_and_crlf(), load(), load_prompts(), message(), plain_text_one_prompt_per_line()

### Community 83 - ".send"
Cohesion: 0.24
Nodes (3): Closed, ProtocolWriter<W>, writer_emits_single_lines_and_survives_a_closed_pipe()

### Community 84 - "LayerSelection"
Cohesion: 0.31
Nodes (5): LayerSpec, LayerSelection, All, List, Range

### Community 85 - "config.rs"
Cohesion: 0.33
Nodes (6): default_act(), load_config(), other_architectures_and_broken_configs_are_refused(), parse_config(), QWEN2, qwen2_parses_with_defaults()

### Community 86 - "HardwareError"
Cohesion: 0.22
Nodes (9): HardwareError, CorruptIndex, InsufficientResources, InvalidExtension, Io, MissingShard, NoSafetensors, PathNotFound (+1 more)

### Community 87 - "Method"
Cohesion: 0.40
Nodes (4): Method, Leace, MeanDiff, Svd

### Community 88 - "deps_purity.rs"
Cohesion: 0.60
Nodes (4): FORBIDDEN, no_native_build_dependency_in_the_lockfile(), packages(), zip_has_no_compression_backend()

### Community 90 - "LineRead"
Cohesion: 0.50
Nodes (4): LineRead, Eof, Line, TooLong

## Knowledge Gaps
- **251 isolated node(s):** `Ansi256`, `Off`, `Plain`, `TrueColor`, `Accent` (+246 more)
  These have ≤1 connection - possible missing edges. (Counts symbols only; 581 node(s) total have ≤1 connection when file, concept and rationale nodes are included.)
- **4 thin communities (<3 nodes) omitted from report** — run `graphify query` to explore isolated nodes.

## Suggested Questions
_Questions this graph is uniquely positioned to answer:_

- **Why does `Value` connect `Value` to `ppl_runner.rs`, `invalid`, `integration.rs`, `model`, `dataset.rs`, `tools.rs`, `ui_deps`, `src/app.rs`, `Event`, `validate.rs`, `server.rs`, `.send`, `deps`, `tui_app`, `AppDeps`, `.args`?**
  _High betweenness centrality (0.034) - this node is a cross-community bridge._
- **What connects `Ansi256`, `Off`, `Plain` to the rest of the system?**
  _251 weakly-connected nodes found - possible documentation gaps or missing edges._
- **Should `integration.rs` be split into smaller, more focused modules?**
  _Cohesion score 0.038923395445134575 - nodes in this community are weakly interconnected._
- **Why does `DType` connect `DType` to `engine/mod.rs`, `tui/app.rs`, `autotune.rs`, `logits_io.rs`, `mmap_file.rs`, `src/app.rs`, `Pooling`, `engine/stats.rs`, `shift.rs`, `hooked_qwen2.rs`, `TensorStats`, `EngineError`, `extract/mod.rs`, `weights.rs`, `dump.rs`?**
  _High betweenness centrality (0.033) - this node is a cross-community bridge._
- **Should `Ui` be split into smaller, more focused modules?**
  _Cohesion score 0.06180733162830349 - nodes in this community are weakly interconnected._
- **Why does `Projection` connect `schema.rs` to `autotune.rs`, `tools.rs`, `cli.rs`, `src/app.rs`, `validate.rs`, `OperationKind`, `executor.rs`, `Axis`, `LayerSelection`, `AppDeps`, `Site`?**
  _High betweenness centrality (0.026) - this node is a cross-community bridge._
- **Should `tui/app.rs` be split into smaller, more focused modules?**
  _Cohesion score 0.07622504537205081 - nodes in this community are weakly interconnected._