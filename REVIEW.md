# Codebase Review and Improvement Suggestions

This review highlights opportunities to harden data ingestion, improve reliability, and make the training pipeline safer to operate. Suggestions are scoped to files inspected in this pass.

## Security and Data Integrity
- **Use HTTPS and verify downloads for GDELT.** `data/download_gdelt.py` pulls zips over plain HTTP with no checksum or size validation, leaving the ingestion path vulnerable to tampering and corrupted files. Switching to `https://`, enabling certificate verification (on by default), and validating content length or a hash before writing would mitigate man-in-the-middle and integrity risks. 【F:data/download_gdelt.py†L11-L88】

## Reliability and Operational Safety
- **Bound HistData backfills.** `data/download_all_fx_data.py` loops through years indefinitely until an exception occurs, which can hammer the upstream API or fill disks if invoked without guardrails. Add explicit `--start-year/--end-year` arguments (defaulting to the latest year in `pairs.csv`), break conditions, and per-run download quotas; also surface non-fatal errors with structured logging instead of relying on exceptions to terminate. 【F:data/download_all_fx_data.py†L1-L50】

## Data Quality and Training Determinism
- **Validate and sanitize raw price files before feature building.** `data/prepare_dataset.py` currently trusts CSV/zip contents and only warns on load failures. Introduce schema checks (column presence/dtypes), NaN/outlier filtering, and duplicate-timestamp detection prior to feature engineering to prevent silent training drift. Recording basic summary stats per split would also help detect data leakage or misaligned windows. 【F:data/prepare_dataset.py†L1-L78】

## Latest changes reviewed
- **UTC conversion + dedup is now mandatory in data prep.** `process_pair` now routes every dataset through `convert_to_utc_and_dedup`, which reduces timezone drift and overlapping rows risk. Consider extending this to enforce strictly monotonic timestamps and capture duplicate counts in logs/metrics for observability. 【F:data/prepare_dataset.py†L146-L181】
- **Configurable multi-head attention path.** `ModelConfig` exposes `use_multihead_attention`, and `PriceSequenceEncoder` switches to `MultiHeadTemporalAttention` when enabled. The encoder asserts divisibility of `input_dim` by the head count at runtime; surface this as an argparse/config validation earlier to fail fast before launching a long training job. Also guard against pathological head counts (e.g., zero or extremely large) that could be supplied via user config. 【F:config/config.py†L39-L65】【F:models/agent_hybrid.py†L28-L165】

## Observability and Testing
- **Add fast smoke tests for the pipeline.** Consider a minimal test that builds a tiny synthetic dataset, runs `features.build_feature_frame`, `DataAgent.window_data`, and a single training/eval step to detect regressions in data formats and model wiring. This can live under `tests/` and run in CI with CPU-only settings.

## Next Steps
- Prioritize securing external downloads (GDELT and HistData) since these are entry points for untrusted data.
- Add bounded download controls and integrity checks before expanding training coverage.
- Layer lightweight validation and observability around dataset preparation to catch issues early.
