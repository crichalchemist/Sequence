# Codebase Review and Improvement Suggestions

This review highlights opportunities to harden data ingestion, improve reliability, and make the training pipeline safer to operate. Suggestions are scoped to files inspected in this pass.

## Security and Data Integrity
- **Use HTTPS and verify downloads for GDELT.** `data/download_gdelt.py` pulls zips over plain HTTP with no checksum or size validation, leaving the ingestion path vulnerable to tampering and corrupted files. Switching to `https://`, enabling certificate verification (on by default), and validating content length or a hash before writing would mitigate man-in-the-middle and integrity risks. 【F:data/download_gdelt.py†L11-L88】

## Reliability and Operational Safety
- **Bound HistData backfills.** `data/download_all_fx_data.py` loops through years indefinitely until an exception occurs, which can hammer the upstream API or fill disks if invoked without guardrails. Add explicit `--start-year/--end-year` arguments (defaulting to the latest year in `pairs.csv`), break conditions, and per-run download quotas; also surface non-fatal errors with structured logging instead of relying on exceptions to terminate. 【F:data/download_all_fx_data.py†L1-L50】

## Data Quality and Training Determinism
- **Validate and sanitize raw price files before feature building.** `data/prepare_dataset.py` currently trusts CSV/zip contents and only warns on load failures. Introduce schema checks (column presence/dtypes), NaN/outlier filtering, and duplicate-timestamp detection prior to feature engineering to prevent silent training drift. Recording basic summary stats per split would also help detect data leakage or misaligned windows. 【F:data/prepare_dataset.py†L1-L78】

## Observability and Testing
- **Add fast smoke tests for the pipeline.** Consider a minimal test that builds a tiny synthetic dataset, runs `features.build_feature_frame`, `DataAgent.window_data`, and a single training/eval step to detect regressions in data formats and model wiring. This can live under `tests/` and run in CI with CPU-only settings.

## Next Steps
- Prioritize securing external downloads (GDELT and HistData) since these are entry points for untrusted data.
- Add bounded download controls and integrity checks before expanding training coverage.
- Layer lightweight validation and observability around dataset preparation to catch issues early.
