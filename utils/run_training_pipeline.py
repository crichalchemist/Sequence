"""
Interactive pipeline runner to prepare data, train, and optionally evaluate
for one or more FX pairs. It pauses after each training run so you can decide
whether to queue additional datasets before proceeding.

Example (with optional GDELT download first):
  python utils/run_training_pipeline.py \
    --run-gdelt-download \
    --gdelt-start-date 2024-01-01 --gdelt-end-date 2024-01-07 --gdelt-resolution daily \
    --pairs gbpusd,eurusd \
    --t-in 120 --t-out 10 \
    --epochs 3 \
    --checkpoint-dir models
"""

import argparse
import os
import subprocess
import sys
from datetime import datetime
from pathlib import Path
from typing import List, Optional
from copy import deepcopy

import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from config.config import ModelConfig, TrainingConfig  # noqa: E402
from data.prepare_dataset import process_pair  # noqa: E402
from eval.agent_eval import evaluate_model  # noqa: E402
from models.agent_hybrid import build_model  # noqa: E402
from train.agent_train import train_model  # noqa: E402


def parse_pairs(pairs: str, pairs_file: Optional[Path]) -> List[str]:
    seeds: List[str] = []
    if pairs:
        seeds.extend([p.strip().lower() for p in pairs.split(",") if p.strip()])
    if pairs_file:
        seeds.extend([line.strip().lower() for line in pairs_file.read_text().splitlines() if line.strip()])
    # Preserve order, drop duplicates.
    seen = set()
    ordered: List[str] = []
    for p in seeds:
        if p not in seen:
            seen.add(p)
            ordered.append(p)
    return ordered


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Prepare, train, and optionally evaluate pairs sequentially.")
    parser.add_argument("--pairs", default="gbpusd", help="Comma-separated pair codes to run first.")
    parser.add_argument("--pairs-file", type=Path, help="Optional file with one pair per line to append.")
    parser.add_argument("--years", default=None, help="Comma-separated years to include (default: all available).")
    parser.add_argument("--input-root", default="output_central", help="Root containing Central-time zips.")
    parser.add_argument("--t-in", type=int, default=120, help="Lookback window length.")
    parser.add_argument("--t-out", type=int, default=10, help="Prediction horizon.")
    parser.add_argument("--task-type", choices=["classification", "regression"], default="classification")
    parser.add_argument("--flat-threshold", type=float, default=0.0001)
    parser.add_argument("--train-ratio", type=float, default=0.7)
    parser.add_argument("--val-ratio", type=float, default=0.15)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=0.0)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--checkpoint-dir", default="models", help="Directory to write checkpoints per pair.")
    parser.add_argument(
        "--force-train",
        dest="resume_if_ckpt",
        action="store_false",
        help="Train even if a checkpoint already exists for the pair.",
    )
    parser.set_defaults(resume_if_ckpt=True)
    parser.add_argument("--skip-eval", action="store_true", help="Skip test-set evaluation after training.")
    parser.add_argument(
        "--pause",
        dest="no_pause",
        action="store_false",
        help="Enable interactive prompts after each training run.",
    )
    parser.add_argument(
        "--no-game",
        dest="offer_game",
        action="store_false",
        help="Skip offering the optional training mini-game.",
    )
    parser.set_defaults(no_pause=True, offer_game=True)
    parser.add_argument(
        "--run-gdelt-download",
        action="store_true",
        help="Download GDELT GKG files before training.",
    )
    parser.add_argument("--gdelt-start-date", default="2016-01-01", help="Start date for GDELT download (YYYY-MM-DD).")
    parser.add_argument(
        "--gdelt-end-date",
        default=None,
        help="End date for GDELT download (YYYY-MM-DD). Defaults to today if omitted.",
    )
    parser.add_argument(
        "--gdelt-resolution",
        choices=["daily", "15min"],
        default="daily",
        help="GDELT download cadence.",
    )
    parser.add_argument("--gdelt-step-minutes", type=int, default=15, help="Cadence for 15min resolution.")
    parser.add_argument("--gdelt-out-dir", default="data/gdelt", help="Destination folder for GDELT zips.")
    parser.add_argument("--gdelt-overwrite", action="store_true", help="Re-download even if files exist.")
    parser.add_argument("--gdelt-timeout", type=int, default=10, help="HTTP timeout per GDELT request (seconds).")
    parser.add_argument("--gdelt-max-retries", type=int, default=3, help="Retries per GDELT file.")
    parser.add_argument("--gdelt-retry-backoff", type=float, default=2.0, help="Backoff seconds multiplied by attempt.")
    parser.add_argument(
        "--delete-input-zips",
        action="store_true",
        help="After a pair finishes training/eval, delete its input zip files to reclaim disk.",
    )
    parser.add_argument(
        "--run-histdata-download",
        action="store_true",
        help="Download HistData zips before training using data/download_all_fx_data.py (respects pairs/years).",
    )
    parser.add_argument(
        "--histdata-output-root",
        default=None,
        help="Override output directory for HistData download (defaults to input-root).",
    )
    parser.add_argument(
        "--run-yfinance-download",
        action="store_true",
        help="Download FX data from yfinance before training (pairs and date range required).",
    )
    parser.add_argument("--yf-start", default=None, help="Start date for yfinance download (YYYY-MM-DD).")
    parser.add_argument("--yf-end", default=None, help="End date for yfinance download (YYYY-MM-DD).")
    parser.add_argument("--yf-interval", default="1m", help="yfinance interval (e.g., 1m, 5m, 1h).")
    parser.add_argument(
        "--yf-output-root",
        default=None,
        help="Override output directory for yfinance download (defaults to input-root).",
    )
    parser.add_argument(
        "--normalize-yfinance",
        action="store_true",
        help="After yfinance download, upsample to synthetic 1-minute bars.",
    )
    parser.add_argument(
        "--no-auto-download",
        dest="auto_download_missing",
        action="store_false",
        help="Disable automatic download when pair data is missing.",
    )
    parser.set_defaults(auto_download_missing=True)
    return parser.parse_args()


def maybe_prompt_for_more(queue: List[str]) -> List[str]:
    try:
        extra = input("Add more pairs (comma-separated) or press Enter to continue: ").strip()
    except EOFError:
        extra = ""
    if not extra:
        return queue
    for p in [part.strip().lower() for part in extra.split(",") if part.strip()]:
        if p and p not in queue:
            queue.append(p)
    return queue


def cleanup_pair_zips(pair: str, input_root: Path, years: Optional[str]) -> None:
    root = Path(input_root)
    target_dir = root / pair
    if not target_dir.exists():
        return
    filters = None
    if years:
        filters = {y.strip() for y in years.split(",") if y.strip()}
    removed = 0
    for zp in target_dir.glob("*.zip"):
        if filters and not any(y in zp.name for y in filters):
            continue
        try:
            zp.unlink()
            removed += 1
        except Exception as exc:  # pragma: no cover - defensive
            print(f"[warn] could not delete {zp}: {exc}")
    if removed:
        print(f"[info] deleted {removed} zip(s) for {pair} under {target_dir}")


def has_local_data(pair: str, input_root: Path, years: Optional[str]) -> bool:
    pair_dir = Path(input_root) / pair
    if not pair_dir.exists():
        return False
    zips = list(pair_dir.glob("*.zip"))
    csvs = list(pair_dir.glob("*.csv"))
    if years:
        filt = {y.strip() for y in years.split(",") if y.strip()}
        zips = [z for z in zips if any(y in z.name for y in filt)]
        csvs = [c for c in csvs if any(y in c.name for y in filt)]
    return bool(zips or csvs)


def run_yfinance_download(args) -> None:
    script = ROOT / "data" / "download_yfinance_fx.py"
    if not script.exists():
        print("[warn] yfinance downloader script not found; skipping.")
        return
    output_root = args.yf_output_root or args.input_root
    if not args.yf_start or not args.yf_end:
        print("[warn] yfinance download requires --yf-start and --yf-end; skipping.")
        return
    cmd = [
        sys.executable,
        str(script),
        "--pairs",
        args.pairs,
        "--start",
        args.yf_start,
        "--end",
        args.yf_end,
        "--interval",
        args.yf_interval,
        "--output-root",
        str(output_root),
    ]
    try:
        subprocess.run(cmd, check=True, cwd=ROOT)
        # Ensure pair folders exist even if download returned empty.
        for pair in [p.strip().lower() for p in args.pairs.split(",") if p.strip()]:
            (Path(output_root) / pair).mkdir(parents=True, exist_ok=True)
    except Exception as exc:  # pragma: no cover - defensive
        print(f"[warn] yfinance download failed: {exc}")


def normalize_yfinance(args) -> None:
    if not args.normalize_yfinance:
        return
    script = ROOT / "data" / "normalize_yfinance_to_m1.py"
    if not script.exists():
        print("[warn] yfinance normalization script not found; skipping.")
        return
    output_root = args.yf_output_root or args.input_root
    cmd = [
        sys.executable,
        str(script),
        "--input-root",
        str(output_root),
        "--output-root",
        str(output_root),
        "--pairs",
        args.pairs,
    ]
    try:
        subprocess.run(cmd, check=True, cwd=ROOT)
    except Exception as exc:  # pragma: no cover - defensive
        print(f"[warn] yfinance normalization failed: {exc}")


def auto_download_if_missing(pair: str, args) -> None:
    if not args.auto_download_missing:
        return
    if has_local_data(pair, Path(args.input_root), args.years):
        return
    print(f"[info] No local data found for {pair}; triggering downloads.")
    tmp_args = deepcopy(args)
    tmp_args.pairs = pair
    # Run histdata if requested globally.
    if args.run_histdata_download:
        run_histdata_download(tmp_args)
    # Run yfinance if requested and start/end provided.
    if args.run_yfinance_download:
        if not (args.yf_start and args.yf_end):
            print("[warn] yfinance download skipped: --yf-start/--yf-end required.")
        else:
            run_yfinance_download(tmp_args)
            normalize_yfinance(tmp_args)


def run_histdata_download(args) -> None:
    script = ROOT / "data" / "download_all_fx_data.py"
    if not script.exists():
        print("[warn] HistData downloader script not found; skipping.")
        return
    env = os.environ.copy()
    output_root = args.histdata_output_root or args.input_root
    env["FX_DATA_OUTPUT"] = str(output_root)
    try:
        # Respect pairs by trimming pairs.csv if pairs was provided.
        if args.pairs:
            pairs_set = {p.strip().lower() for p in args.pairs.split(",") if p.strip()}
            pairs_csv = ROOT / "pairs.csv"
            if pairs_csv.exists():
                tmp_csv = Path(env.get("TMPDIR", "/tmp")) / "pairs_filtered.csv"
                with open(pairs_csv, "r") as src, open(tmp_csv, "w") as dst:
                    for i, line in enumerate(src):
                        if i == 0:
                            dst.write(line)
                            continue
                        parts = line.strip().split(",")
                        if len(parts) >= 2 and parts[1].lower() in pairs_set:
                            dst.write(line + "\n")
                env["PAIRS_CSV"] = str(tmp_csv)
        subprocess.run([sys.executable, str(script)], check=True, env=env, cwd=ROOT)
    except Exception as exc:  # pragma: no cover - defensive
        print(f"[warn] HistData download failed: {exc}")


def maybe_offer_game(enabled: bool) -> None:
    if not enabled or not sys.stdout.isatty():
        return
    game_path = ROOT / "utils" / "play_training_game.py"
    if not game_path.exists():
        return
    try:
        resp = input(
            "Launch the mini-game during training? This may slow training. [y/N]: "
        ).strip()
    except EOFError:
        return
    if resp.lower() != "y":
        return
    print("Starting mini-game in this terminal. Quit with Q to return.")
    subprocess.Popen([sys.executable, str(game_path)])


def run_gdelt_download(args) -> None:
    end_date = args.gdelt_end_date or datetime.utcnow().strftime("%Y-%m-%d")
    cmd = [
        sys.executable,
        str(ROOT / "data" / "download_gdelt.py"),
        "--start-date",
        args.gdelt_start_date,
        "--end-date",
        end_date,
        "--resolution",
        args.gdelt_resolution,
        "--step-minutes",
        str(args.gdelt_step_minutes),
        "--out-dir",
        args.gdelt_out_dir,
        "--timeout",
        str(args.gdelt_timeout),
        "--max-retries",
        str(args.gdelt_max_retries),
        "--retry-backoff",
        str(args.gdelt_retry_backoff),
    ]
    if args.gdelt_overwrite:
        cmd.append("--overwrite")
    print(
        f"[info] Downloading GDELT to {args.gdelt_out_dir} "
        f"({args.gdelt_resolution}, {args.gdelt_start_date} -> {end_date})"
    )
    subprocess.run(cmd, check=True)


def main() -> None:
    args = parse_args()
    pair_queue = parse_pairs(args.pairs, args.pairs_file)
    if not pair_queue:
        raise ValueError("No pairs provided.")

    if args.run_gdelt_download:
        try:
            run_gdelt_download(args)
        except Exception as exc:  # pragma: no cover - defensive
            print(f"[warn] GDELT download failed: {exc}")
    if args.run_histdata_download:
        run_histdata_download(args)
    if args.run_yfinance_download:
        run_yfinance_download(args)
        normalize_yfinance(args)

    # Offer the optional mini-game once before training starts.
    maybe_offer_game(args.offer_game)

    # Choose device sensibly.
    device = args.device
    if device.startswith("cuda") and not torch.cuda.is_available():
        device = "cpu"
    device = torch.device(device)

    ckpt_root = Path(args.checkpoint_dir)
    ckpt_root.mkdir(parents=True, exist_ok=True)

    while pair_queue:
        pair = pair_queue.pop(0)
        print(f"\n=== Running pair: {pair} ===")

        # Auto-download if the pair folder is missing or empty.
        auto_download_if_missing(pair, args)

        class PrepArgs:
            pairs = pair
            input_root = args.input_root
            years = args.years
            t_in = args.t_in
            t_out = args.t_out
            target_type = args.task_type
            flat_threshold = args.flat_threshold
            train_ratio = args.train_ratio
            val_ratio = args.val_ratio

        try:
            pair_name, loaders = process_pair(pair, PrepArgs)
        except Exception as exc:  # pragma: no cover - defensive logging only
            print(f"[error] data prep failed for {pair}: {exc}")
            continue

        train_loader = loaders["train"]
        val_loader = loaders["val"]
        test_loader = loaders["test"]
        num_features = next(iter(train_loader))[0].shape[-1]

        ckpt_path = ckpt_root / f"{pair_name}_best_model.pt"
        ckpt_exists = ckpt_path.exists()
        model_cfg = ModelConfig(
            num_features=num_features,
            num_classes=3 if args.task_type == "classification" else None,
        )
        train_cfg = TrainingConfig(
            batch_size=args.batch_size,
            epochs=args.epochs,
            learning_rate=args.learning_rate,
            weight_decay=args.weight_decay,
            device=str(device),
            checkpoint_path=str(ckpt_path),
        )

        model = build_model(model_cfg, task_type=args.task_type).to(device)
        history = None

        if ckpt_exists and args.resume_if_ckpt:
            try:
                state = torch.load(ckpt_path, map_location=device)
                model.load_state_dict(state)
                print(f"[info] Found checkpoint for {pair_name}, skipping training (use --force-train to retrain).")
            except Exception as exc:  # pragma: no cover - defensive
                print(f"[warn] Failed to load existing checkpoint for {pair_name}: {exc}")

        if not (ckpt_exists and args.resume_if_ckpt):
            try:
                history = train_model(
                    model,
                    train_loader,
                    val_loader,
                    train_cfg,
                    task_type=args.task_type,
                )
                print(f"[done] training {pair_name}; history keys: {list(history.keys())}")
            except KeyboardInterrupt:
                print(f"[warn] Training interrupted for {pair_name}; keeping current model state.")
            except Exception as exc:  # pragma: no cover - defensive
                print(f"[error] training failed for {pair_name}: {exc}")
                continue

        if not args.skip_eval:
            metrics = evaluate_model(model, test_loader, task_type=args.task_type)
            print(f"[eval] {pair_name}: {metrics}")

        if args.delete_input_zips:
            cleanup_pair_zips(pair_name, Path(args.input_root), args.years)

        if args.no_pause:
            continue
        pair_queue = maybe_prompt_for_more(pair_queue)


if __name__ == "__main__":
    main()
