"""
Batch-convert HistData CSVs from Eastern Time to Central Time.

Reads zipped CSVs under input_root (default: output/) and writes converted
zips to output_root (default: output_central/). Timestamp column is adjusted
by the provided hour offset (default: -1 hour, Central behind Eastern).
"""

import csv
import io
import os
import sys
import tempfile
from datetime import datetime, timedelta
from pathlib import Path
from zipfile import ZipFile


def convert_zip(zip_path: Path, dest_zip: Path, hour_offset: int) -> None:
    if dest_zip.exists():
        print(f"[skip] {dest_zip} already exists")
        return

    with ZipFile(zip_path, "r") as zin:
        csv_names = [n for n in zin.namelist() if n.lower().endswith(".csv")]
        txt_names = [n for n in zin.namelist() if n.lower().endswith(".txt")]
        if not csv_names:
            print(f"[warn] no CSV in {zip_path}")
            return
        csv_name = csv_names[0]

        with zin.open(csv_name) as fin, tempfile.NamedTemporaryFile(
            "w+", delete=False, newline=""
        ) as tmp:
            reader = csv.reader(io.TextIOWrapper(fin, newline=""), delimiter=";")
            writer = csv.writer(tmp, delimiter=";")
            for row in reader:
                if not row:
                    continue
                try:
                    ts = datetime.strptime(row[0], "%Y%m%d %H%M%S")
                    ts += timedelta(hours=hour_offset)
                    row[0] = ts.strftime("%Y%m%d %H%M%S")
                except Exception:
                    # Keep row unchanged if it does not match expected format.
                    pass
                writer.writerow(row)
            tmp_path = Path(tmp.name)

        dest_zip.parent.mkdir(parents=True, exist_ok=True)
        with ZipFile(dest_zip, "w") as zout:
            zout.write(tmp_path, arcname=csv_name)
            for name in txt_names:
                with zin.open(name) as fin, zout.open(name, "w") as fout:
                    fout.write(fin.read())
        tmp_path.unlink(missing_ok=True)
    print(f"[ok] {zip_path.name} -> {dest_zip}")


def main():
    # Defaults: convert output/ -> output_central/ with -1 hour shift (EST -> CST).
    input_root = Path(os.environ.get("INPUT_ROOT", "output")).resolve()
    output_root = Path(os.environ.get("OUTPUT_ROOT", "output_central")).resolve()
    hour_offset = int(os.environ.get("HOUR_OFFSET", "-1"))

    if not input_root.exists():
        print(f"Input root {input_root} does not exist")
        sys.exit(1)

    zip_files = sorted(input_root.rglob("*.zip"))
    if not zip_files:
        print(f"No zip files found under {input_root}")
        return

    for zf in zip_files:
        rel = zf.relative_to(input_root)
        dest_zip = output_root / rel.parent / (zf.stem + "_cst.zip")
        convert_zip(zf, dest_zip, hour_offset)


if __name__ == "__main__":
    main()
