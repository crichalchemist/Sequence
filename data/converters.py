"""Timezone conversion utilities for HistData CSV files.

This module provides functions to convert HistData CSVs between different timezones,
commonly used to standardize data from Eastern Time (EST/EDT) to Central Time (CST/CDT)
or other target timezones.
"""

import csv
import io
import os
import sys
import tempfile
from datetime import datetime, timedelta
from pathlib import Path
from typing import Optional, Union
from zipfile import ZipFile


def convert_csv_timezone(
    input_path: Union[str, Path],
    output_path: Union[str, Path],
    hour_offset: int,
    delimiter: str = ";"
) -> None:
    """Convert a single CSV file's timestamps by a given hour offset.
    
    Parameters
    ----------
    input_path : str or Path
        Path to input CSV file.
    output_path : str or Path
        Path to output CSV file.
    hour_offset : int
        Number of hours to offset timestamps (can be negative).
    delimiter : str, optional
        CSV delimiter. Default: ";".
    
    Examples
    --------
    >>> convert_csv_timezone("data_est.csv", "data_cst.csv", -1)
    # Converts EST to CST (1 hour behind)
    """
    input_path = Path(input_path)
    output_path = Path(output_path)
    
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(input_path, 'r') as r, open(output_path, 'w', newline='') as w:
        reader = csv.reader(r, delimiter=delimiter)
        writer = csv.writer(w, delimiter=delimiter)
        
        for row in reader:
            if not row:
                continue
            
            try:
                # Parse timestamp (HistData format: YYYYMMDD HHMMSS)
                ts = datetime.strptime(row[0], '%Y%m%d %H%M%S')
                # Apply offset
                ts += timedelta(hours=hour_offset)
                # Update row
                row[0] = ts.strftime('%Y%m%d %H%M%S')
            except (ValueError, IndexError):
                # Keep row unchanged if format doesn't match
                pass
            
            writer.writerow(row)


def convert_zip_timezone(
    zip_path: Union[str, Path],
    dest_zip: Union[str, Path],
    hour_offset: int,
    skip_existing: bool = True
) -> bool:
    """Convert a zipped HistData CSV's timestamps to a different timezone.
    
    Reads a ZIP file containing CSV data, adjusts timestamps by the given offset,
    and writes a new ZIP file with converted data. Preserves any .txt metadata files.
    
    Parameters
    ----------
    zip_path : str or Path
        Path to input ZIP file.
    dest_zip : str or Path
        Path to output ZIP file.
    hour_offset : int
        Number of hours to offset timestamps (can be negative).
    skip_existing : bool, optional
        If True, skip conversion if dest_zip already exists. Default: True.
        
    Returns
    -------
    bool
        True if conversion was performed, False if skipped.
        
    Examples
    --------
    >>> convert_zip_timezone("output/gbpusd/2023.zip", "output_central/gbpusd/2023_cst.zip", -1)
    """
    zip_path = Path(zip_path)
    dest_zip = Path(dest_zip)
    
    if skip_existing and dest_zip.exists():
        print(f"[skip] {dest_zip} already exists")
        return False
    
    with ZipFile(zip_path, "r") as zin:
        # Find CSV and TXT files
        csv_names = [n for n in zin.namelist() if n.lower().endswith(".csv")]
        txt_names = [n for n in zin.namelist() if n.lower().endswith(".txt")]
        
        if not csv_names:
            print(f"[warn] No CSV found in {zip_path}")
            return False
        
        csv_name = csv_names[0]
        
        # Convert CSV in temporary file
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
                except (ValueError, IndexError):
                    pass
                writer.writerow(row)
            
            tmp_path = Path(tmp.name)
        
        # Write new ZIP with converted CSV
        dest_zip.parent.mkdir(parents=True, exist_ok=True)
        with ZipFile(dest_zip, "w") as zout:
            zout.write(tmp_path, arcname=csv_name)
            # Copy metadata files
            for name in txt_names:
                with zin.open(name) as fin, zout.open(name, "w") as fout:
                    fout.write(fin.read())
        
        tmp_path.unlink(missing_ok=True)
    
    print(f"[ok] {zip_path.name} -> {dest_zip}")
    return True


def batch_convert_timezone(
    input_root: Union[str, Path],
    output_root: Union[str, Path],
    hour_offset: int,
    suffix: str = "_cst"
) -> int:
    """Batch convert all ZIP files in a directory tree to a different timezone.
    
    Parameters
    ----------
    input_root : str or Path
        Root directory containing ZIP files to convert.
    output_root : str or Path
        Root directory for converted ZIP files.
    hour_offset : int
        Number of hours to offset timestamps (can be negative).
    suffix : str, optional
        Suffix to append to output filenames. Default: "_cst".
        
    Returns
    -------
    int
        Number of files successfully converted.
        
    Examples
    --------
    >>> batch_convert_timezone("output/", "output_central/", -1)
    # Converts all ZIPs from EST to CST
    """
    input_root = Path(input_root).resolve()
    output_root = Path(output_root).resolve()
    
    if not input_root.exists():
        print(f"Input root {input_root} does not exist")
        return 0
    
    zip_files = sorted(input_root.rglob("*.zip"))
    if not zip_files:
        print(f"No ZIP files found under {input_root}")
        return 0
    
    converted = 0
    for zf in zip_files:
        rel = zf.relative_to(input_root)
        dest_zip = output_root / rel.parent / f"{zf.stem}{suffix}.zip"
        if convert_zip_timezone(zf, dest_zip, hour_offset):
            converted += 1
    
    return converted


def main():
    """CLI interface for batch timezone conversion.
    
    Environment Variables
    ---------------------
    INPUT_ROOT : str
        Input directory containing ZIP files. Default: "output".
    OUTPUT_ROOT : str
        Output directory for converted files. Default: "output_central".
    HOUR_OFFSET : int
        Hour offset for conversion. Default: "-1" (EST to CST).
        
    Examples
    --------
    $ python data/converters.py
    $ INPUT_ROOT=output OUTPUT_ROOT=output_utc HOUR_OFFSET=5 python data/converters.py
    """
    input_root = os.environ.get("INPUT_ROOT", "output")
    output_root = os.environ.get("OUTPUT_ROOT", "output_central")
    hour_offset = int(os.environ.get("HOUR_OFFSET", "-1"))
    
    print(f"Converting {input_root} -> {output_root} with offset {hour_offset}h")
    converted = batch_convert_timezone(input_root, output_root, hour_offset)
    print(f"\n✅ Converted {converted} files")


if __name__ == "__main__":
    main()
