"""Cleanup script to eliminate redundancy and consolidate GDELT functionality."""
import logging
import shutil
from pathlib import Path

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def cleanup_gdelt_redundancy():
    """Remove redundant files and consolidate GDELT functionality."""
    
    # Files to remove (redundant with consolidated versions)
    files_to_remove = [
        "data/download_gdelt.py",  # Replaced by gdelt/consolidated_downloader.py
        "gdelt/downloader.py",     # Merged into consolidated_downloader.py
        "data/gdelt_ingest.py",    # Functionality moved to feature_builder.py
    ]
    
    backup_dir = Path("backup_removed_files")
    try:
        backup_dir.mkdir(exist_ok=True)
    except (OSError, PermissionError) as e:
        logger.error(f"Failed to create backup directory {backup_dir}: {e}")
        return
    
    for file_path in files_to_remove:
        file_p = Path(file_path)
        if file_p.exists():
            # Create backup
            backup_path = backup_dir / file_p.name
            try:
                shutil.copy2(file_p, backup_path)
                logger.info(f"Backed up {file_path} to {backup_path}")
            except (OSError, PermissionError, shutil.Error) as e:
                logger.error(f"Failed to backup {file_path}: {e}")
                continue
            
            # Remove original
            try:
                file_p.unlink()
                logger.info(f"Removed redundant file: {file_path}")
            except (OSError, PermissionError) as e:
                logger.error(f"Failed to remove {file_path}: {e}")
                continue
    
    print("\n✅ Redundancy cleanup completed!")
    print(f"Backups stored in: {backup_dir}")

def update_gdelt_imports():
    """Update import statements to use consolidated modules."""
    
    files_to_update = [
        "tests/test_gdelt_alignment.py",
        "tests/test_gdelt_parser.py", 
        "run/training_pipeline.py",
        # Add other files that import GDELT modules
    ]
    
    for file_path in files_to_update:
        file_p = Path(file_path)
        if file_p.exists():
            try:
                content = file_p.read_text()
                
                # Update import statements
                content = content.replace(
                    "from gdelt.downloader import GDELTDownloader",
                    "from gdelt.consolidated_downloader import GDELTDownloader"
                )
                content = content.replace(
                    "from data.download_gdelt import",
                    "from gdelt.consolidated_downloader import"
                )
                content = content.replace(
                    "from data.gdelt_ingest import",
                    "from gdelt.feature_builder import GDELTTimeSeriesBuilder"
                )
                
                file_p.write_text(content)
                logger.info(f"Updated imports in: {file_path}")
            except (OSError, PermissionError, UnicodeDecodeError) as e:
                logger.error(f"Failed to update imports in {file_path}: {e}")
                continue
    
    print("\n✅ Import updates completed!")


if __name__ == "__main__":
    print("🧹 Starting GDELT module cleanup...")
    cleanup_gdelt_redundancy()
    update_gdelt_imports()
    print("\n🎉 Cleanup completed successfully!")

