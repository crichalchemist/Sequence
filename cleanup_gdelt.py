"""Cleanup script to eliminate redundancy and consolidate GDELT functionality."""
import shutil
from pathlib import Path


def cleanup_gdelt_redundancy():
    """Remove redundant files and consolidate GDELT functionality."""
    
    # Files to remove (redundant with consolidated versions)
    files_to_remove = [
        "data/download_gdelt.py",  # Replaced by gdelt/consolidated_downloader.py
        "gdelt/downloader.py",     # Merged into consolidated_downloader.py
        "data/gdelt_ingest.py",    # Functionality moved to feature_builder.py
    ]
    
    backup_dir = Path("backup_removed_files")
    backup_dir.mkdir(exist_ok=True)
    
    for file_path in files_to_remove:
        file_p = Path(file_path)
        if file_p.exists():
            # Create backup
            backup_path = backup_dir / file_p.name
            shutil.copy2(file_p, backup_path)
            print(f"Backed up {file_path} to {backup_path}")
            
            # Remove original
            file_p.unlink()
            print(f"Removed redundant file: {file_path}")
    
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
            print(f"Updated imports in: {file_path}")
    
    print("\n✅ Import updates completed!")


if __name__ == "__main__":
    print("🧹 Starting GDELT module cleanup...")
    cleanup_gdelt_redundancy()
    update_gdelt_imports()
    print("\n🎉 Cleanup completed successfully!")

