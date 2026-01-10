import os
import sys
from pathlib import Path

def rename_photos_to_numbers(folder_path):
    """
    Rename all image files in a folder to sequential numbers (0001, 0002, etc.)
    
    Args:
        folder_path: Path to the folder containing photos
    """
    # Supported image extensions
    image_extensions = {'.jpg', '.jpeg', '.png', '.gif', '.bmp', '.tiff', '.webp', '.raw', '.cr2', '.nef'}
    
    # Get the folder path
    folder = Path(folder_path)
    
    if not folder.exists():
        print(f"Error: Folder '{folder_path}' does not exist!")
        return
    
    if not folder.is_dir():
        print(f"Error: '{folder_path}' is not a directory!")
        return
    
    # Get all image files in the folder
    image_files = []
    for file in folder.iterdir():
        if file.is_file() and file.suffix.lower() in image_extensions:
            image_files.append(file)
    
    if not image_files:
        print(f"No image files found in '{folder_path}'")
        return
    
    # Sort files by name to maintain a consistent order
    image_files.sort(key=lambda x: x.name)
    
    print(f"Found {len(image_files)} image file(s) to rename")
    print("\nPreview of renaming:")
    print("-" * 60)
    
    # Calculate the number of digits needed (minimum 4)
    num_digits = max(4, len(str(len(image_files))))
    
    # Preview the renaming
    rename_map = []
    for index, file in enumerate(image_files, start=1):
        new_name = f"{str(index).zfill(num_digits)}{file.suffix}"
        new_path = folder / new_name
        rename_map.append((file, new_path, new_name))
        print(f"{file.name} -> {new_name}")
    
    print("-" * 60)
    
    # Ask for confirmation
    response = input(f"\nProceed with renaming {len(image_files)} file(s)? (yes/no): ").strip().lower()
    
    if response not in ['yes', 'y']:
        print("Renaming cancelled.")
        return
    
    # Perform the renaming
    print("\nRenaming files...")
    success_count = 0
    
    # Use temporary names first to avoid conflicts
    temp_rename_map = []
    for original, final, _ in rename_map:
        temp_name = original.parent / f"temp_{original.name}"
        temp_rename_map.append((original, temp_name, final))
    
    # First pass: rename to temporary names
    for original, temp, _ in temp_rename_map:
        try:
            original.rename(temp)
        except Exception as e:
            print(f"Error renaming {original.name} to temporary name: {e}")
            return
    
    # Second pass: rename from temporary to final names
    for _, temp, final in temp_rename_map:
        try:
            temp.rename(final)
            success_count += 1
        except Exception as e:
            print(f"Error renaming {temp.name} to {final.name}: {e}")
            return
    
    print(f"\nSuccessfully renamed {success_count} file(s)!")

if __name__ == "__main__":
    if len(sys.argv) > 1:
        folder_path = sys.argv[1]
    else:
        folder_path = "D:/SD-WEBUI-AKI-V5.0/scripts/adjust"
        # Remove quotes if user copied path with quotes
        folder_path = folder_path.strip('"').strip("'")
    
    rename_photos_to_numbers(folder_path)
