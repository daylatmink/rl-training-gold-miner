import os
import json
from pathlib import Path

def merge_levels():
    """
    Tập hợp tất cả các file JSON từ thư mục levels/* thành một file JSON duy nhất
    với cấu trúc giống levels.json
    Key sẽ được đổi thành format "LX_Y" (X = level number, Y = thứ tự trong folder)
    """
    levels_dir = Path("levels")
    merged_data = {}
    
    # Duyệt qua tất cả các thư mục level_* theo thứ tự
    level_folders = sorted(
        [f for f in levels_dir.iterdir() if f.is_dir() and f.name.startswith("level_")],
        key=lambda x: int(x.name.split("_")[1])
    )
    
    for level_folder in level_folders:
        level_num = level_folder.name.split("_")[1]  # Lấy số level (0, 1, 2, ...)
        print(f"Processing folder: {level_folder.name}")
        
        # Duyệt qua tất cả các file JSON trong mỗi thư mục level
        json_files = sorted(level_folder.glob("*.json"))
        
        for idx, json_file in enumerate(json_files, start=1):
            print(f"  Reading file: {json_file.name}")
            
            try:
                with open(json_file, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                
                # Lấy value từ file JSON và đổi key thành LX_Y
                for old_key, value in data.items():
                    new_key = f"L{level_num}_{idx}"
                    if new_key in merged_data:
                        print(f"    Warning: Duplicate key '{new_key}', overwriting...")
                    merged_data[new_key] = value
                    print(f"    Renamed: {old_key} -> {new_key}")
                    
            except json.JSONDecodeError as e:
                print(f"    Error reading {json_file}: {e}")
            except Exception as e:
                print(f"    Unexpected error with {json_file}: {e}")
    
    # Ghi ra file merged_levels.json
    output_file = "merged_levels.json"
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(merged_data, f, indent=4)
    
    print(f"\nMerged {len(merged_data)} levels into {output_file}")
    return merged_data

if __name__ == "__main__":
    merge_levels()
