import os
import shutil


def copy_files_from_metadata(metadata_path, dest_dir):
    os.makedirs(dest_dir, exist_ok=True)

    with open(metadata_path, "r", encoding="utf-8") as f:
        for line in f:
            src_path = line.strip()
            if not src_path:
                continue
            if not os.path.isfile(src_path):
                print(f"[WARN] File does not exist: {src_path}")
                continue

            filename = os.path.basename(src_path)
            dest_path = os.path.join(dest_dir, filename)

            shutil.copy2(src_path, dest_path)
            print(f"[OK] Copied: {src_path} -> {dest_path}")


metadata_file = "/mnt/QNAP/comdav/addvisor/metadata/metrics_metadata.txt"
destination_folder = "data_copy/"
copy_files_from_metadata(metadata_file, destination_folder)
