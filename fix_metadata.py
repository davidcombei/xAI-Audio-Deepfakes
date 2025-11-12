import os


def extract_filenames(old_metadata, new_metadata):
    with open(old_metadata, "r", encoding="utf-8") as fin, open(
        new_metadata, "w", encoding="utf-8"
    ) as fout:
        for line in fin:
            path = line.strip()
            if not path:
                continue
            filename = os.path.basename(path)
            fout.write(filename + "\n")


old_metadata = "/mnt/QNAP/comdav/addvisor/metadata/metrics_metadata.txt"
new_metadata = "metrics_metadata.txt"

extract_filenames(old_metadata, new_metadata)
print(f"[OK] Saved filenames only to {new_metadata}")
