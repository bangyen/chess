import os
import re
import sys
import urllib.request
from typing import List

BASE_URL = "http://tablebase.sesse.net/syzygy/3-4-5/"
DEFAULT_DEST = os.path.expanduser("~/syzygy")


def fetch_file_list() -> List[str]:
    print(f"Fetching file list from {BASE_URL}...")
    try:
        with urllib.request.urlopen(BASE_URL) as response:  # noqa: S310
            html = response.read().decode("utf-8")
            # Extract links like KBBvK.rtbw
            files = re.findall(r'href="([^"?]+?\.(?:rtbw|rtbz))"', html)
            return sorted(set(files))
    except Exception as e:
        print(f"Error fetching file list: {e}")
        return []


def download_files(files: List[str], dest_dir: str):
    if not os.path.exists(dest_dir):
        os.makedirs(dest_dir)
        print(f"Created directory: {dest_dir}")

    total = len(files)
    print(f"Starting download of {total} files to {dest_dir}...")

    for i, filename in enumerate(files):
        url = BASE_URL + filename
        dest_path = os.path.join(dest_dir, filename)

        if os.path.exists(dest_path):
            # rudimentary check - size would be better but this is a start
            print(f"[{i+1}/{total}] Skipping {filename} (already exists)")
            continue

        print(f"[{i+1}/{total}] Downloading {filename}...", end="\r")
        try:
            urllib.request.urlretrieve(url, dest_path)  # noqa: S310
        except Exception as e:
            print(f"\nError downloading {filename}: {e}")

    print(f"\nFinished downstreaming {total} files.")


def main():
    dest = sys.argv[1] if len(sys.argv) > 1 else DEFAULT_DEST
    files = fetch_file_list()
    if not files:
        print("No files found to download.")
        return

    print(f"Found {len(files)} tablebase files.")
    download_files(files, dest)

    print("\nNext Steps:")
    print(f"1. Run: export SYZYGY_PATH={dest}")
    print("2. Run your verification script.")


if __name__ == "__main__":
    main()
