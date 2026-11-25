"""
Download the raw poem CSVs from the chinese-poetry-xiu-ze GitHub repository.

Source: https://github.com/mcjkurz/chinese-poetry-xiu-ze/tree/main/诗歌数据集

This repository stores public-domain classical Chinese poems (Qin through
Qing dynasties, plus modern/contemporary material we don't use here), split
into one CSV per dynasty (or several CSVs for dynasties with many poems).
Each row has five fields: 标题 (title), 朝代 (dynasty), 作者 (author),
体裁 (genre), 内容 (content).

This script only downloads the raw files into data/2-3_regulated_verse/raw/; filtering for
regulated verse (律诗) happens in build_regulated_csv.py.
"""

import argparse
import json
import os
import time
import urllib.error
import urllib.parse
import urllib.request


REPO = "mcjkurz/chinese-poetry-xiu-ze"
REPO_BRANCH = "main"
REPO_DIR = "诗歌数据集"
DEFAULT_OUTPUT_DIR = "data/2-3_regulated_verse/raw"
API_URL = (
    f"https://api.github.com/repos/{REPO}/contents/"
    f"{urllib.parse.quote(REPO_DIR)}?ref={REPO_BRANCH}"
)


def list_repo_files(retries: int = 3, delay: float = 2.0) -> list[dict]:
    """Return the GitHub API directory listing for the poem-data folder."""
    request = urllib.request.Request(
        API_URL,
        headers={"User-Agent": "parallelism-expectation-fetch-data"},
    )
    last_error = None
    for attempt in range(retries):
        try:
            with urllib.request.urlopen(request, timeout=30) as response:
                return json.loads(response.read().decode("utf-8"))
        except (urllib.error.URLError, urllib.error.HTTPError) as error:
            last_error = error
            time.sleep(delay)
    raise RuntimeError(f"Could not list repository contents: {last_error}")


def encode_url(url: str) -> str:
    """Percent-encode a URL's path so non-ASCII filenames can be sent over HTTP."""
    parts = urllib.parse.urlsplit(url)
    safe_path = urllib.parse.quote(parts.path)
    return urllib.parse.urlunsplit(parts._replace(path=safe_path))


def download_file(download_url: str, output_path: str, retries: int = 3, delay: float = 2.0) -> None:
    """Download a single raw CSV file, retrying on transient failures."""
    request = urllib.request.Request(
        encode_url(download_url),
        headers={"User-Agent": "parallelism-expectation-fetch-data"},
    )
    last_error = None
    for attempt in range(retries):
        try:
            with urllib.request.urlopen(request, timeout=60) as response:
                data = response.read()
            with open(output_path, "wb") as f:
                f.write(data)
            return
        except (urllib.error.URLError, urllib.error.HTTPError) as error:
            last_error = error
            time.sleep(delay)
    raise RuntimeError(f"Could not download {download_url}: {last_error}")


def main():
    parser = argparse.ArgumentParser(
        description="Download raw poem CSVs from chinese-poetry-xiu-ze"
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=DEFAULT_OUTPUT_DIR,
        help=f"Directory to save the raw CSVs (default: {DEFAULT_OUTPUT_DIR})",
    )
    parser.add_argument(
        "--skip-existing",
        action="store_true",
        help="Skip files that already exist in the output directory",
    )
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    print(f"Listing files in {REPO}/{REPO_DIR} ...")
    entries = list_repo_files()
    csv_entries = [entry for entry in entries if entry["name"].endswith(".csv")]
    print(f"Found {len(csv_entries)} CSV files")

    for entry in csv_entries:
        output_path = os.path.join(args.output_dir, entry["name"])
        if args.skip_existing and os.path.exists(output_path):
            print(f"  Skipping (exists): {entry['name']}")
            continue
        print(f"  Downloading: {entry['name']} ({entry['size']:,} bytes)")
        download_file(entry["download_url"], output_path)

    print(f"\nDone. Raw CSVs saved to {args.output_dir}/")


if __name__ == "__main__":
    main()
