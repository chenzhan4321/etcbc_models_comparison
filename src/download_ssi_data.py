#!/usr/bin/env python3
"""
Download all SSI morphology data versions from GitHub.
Saves to data/raw_ssi_morphology_v2/.
"""
import urllib.request
import os
import sys

OUT_DIR = '<REPO_ROOT>/data/raw_ssi_morphology_v2'
os.makedirs(OUT_DIR, exist_ok=True)

# Files to download
FILES = []
for n in range(2, 10):
    if n == 3:
        FILES += ['s3-in.txt', 's3-out.txt']
    else:
        FILES += [f's{n}-in', f's{n}-out']
FILES += ['t-in_con', 't-in_voc', 't-out']

BASE = 'https://raw.githubusercontent.com/ETCBC/ssi_morphology/master/data/'

print(f"Downloading {len(FILES)} files to {OUT_DIR}...")
for f in FILES:
    url = BASE + f
    out_path = os.path.join(OUT_DIR, f)
    if os.path.exists(out_path):
        size = os.path.getsize(out_path)
        print(f"  ✓ {f} (already downloaded, {size:,} bytes)")
        continue
    try:
        urllib.request.urlretrieve(url, out_path)
        size = os.path.getsize(out_path)
        print(f"  ✓ {f} ({size:,} bytes)")
    except Exception as e:
        print(f"  ✗ {f}: {e}")

# Quick stats
print()
print("=== Summary ===")
total_size = 0
for f in FILES:
    p = os.path.join(OUT_DIR, f)
    if os.path.exists(p):
        s = os.path.getsize(p)
        total_size += s
        with open(p, encoding='utf-8') as fh:
            n_lines = sum(1 for _ in fh)
        print(f"  {f:18s}  {s:>10,} bytes  {n_lines:>6} lines")
print(f"\nTotal: {total_size:,} bytes ({total_size/1024/1024:.1f} MB)")
