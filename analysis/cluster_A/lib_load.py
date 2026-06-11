#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Cluster A — shared loaders for the MDLM vs Encoder-only contribution-separation
analysis (reviewer points R2-M3 / R3-1 / R1-W9).

Key facts established by manual verification (see SUMMARY.md):
  * Task = per-character morphological boundary labelling on the S4-on-S2 split.
  * test.in / test.out / every prediction .out file are EXACTLY position aligned:
    one integer label per input character (including the leading space and all
    internal word-boundary spaces, which always carry label 0).
  * All five encoder seeds and all five MDLM seeds share an IDENTICAL test.out
    (md5 17cbfad0339f82a2f5bc94730c092739) and an IDENTICAL patterns.csv
    (md5 4449560e747b03d8322354dc4701f89f) => predictions are directly comparable
    position-by-position with no realignment needed.

This module only reads files; it never writes into the repo.
"""

import os
import csv
from functools import lru_cache

REPO = "/Users/zhanchen/Library/CloudStorage/Dropbox/Projects/etcbc_update"

ENC_DIR = os.path.join(
    REPO, "outputs",
    "encoder_only_100ep_bs128_lr3e-4_d512_l4_h16_d0.25_5seeds", "s4_on_s2",
)
MDLM_DIR = os.path.join(
    REPO, "outputs",
    "mdlm_better_result_after_submission_d768_l10_h6_dr0.23_lr5e-05_steps3_s4-on-s2_5seeds",
)
RAW_DIR = os.path.join(REPO, "data", "raw_s4_on_s2")

# seed -> per-seed directory (resolved from the levenshtein result name for MDLM,
# and from the directory suffix for the encoder).
ENC_SEEDS = {
    "42": "transformer_train_20260127_233759_seed42",
    "49": "transformer_train_20260127_235126_seed49",
    "50": "transformer_train_20260127_235126_seed50",
    "51": "transformer_train_20260127_235146_seed51",
    "52": "transformer_train_20260127_235149_seed52",
}
MDLM_SEEDS = {
    "42": "mdlm_train_20260209_213751",
    "43": "mdlm_train_20260209_213754",
    "46": "mdlm_train_20260209_213822",
    "48": "mdlm_train_20260209_213839",
    "49": "mdlm_train_20260209_213845",
}
SHARED_SEEDS = ["42", "49"]  # seeds present in BOTH model families


def _pred_path(model, seed):
    if model == "enc":
        d = os.path.join(ENC_DIR, ENC_SEEDS[seed], "results")
        # find the predictions .out file
        for f in os.listdir(d):
            if f.startswith("transformer_predictions_") and f.endswith(".out"):
                return os.path.join(d, f)
    else:
        d = os.path.join(MDLM_DIR, MDLM_SEEDS[seed], "results")
        for f in os.listdir(d):
            if f.startswith("mdlm_predictions_") and f.endswith(".out"):
                return os.path.join(d, f)
    raise FileNotFoundError(f"no prediction .out for {model} seed {seed}")


def _restored_path(model, seed, kind):
    """kind in {'original','reduced','final'} -> the restored string file."""
    if model == "enc":
        base = os.path.join(ENC_DIR, ENC_SEEDS[seed], "results", "restore_and_levenshtein")
        prefix = "transformer_predictions_"
    else:
        base = os.path.join(MDLM_DIR, MDLM_SEEDS[seed], "results", "restore_and_levenshtein")
        prefix = "mdlm_predictions_"
    for f in os.listdir(base):
        if f.startswith(prefix) and f.endswith(".out." + kind):
            return os.path.join(base, f)
    raise FileNotFoundError(f"no .out.{kind} for {model} seed {seed}")


@lru_cache(maxsize=None)
def load_labels(model, seed):
    """Return list[list[int]] : per-line list of per-character predicted labels."""
    path = _pred_path(model, seed)
    out = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            out.append([int(x) for x in line.split()])
    return out


@lru_cache(maxsize=None)
def load_gt_labels():
    out = []
    with open(os.path.join(RAW_DIR, "test.out"), encoding="utf-8") as f:
        for line in f:
            out.append([int(x) for x in line.split()])
    return out


@lru_cache(maxsize=None)
def load_test_in():
    """Per-line raw input string (kept verbatim, incl. leading space)."""
    with open(os.path.join(RAW_DIR, "test.in"), encoding="utf-8") as f:
        return [line.rstrip("\n") for line in f]


@lru_cache(maxsize=None)
def load_restored(model, seed, kind):
    with open(_restored_path(model, seed, kind), encoding="utf-8") as f:
        return [line.rstrip("\n") for line in f]


@lru_cache(maxsize=None)
def load_gt_restored(kind):
    with open(os.path.join(RAW_DIR, "test.out." + kind), encoding="utf-8") as f:
        return [line.rstrip("\n") for line in f]


@lru_cache(maxsize=None)
def load_patterns():
    """label(int) -> pattern string ('' for the null/0 label)."""
    m = {}
    with open(os.path.join(RAW_DIR, "patterns.csv"), encoding="utf-8") as f:
        for row in csv.DictReader(f):
            m[int(row["label"])] = row["pattern"]
    return m


@lru_cache(maxsize=None)
def load_pattern_counts():
    """label(int) -> training count (frequency); also a frequency rank."""
    counts = {}
    with open(os.path.join(RAW_DIR, "patterns.csv"), encoding="utf-8") as f:
        for row in csv.DictReader(f):
            counts[int(row["label"])] = int(row["count"])
    # rank 0 = most frequent
    order = sorted(counts, key=lambda k: counts[k], reverse=True)
    rank = {lab: i for i, lab in enumerate(order)}
    return counts, rank


@lru_cache(maxsize=None)
def load_train_forms():
    """Return (train_in_forms set, train_out_word_forms set).

    train.in forms = surface input word strings.
    For 'seen output' we use the per-WORD label tuple appearing in train.out,
    aligned to train.in words, because test.out is a label sequence not a word
    list.  This mirrors accuracy_analyzer's intent (input/output seen) but adapts
    it to the label-sequence representation.
    """
    train_in_forms = set()
    with open(os.path.join(RAW_DIR, "train.in"), encoding="utf-8") as f:
        for line in f:
            train_in_forms.update(line.split())
    return train_in_forms


def split_words(in_line, label_line):
    """Split a position-aligned (chars, labels) line into per-word units.

    Returns list of (word_string, tuple_of_labels) using internal spaces as
    boundaries. Leading space is dropped. Each non-space char keeps its label.
    """
    words = []
    cur_chars = []
    cur_labels = []
    for ch, lab in zip(in_line, label_line):
        if ch == " ":
            if cur_chars:
                words.append(("".join(cur_chars), tuple(cur_labels)))
                cur_chars, cur_labels = [], []
        else:
            cur_chars.append(ch)
            cur_labels.append(lab)
    if cur_chars:
        words.append(("".join(cur_chars), tuple(cur_labels)))
    return words
