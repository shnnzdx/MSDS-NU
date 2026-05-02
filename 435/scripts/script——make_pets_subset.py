#!/usr/bin/env python3
"""
Create a small Oxford Pets subset from Hugging Face Datasets, save images to disk,
and write a CSV manifest (relative_path,label).

Example:
  python make_pets_subset.py

Output:
  pets_subset/
    images/<label>/*.jpg
    manifest.csv
"""

from datasets import load_dataset
from pathlib import Path
import random
import re
import pandas as pd
from tqdm import tqdm


def norm(s: str) -> str:
    """Normalize labels for robust matching (case/spacing/punctuation-insensitive)."""
    return re.sub(r"[^a-z0-9]+", "", s.lower())


def main() -> None:
    random.seed(42)

    ds = load_dataset("enterprise-explorers/oxford-pets")["train"]

    # In this dataset, `label` is a string (e.g., "Birman", "wheaten terrier").
    # Build a normalized lookup to match names robustly.
    all_labels = sorted(set(ds["label"]))
    norm_to_label = {norm(l): l for l in all_labels}

    wanted = [
        "Abyssinian",
        "Bengal",
        "Birman",
        "Egyptian_Mau",
        "Siamese",
        "american_bulldog",
        "english_cocker_spaniel",
        "pomeranian",
    ]

    selected_labels = []
    for w in wanted:
        k = norm(w)
        if k not in norm_to_label:
            print("Not found:", w)
            print("You can choose from these labels (first 30):", all_labels[:30])
            raise SystemExit(1)
        selected_labels.append(norm_to_label[k])

    per_class = 80  # samples per class (small amount is enough to prove concept)

    # Collect samples per class
    by_label = {l: [] for l in selected_labels}
    for ex in ds:
        if ex["label"] in by_label:
            by_label[ex["label"]].append(ex)

    # Sample
    sampled = []
    for lab, items in by_label.items():
        random.shuffle(items)
        sampled.extend(items[:per_class])
    random.shuffle(sampled)

    # Save images + manifest
    out_root = Path("pets_subset")
    img_root = out_root / "images"
    img_root.mkdir(parents=True, exist_ok=True)

    rows = []
    for i, ex in enumerate(tqdm(sampled, desc="Saving images")):
        label = ex["label"]
        safe_label = label.replace(" ", "_")
        (img_root / safe_label).mkdir(parents=True, exist_ok=True)

        fname = f"{i:06d}.jpg"
        rel_path = f"images/{safe_label}/{fname}"
        local_path = out_root / rel_path

        ex["image"].convert("RGB").save(local_path, quality=95)
        rows.append([rel_path, safe_label])

    pd.DataFrame(rows, columns=["relative_path", "label"]).to_csv(
        out_root / "manifest.csv", index=False
    )

    print("✅ Done")
    print("Saved folder:", out_root.resolve())
    print("Manifest:", (out_root / "manifest.csv").resolve())
    print("Labels used:", sorted(set(r[1] for r in rows)))


if __name__ == "__main__":
    main()
