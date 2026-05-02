#!/usr/bin/env python3
"""
Parse AutoML batch prediction JSONL outputs, join with ground-truth labels,
export a top-3 predictions CSV, and compute basic metrics + confusion matrix.

Defaults match the original snippet:
  - Ground truth CSV: vertex_import.csv
  - Batch outputs directory: automl_batch_out/
  - Outputs:
      automl_predictions_top3.csv
      automl_confusion_matrix_counts.csv

Example:
  python evaluate_automl_batch_outputs.py

Optional arguments:
  python evaluate_automl_batch_outputs.py \
    --gt-csv vertex_import.csv \
    --out-dir automl_batch_out \
    --preds-csv automl_predictions_top3.csv \
    --cm-csv automl_confusion_matrix_counts.csv
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

import pandas as pd
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix


def _as_list(x: Any) -> List[Any]:
    if x is None:
        return []
    if isinstance(x, list):
        return x
    if isinstance(x, dict):
        return list(x.values())
    return [x]


def extract_uri(rec: Dict[str, Any]) -> Optional[str]:
    """Try common fields used to store the instance URI."""
    inst = rec.get("instance") or {}
    if isinstance(inst, list) and inst:
        inst = inst[0]
    if not isinstance(inst, dict):
        return None
    return inst.get("content") or inst.get("gcsUri") or inst.get("uri") or inst.get("gcs_uri")


def extract_topk(rec: Dict[str, Any], k: int = 3) -> List[Tuple[Optional[str], Optional[float]]]:
    """Extract top-k (name, confidence) pairs from a prediction record."""
    pred = rec.get("prediction")
    if pred is None:
        pred = rec.get("predictions")
    if pred is None:
        return []

    # Sometimes predictions is a list of dicts
    if isinstance(pred, list):
        pred = pred[0] if pred else {}
    if not isinstance(pred, dict):
        return []

    names = pred.get("displayNames") or pred.get("display_names") or pred.get("classes") or []
    confs = pred.get("confidences") or pred.get("confidence") or pred.get("scores") or []

    names_list = _as_list(names)
    confs_list = _as_list(confs)

    # If we have confidences, zip names+confs; otherwise keep names with None conf.
    pairs: List[Tuple[Optional[str], Optional[float]]] = []
    if names_list and confs_list:
        for n, c in zip(names_list, confs_list):
            try:
                c_f = float(c) if c is not None else None
            except (TypeError, ValueError):
                c_f = None
            pairs.append((str(n) if n is not None else None, c_f))
    elif names_list:
        pairs = [(str(n) if n is not None else None, None) for n in names_list]

    return pairs[:k]


def load_ground_truth(gt_csv: Path) -> Dict[str, str]:
    """Read ground truth mapping from a 2-column CSV with no header."""
    gt = pd.read_csv(gt_csv, header=None, names=["gcs_uri", "true_label"])
    return dict(zip(gt["gcs_uri"], gt["true_label"]))


def iter_jsonl_files(out_dir: Path) -> Iterable[Path]:
    return out_dir.rglob("*.jsonl")


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate AutoML batch JSONL outputs.")
    parser.add_argument("--gt-csv", default="vertex_import.csv", help="Ground truth CSV (default: vertex_import.csv)")
    parser.add_argument("--out-dir", default="automl_batch_out", help="Directory containing JSONL outputs")
    parser.add_argument("--preds-csv", default="automl_predictions_top3.csv", help="Output CSV for top-3 predictions")
    parser.add_argument("--cm-csv", default="automl_confusion_matrix_counts.csv", help="Output CSV for confusion matrix counts")
    args = parser.parse_args()

    gt_csv = Path(args.gt_csv)
    out_dir = Path(args.out_dir)
    preds_csv = Path(args.preds_csv)
    cm_csv = Path(args.cm_csv)

    # 1) Read ground truth (vertex_import.csv)
    gt_map = load_ground_truth(gt_csv)

    # 2) Find all batch output JSONL files
    jsonl_files = list(iter_jsonl_files(out_dir))
    assert jsonl_files, f"No JSONL output files found under: {out_dir}"

    rows: List[Dict[str, Any]] = []

    for fp in jsonl_files:
        with fp.open("r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    rec = json.loads(line)
                except json.JSONDecodeError:
                    continue

                uri = extract_uri(rec)
                if not uri:
                    continue

                true_label = gt_map.get(uri)

                top = extract_topk(rec, k=3)

                row = {
                    "gcs_uri": uri,
                    "true_label": true_label,
                    "pred1": top[0][0] if len(top) > 0 else None,
                    "conf1": top[0][1] if len(top) > 0 else None,
                    "pred2": top[1][0] if len(top) > 1 else None,
                    "conf2": top[1][1] if len(top) > 1 else None,
                    "pred3": top[2][0] if len(top) > 2 else None,
                    "conf3": top[2][1] if len(top) > 2 else None,
                }
                rows.append(row)

    df = pd.DataFrame(rows).drop_duplicates(subset=["gcs_uri"])
    df["correct"] = df["pred1"] == df["true_label"]
    df.to_csv(preds_csv, index=False)

    # 3) Metrics (only rows that have both ground truth and a top-1 prediction)
    df_eval = df.dropna(subset=["true_label", "pred1"])
    y_true = df_eval["true_label"].tolist()
    y_pred = df_eval["pred1"].tolist()

    acc = accuracy_score(y_true, y_pred) if y_true else 0.0
    macro_f1 = f1_score(y_true, y_pred, average="macro") if y_true else 0.0

    labels = sorted(set(y_true))
    cm = confusion_matrix(y_true, y_pred, labels=labels) if labels else []

    print("AutoML top-1 accuracy:", round(acc, 4))
    print("AutoML macro-F1     :", round(macro_f1, 4))

    if labels:
        pd.DataFrame(cm, index=labels, columns=labels).to_csv(cm_csv)
        print(f"Saved: {preds_csv} and {cm_csv}")
    else:
        print(f"Saved: {preds_csv} (no labels found to compute confusion matrix)")


if __name__ == "__main__":
    main()
