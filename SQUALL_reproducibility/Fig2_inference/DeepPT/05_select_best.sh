#!/usr/bin/env bash
set -euo pipefail

BASE_DIR="/lustre1/zxzeng/bwqin/STORM_main/clustering/DeepPT"
OUT_ROOT="${BASE_DIR}/outputs"

cd "${BASE_DIR}"

python - <<'PY'
import os
import re
import glob
import json
import h5py
import shutil
import numpy as np
from pathlib import Path

OUT_ROOT = Path("/lustre1/zxzeng/bwqin/STORM_main/clustering/DeepPT/outputs")

jobs = {
    "OV_Xenium_all_new": {
        "fold_pattern": "DeepPT_nested_macenko_OV_to_OVXenium_fold*_*",
        "h5_name": "OV_Xenium_all_new_predicted_expression.h5",
        "ensemble_dir": "DeepPT_nested_macenko_OV_to_OVXenium_ensemble25",
        "best_dir": "DeepPT_nested_macenko_OV_to_OVXenium_best",
    },
    "HCC_Xenium_all_new": {
        "fold_pattern": "DeepPT_nested_macenko_HCC_to_HCCXenium_fold*_*",
        "h5_name": "HCC_Xenium_all_new_predicted_expression.h5",
        "ensemble_dir": "DeepPT_nested_macenko_HCC_to_HCCXenium_ensemble25",
        "best_dir": "DeepPT_nested_macenko_HCC_to_HCCXenium_best",
    },
    "OC_all_new": {
        "fold_pattern": "DeepPT_nested_macenko_OV_fold*_*_predict_OC",
        "h5_name": "OC_all_new_predicted_expression.h5",
        "ensemble_dir": "DeepPT_nested_macenko_OV_predict_OC_ensemble25",
        "best_dir": "DeepPT_nested_macenko_OV_predict_OC_best",
        "score_source_pattern": "DeepPT_nested_macenko_OV_to_OVXenium_fold{ik}_{il}",
    },
    "CC_all_new": {
        "fold_pattern": "DeepPT_nested_macenko_OV_fold*_*_predict_CC",
        "h5_name": "CC_all_new_predicted_expression.h5",
        "ensemble_dir": "DeepPT_nested_macenko_OV_predict_CC_ensemble25",
        "best_dir": "DeepPT_nested_macenko_OV_predict_CC_best",
        "score_source_pattern": "DeepPT_nested_macenko_OV_to_OVXenium_fold{ik}_{il}",
    },
}

def read_string_dataset(ds):
    arr = ds[:]
    out = []
    for x in arr:
        if isinstance(x, bytes):
            out.append(x.decode("utf-8"))
        else:
            out.append(str(x))
    return out

def parse_fold_from_name(name):
    # Works for:
    # DeepPT_nested_macenko_OV_to_OVXenium_fold0_1
    # DeepPT_nested_macenko_OV_fold0_1_predict_OC
    m = re.search(r"fold(\d+)_(\d+)", name)
    if m is None:
        return None
    return int(m.group(1)), int(m.group(2))

def get_internal_score_for_dir(d):
    metrics = Path(d) / "internal_test_metrics.json"
    if not metrics.exists():
        return None

    with open(metrics) as f:
        js = json.load(f)

    return js.get("internal_test_mean_gene_pearson", None)

def get_score_for_prediction_dir(pred_dir, score_source_pattern=None):
    pred_dir = Path(pred_dir)
    fold = parse_fold_from_name(pred_dir.name)
    if fold is None:
        return None

    ik, il = fold

    if score_source_pattern is None:
        source_dir = pred_dir
    else:
        source_dir = OUT_ROOT / score_source_pattern.format(ik=ik, il=il)

    return get_internal_score_for_dir(source_dir)

def make_symlink_or_copy(src, dst):
    src = Path(src).resolve()
    dst = Path(dst)

    if dst.exists() or dst.is_symlink():
        dst.unlink()

    try:
        os.symlink(src, dst)
    except Exception:
        shutil.copy2(src, dst)

for sample, cfg in jobs.items():
    print("\n" + "=" * 100)
    print("[JOB]", sample)

    dirs = sorted(glob.glob(str(OUT_ROOT / cfg["fold_pattern"])))
    dirs = [Path(d) for d in dirs]

    h5_files = []
    for d in dirs:
        fp = d / cfg["h5_name"]
        if fp.exists():
            h5_files.append(fp)

    print("n_prediction_h5:", len(h5_files))

    if len(h5_files) == 0:
        print("[WARN] No prediction h5 found, skip:", sample)
        continue

    # ------------------------------------------------------------
    # 1. Ensemble average
    # ------------------------------------------------------------
    preds_sum = None
    true_expression = None
    tile_coords = None
    genes = None
    attrs = {}

    for i, fp in enumerate(h5_files):
        print(f"[ENSEMBLE READ] {i+1}/{len(h5_files)} {fp}")

        with h5py.File(fp, "r") as f:
            pred = f["predicted_expression"][:].astype(np.float32)

            if preds_sum is None:
                preds_sum = np.zeros_like(pred, dtype=np.float64)

                if "true_expression" in f:
                    true_expression = f["true_expression"][:].astype(np.float32)

                tile_coords = f["tile_coords"][:].astype(np.float32)
                genes = read_string_dataset(f["genes"])
                attrs = dict(f.attrs)
            else:
                if pred.shape != preds_sum.shape:
                    raise ValueError(f"Shape mismatch: {fp}, {pred.shape} vs {preds_sum.shape}")

            preds_sum += pred.astype(np.float64)

    pred_mean = (preds_sum / len(h5_files)).astype(np.float32)

    ensemble_dir = OUT_ROOT / cfg["ensemble_dir"]
    ensemble_dir.mkdir(parents=True, exist_ok=True)

    ensemble_h5 = ensemble_dir / f"{sample}_predicted_expression_ensemble{len(h5_files)}.h5"

    str_dtype = h5py.string_dtype(encoding="utf-8")

    with h5py.File(ensemble_h5, "w") as f:
        f.create_dataset(
            "predicted_expression",
            data=pred_mean,
            compression="gzip",
            chunks=(min(512, pred_mean.shape[0]), min(512, pred_mean.shape[1])),
        )

        if true_expression is not None:
            f.create_dataset(
                "true_expression",
                data=true_expression,
                compression="gzip",
                chunks=(min(512, true_expression.shape[0]), min(512, true_expression.shape[1])),
            )

        f.create_dataset("tile_coords", data=tile_coords, compression="gzip")
        f.create_dataset("genes", data=np.asarray(genes, dtype=object), dtype=str_dtype)

        f.attrs["ensemble_n"] = len(h5_files)
        f.attrs["source"] = "DeepPT nested CV ensemble average"

        for k, v in attrs.items():
            try:
                f.attrs[k] = v
            except Exception:
                pass

    print("[SAVED ENSEMBLE]", ensemble_h5)

    # ------------------------------------------------------------
    # 2. Select best fold by internal_test_mean_gene_pearson
    # ------------------------------------------------------------
    scored = []

    for d in dirs:
        h5_path = d / cfg["h5_name"]
        if not h5_path.exists():
            continue

        score = get_score_for_prediction_dir(
            d,
            score_source_pattern=cfg.get("score_source_pattern", None),
        )

        fold = parse_fold_from_name(d.name)

        if score is None:
            continue

        scored.append({
            "dir": str(d),
            "h5": str(h5_path),
            "score": float(score),
            "fold": fold,
        })

    best_dir = OUT_ROOT / cfg["best_dir"]
    best_dir.mkdir(parents=True, exist_ok=True)

    if len(scored) == 0:
        print("[WARN] No fold score found. Cannot select best for", sample)
        continue

    scored = sorted(scored, key=lambda x: x["score"], reverse=True)
    best = scored[0]

    best_json = best_dir / "best_fold.json"
    with open(best_json, "w") as f:
        json.dump(
            {
                "sample": sample,
                "best": best,
                "all_scored_folds": scored,
            },
            f,
            indent=2,
        )

    best_h5_link = best_dir / f"{sample}_predicted_expression_best.h5"
    make_symlink_or_copy(best["h5"], best_h5_link)

    print("[BEST]", sample)
    print("  fold:", best["fold"])
    print("  score:", best["score"])
    print("  source_h5:", best["h5"])
    print("  best_h5:", best_h5_link)
    print("  best_json:", best_json)

print("\n[DONE] ensemble + best selection")
PY
