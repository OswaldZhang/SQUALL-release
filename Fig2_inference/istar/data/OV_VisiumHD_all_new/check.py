import os
import glob
import numpy as np
import pandas as pd
from pathlib import Path
from scipy.io import mmread

prefix = Path(".")

# =====================================================
# 你要检查的 genes
# =====================================================
query_genes = [
    "ABCC6",
    "APLF",
    "CBS",
    "CBX1",
    "CCT5",
    "DPY19L2",
    "DRG1",
    "ELOB",
    "FST",
    "GTF2A1L",
    "GUCA1A",
    "HS3ST3A1",
    "IGF2",
    "IL6ST",
    "IMPDH1",
    "NCF1",
    "PARK7",
    "PDE4DIP",
    "PDXDC1",
    "PNPT1",
    "POLR2M",
    "PRODH",
    "PTCD3",
    "RACGAP1",
    "RBMS2",
    "SKP1",
    "TARDBP",
    "TDGF1",
    "TJP2",
    "TUBB3",
    "TUBB4A",
    "VPS35",
    "WASHC2A",
    "ZCCHC10",
    "ZFAND5",
    "ZNF395",
]

query_genes = [g.strip() for g in query_genes if g.strip()]
query_set = set(query_genes)

# =====================================================
# helper
# =====================================================
def load_gene_names_from_genes_tsv(path):
    genes_df = pd.read_csv(path, sep="\t", header=None)
    print("[genes.tsv]")
    print("shape:", genes_df.shape)
    print(genes_df.head())

    # 常见格式：
    # 1列: gene_name
    # 2列: gene_id, gene_name
    # 3列: gene_id, gene_name, feature_type
    if genes_df.shape[1] >= 2:
        gene_names = genes_df.iloc[:, 1].astype(str).values
        gene_ids = genes_df.iloc[:, 0].astype(str).values
    else:
        gene_names = genes_df.iloc[:, 0].astype(str).values
        gene_ids = genes_df.iloc[:, 0].astype(str).values

    return genes_df, gene_ids, gene_names


def get_matrix_orientation(X, n_barcodes, n_genes):
    if X.shape == (n_barcodes, n_genes):
        return "barcodes_by_genes"
    elif X.shape == (n_genes, n_barcodes):
        return "genes_by_barcodes"
    else:
        raise ValueError(
            f"cnts.mtx shape {X.shape} does not match "
            f"barcodes={n_barcodes}, genes={n_genes}"
        )


def calc_raw_gene_stats(X, gene_names, gene_ids, barcodes):
    n_barcodes = len(barcodes)
    n_genes = len(gene_names)

    orientation = get_matrix_orientation(X, n_barcodes, n_genes)
    print("\n[cnts.mtx]")
    print("shape:", X.shape)
    print("orientation:", orientation)

    # 转 CSC 方便按 gene 取列；如果矩阵是 gene x barcode，则转 CSR 按行取
    rows = []

    name_to_indices = {}
    for i, g in enumerate(gene_names):
        name_to_indices.setdefault(g, []).append(i)

    for g in query_genes:
        if g not in name_to_indices:
            rows.append({
                "gene": g,
                "in_raw_cnts": False,
                "n_gene_entries_in_genes_tsv": 0,
                "gene_indices": "",
                "gene_ids": "",
                "total_counts": np.nan,
                "n_nonzero_spots": np.nan,
                "pct_nonzero_spots": np.nan,
                "mean_counts_all_spots": np.nan,
                "mean_counts_nonzero_spots": np.nan,
                "max_counts": np.nan,
                "is_all_zero": np.nan,
            })
            continue

        idxs = name_to_indices[g]

        # 如果 genes.tsv 中同名 gene 重复，则把这些 entry 的表达合并
        if orientation == "barcodes_by_genes":
            sub = X[:, idxs]
            vec = np.asarray(sub.sum(axis=1)).ravel()
        else:
            sub = X[idxs, :]
            vec = np.asarray(sub.sum(axis=0)).ravel()

        total = float(vec.sum())
        nnz = int(np.count_nonzero(vec))
        maxv = float(vec.max()) if vec.size > 0 else np.nan
        mean_all = float(vec.mean()) if vec.size > 0 else np.nan
        mean_nz = float(vec[vec > 0].mean()) if nnz > 0 else 0.0

        rows.append({
            "gene": g,
            "in_raw_cnts": True,
            "n_gene_entries_in_genes_tsv": len(idxs),
            "gene_indices": ",".join(map(str, idxs)),
            "gene_ids": ",".join(map(str, gene_ids[idxs])),
            "total_counts": total,
            "n_nonzero_spots": nnz,
            "pct_nonzero_spots": nnz / n_barcodes * 100,
            "mean_counts_all_spots": mean_all,
            "mean_counts_nonzero_spots": mean_nz,
            "max_counts": maxv,
            "is_all_zero": total == 0,
        })

    return pd.DataFrame(rows)


def check_gene_names_txt(path):
    if not path.exists():
        return pd.DataFrame({"gene": query_genes, "in_gene_names_txt": False})

    selected = [x.strip() for x in open(path) if x.strip()]
    selected_set = set(selected)

    return pd.DataFrame({
        "gene": query_genes,
        "in_gene_names_txt": [g in selected_set for g in query_genes],
        "gene_names_txt_line_count": [selected.count(g) for g in query_genes],
    })


def try_check_cnts_super(prefix):
    """
    尝试检查 iStar 的 cnts-super 输出。
    不同版本 iStar 输出格式可能不同：
    1. cnts-super 是目录，里面每个 gene 一个文件
    2. cnts-super 是 tsv/csv
    3. cnts-super 下有若干 txt/tsv/csv 文件
    这个函数尽量自动识别。
    """
    super_path = prefix / "cnts-super"

    base = pd.DataFrame({
        "gene": query_genes,
        "in_cnts_super": np.nan,
        "cnts_super_total": np.nan,
        "cnts_super_nonzero_pixels": np.nan,
        "cnts_super_max": np.nan,
        "cnts_super_note": "",
    })

    if not super_path.exists():
        base["cnts_super_note"] = "cnts-super not found"
        return base

    # 情况1：cnts-super 是目录
    if super_path.is_dir():
        files = sorted(glob.glob(str(super_path / "*")))
        file_names = [Path(f).name for f in files]

        rows = []
        for g in query_genes:
            # 常见情况：文件名包含 gene
            matched = []
            for f in files:
                stem = Path(f).stem
                name = Path(f).name
                if stem == g or name == g or name.startswith(g + "."):
                    matched.append(f)

            if len(matched) == 0:
                rows.append({
                    "gene": g,
                    "in_cnts_super": False,
                    "cnts_super_total": np.nan,
                    "cnts_super_nonzero_pixels": np.nan,
                    "cnts_super_max": np.nan,
                    "cnts_super_note": "no matched file in cnts-super directory",
                })
                continue

            # 尝试读取 matched 文件中的数值
            vals_all = []
            note = []
            for f in matched:
                try:
                    if f.endswith(".npy"):
                        arr = np.load(f)
                        vals = arr.ravel()
                    else:
                        arr = pd.read_csv(f, sep=None, engine="python", header=None)
                        vals = pd.to_numeric(arr.values.ravel(), errors="coerce")
                        vals = vals[~pd.isna(vals)]
                    vals_all.append(np.asarray(vals, dtype=float))
                    note.append(Path(f).name)
                except Exception as e:
                    note.append(f"{Path(f).name}: read failed {repr(e)}")

            if len(vals_all) > 0:
                vals = np.concatenate(vals_all)
                rows.append({
                    "gene": g,
                    "in_cnts_super": True,
                    "cnts_super_total": float(np.nansum(vals)),
                    "cnts_super_nonzero_pixels": int(np.count_nonzero(vals)),
                    "cnts_super_max": float(np.nanmax(vals)) if vals.size else np.nan,
                    "cnts_super_note": ";".join(note),
                })
            else:
                rows.append({
                    "gene": g,
                    "in_cnts_super": True,
                    "cnts_super_total": np.nan,
                    "cnts_super_nonzero_pixels": np.nan,
                    "cnts_super_max": np.nan,
                    "cnts_super_note": ";".join(note),
                })

        return pd.DataFrame(rows)

    # 情况2：cnts-super 是文件
    else:
        try:
            df = pd.read_csv(super_path, sep=None, engine="python", nrows=5)
            cols = list(df.columns)

            # 如果 gene 是列名
            rows = []
            if any(g in cols for g in query_genes):
                full = pd.read_csv(super_path, sep=None, engine="python")
                for g in query_genes:
                    if g in full.columns:
                        vals = pd.to_numeric(full[g], errors="coerce").dropna().values
                        rows.append({
                            "gene": g,
                            "in_cnts_super": True,
                            "cnts_super_total": float(np.sum(vals)),
                            "cnts_super_nonzero_pixels": int(np.count_nonzero(vals)),
                            "cnts_super_max": float(np.max(vals)) if vals.size else np.nan,
                            "cnts_super_note": "gene is column in cnts-super file",
                        })
                    else:
                        rows.append({
                            "gene": g,
                            "in_cnts_super": False,
                            "cnts_super_total": np.nan,
                            "cnts_super_nonzero_pixels": np.nan,
                            "cnts_super_max": np.nan,
                            "cnts_super_note": "gene not column in cnts-super file",
                        })
                return pd.DataFrame(rows)

            base["cnts_super_note"] = "cnts-super file found but gene columns not detected"
            return base

        except Exception as e:
            base["cnts_super_note"] = f"failed to read cnts-super: {repr(e)}"
            return base


# =====================================================
# 1. 检查 raw cnts.mtx
# =====================================================
genes_df, gene_ids, gene_names = load_gene_names_from_genes_tsv(prefix / "genes.tsv")

barcodes = pd.read_csv(prefix / "barcodes.tsv", sep="\t", header=None).iloc[:, 0].astype(str).values
print("\n[barcodes.tsv]")
print("n_barcodes:", len(barcodes))

X = mmread(prefix / "cnts.mtx").tocsr()

raw_stats = calc_raw_gene_stats(X, gene_names, gene_ids, barcodes)

# =====================================================
# 2. 检查 gene-names.txt
# =====================================================
gene_names_stats = check_gene_names_txt(prefix / "gene-names.txt")

# =====================================================
# 3. 尝试检查 cnts-super
# =====================================================
super_stats = try_check_cnts_super(prefix)

# =====================================================
# 4. 合并结果
# =====================================================
out = raw_stats.merge(gene_names_stats, on="gene", how="left")
out = out.merge(super_stats, on="gene", how="left")

# 排序：先显示 raw 中不存在或全0的
out["sort_key"] = (
    (~out["in_raw_cnts"].fillna(False)).astype(int) * 100
    + (out["is_all_zero"].fillna(False)).astype(int) * 10
)
out = out.sort_values(["sort_key", "gene"], ascending=[False, True]).drop(columns=["sort_key"])

# =====================================================
# 5. 保存
# =====================================================
out_path = prefix / "check_query_genes_expression.tsv"
out.to_csv(out_path, sep="\t", index=False)

print("\n===================================================")
print("Summary")
print("===================================================")
print("query genes:", len(query_genes))
print("in raw cnts:", int(out["in_raw_cnts"].fillna(False).sum()))
print("missing from raw cnts:", int((~out["in_raw_cnts"].fillna(False)).sum()))
print("all-zero in raw cnts:", int(out["is_all_zero"].fillna(False).sum()))
print("in gene-names.txt:", int(out["in_gene_names_txt"].fillna(False).sum()))

if "in_cnts_super" in out.columns:
    print("in cnts-super:", int(out["in_cnts_super"].fillna(False).sum()))

print("\nSaved:")
print(out_path)

print("\nFull result:")
pd.set_option("display.max_rows", 200)
pd.set_option("display.max_columns", 50)
pd.set_option("display.width", 200)
print(out.to_string(index=False))
