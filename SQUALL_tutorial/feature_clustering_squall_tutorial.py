"""Reproducible SQUALL clustering from saved RGB and expression embeddings."""

from __future__ import annotations

import argparse
import importlib.util
import json
from pathlib import Path

import anndata as ad
import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scanpy as sc
import scipy.sparse as sp
import torch
from matplotlib.lines import Line2D
from numba import njit
from PIL import Image
from scipy.spatial import KDTree
from sklearn.cluster import KMeans


def load_config(path: Path):
    spec = importlib.util.spec_from_file_location("squall_tutorial_config", path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot load configuration from {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def load_expression(root: Path) -> ad.AnnData:
    for path in [
        root / "total.h5ads",
        root / "totals.h5ad",
        root / "total.h5ad",
        root / "expression.h5ad",
    ]:
        if path.is_file():
            return sc.read_h5ad(path)

    for path in [
        root / "filtered_feature_bc_matrix.h5",
        root / "raw_feature_bc_matrix.h5",
        root / "expression.h5",
    ]:
        if path.is_file():
            return sc.read_10x_h5(path, gex_only=False)

    if (root / "matrix.mtx").is_file() or (root / "matrix.mtx.gz").is_file():
        return sc.read_10x_mtx(
            root, var_names="gene_symbols", make_unique=False, cache=False
        )
    raise FileNotFoundError(f"No supported expression matrix found in {root}")


def load_positions(root: Path) -> pd.DataFrame:
    directory = root / "spatial"
    paths = [directory / "tissue_positions.csv", directory / "tissue_positions_list.csv"]
    path = next((item for item in paths if item.is_file()), None)
    if path is None:
        matches = [item for item in directory.glob("*.csv") if "tissue" in item.name]
        if not matches:
            raise FileNotFoundError(f"No tissue-position file found in {directory}")
        path = matches[0]

    first = str(pd.read_csv(path, header=None, nrows=1).iloc[0, 0]).lower()
    if first in {"barcode", "barcodes"}:
        frame = pd.read_csv(path)
        if "tl_xn" not in frame and "pxl_row_in_fullres" in frame:
            frame["tl_xn"] = frame["pxl_row_in_fullres"]
        if "tl_yn" not in frame and "pxl_col_in_fullres" in frame:
            frame["tl_yn"] = frame["pxl_col_in_fullres"]
    else:
        frame = pd.read_csv(path, header=None).iloc[:, :6]
        frame.columns = [
            "barcode", "in_tissue", "array_row", "array_col", "tl_xn", "tl_yn"
        ]

    frame["barcode"] = frame["barcode"].astype(str)
    frame[["tl_xn", "tl_yn"]] = frame[["tl_xn", "tl_yn"]].apply(
        pd.to_numeric, errors="coerce"
    )
    return frame.dropna(subset=["tl_xn", "tl_yn"]).drop_duplicates("barcode")


def load_scale_and_image(root: Path):
    directory = root / "spatial"
    scale_path = next(
        (
            path
            for path in [
                directory / "scalefactors_json.json",
                directory / "sub_scalefactors_json.json",
            ]
            if path.is_file()
        ),
        None,
    )
    if scale_path is None:
        raise FileNotFoundError(f"No scale-factor file found in {directory}")
    scale = json.loads(scale_path.read_text())

    image_path = directory / "tissue_hires_image.png"
    if not image_path.is_file():
        images = [
            path
            for path in directory.iterdir()
            if "hires" in path.name.lower()
            and path.suffix.lower() in {".png", ".jpg", ".jpeg", ".tif", ".tiff"}
        ]
        if not images:
            raise FileNotFoundError(f"No hires image found in {directory}")
        image_path = images[0]
    image = np.asarray(Image.open(image_path).convert("RGB"))
    return scale, image


def read_offset(path: Path, sample_id: str):
    values = json.loads(path.read_text())
    key = next(
        (name for name in [sample_id, sample_id.upper(), sample_id.lower()] if name in values),
        None,
    )
    if key is None:
        raise KeyError(f"{sample_id} is not present in {path}")
    value = values[key]
    if isinstance(value, dict):
        x = value.get("x_offset", value.get("x"))
        y = value.get("y_offset", value.get("y"))
    else:
        x, y = value
    return float(x), float(y)


def prepare_spots(data_root: Path, sample_id: str, offset_path: Path):
    sample_root = data_root / sample_id
    adata = load_expression(sample_root)
    adata.obs_names = adata.obs_names.astype(str)
    positions = load_positions(sample_root).set_index("barcode")
    barcodes = adata.obs_names[adata.obs_names.isin(positions.index)]
    adata = adata[barcodes].copy()
    positions = positions.loc[barcodes]
    for column in positions.columns:
        adata.obs[column] = positions[column].to_numpy()

    scale, image = load_scale_and_image(sample_root)
    hires_scale = float(scale["tissue_hires_scalef"])
    diameter = float(scale["spot_diameter_fullres"])
    x_offset, y_offset = read_offset(offset_path, sample_id)

    adata.obs["tl_xn_down"] = (adata.obs["tl_xn"] * hires_scale).astype(int)
    adata.obs["tl_yn_down"] = (adata.obs["tl_yn"] * hires_scale).astype(int)
    adata.obs["xn_org"] = (adata.obs["tl_xn"] * hires_scale - y_offset).astype(int)
    adata.obs["yn_org"] = (adata.obs["tl_yn"] * hires_scale - x_offset).astype(int)
    adata.obs["radius"] = hires_scale * diameter / 2
    return adata, image, (x_offset, y_offset)


def load_embeddings(path: Path, stride: int):
    try:
        payload = torch.load(path, map_location="cpu", weights_only=False)
    except TypeError:
        payload = torch.load(path, map_location="cpu")
    expression = payload["expr_embedding"].detach().float().cpu().numpy()
    rgb = payload["rgb_embedding"].detach().float().cpu().numpy()
    max_x, max_y = payload["dimensions"]
    height = min(int(max_y / stride), expression.shape[0])
    width = min(int(max_x / stride), expression.shape[1])
    return expression[:height, :width], rgb[:height, :width]


def random_points(count: int, seed: int):
    np.random.seed(seed)
    theta = np.random.uniform(0, 2 * np.pi, count)
    radius = np.sqrt(np.random.uniform(0, 1, count))
    return np.column_stack((radius * np.cos(theta), radius * np.sin(theta)))


@njit
def count_points(x, y, x_min, x_max, y_min, y_max):
    counts = np.zeros(x_min.shape[0], dtype=np.int64)
    for point in range(x.shape[0]):
        for unit in range(x_min.shape[0]):
            if (
                x[point] >= x_min[unit]
                and x[point] <= x_max[unit]
                and y[point] >= y_min[unit]
                and y[point] <= y_max[unit]
            ):
                counts[unit] += 1
                break
    return counts


def aggregate(grid, adata: ad.AnnData, stride: int, point_count: int, seed: int):
    centers = np.column_stack(
        (adata.obs["yn_org"].to_numpy(int), adata.obs["xn_org"].to_numpy(int))
    )
    tree = KDTree(centers)
    points = random_points(point_count, seed)
    vectors = []

    for row in range(adata.n_obs):
        center_x, center_y = centers[row]
        radius = float(adata.obs.iloc[row]["radius"])
        half = radius / 2
        indices = tree.query_ball_point(
            [center_x, center_y], r=radius + half * np.sqrt(2)
        )
        if not indices:
            vectors.append(np.zeros(grid.shape[-1]))
            continue

        units = np.asarray(
            [
                (index, x - half, x + half, y - half, y + half)
                for index in indices
                for x, y in [centers[index]]
            ],
            dtype=float,
        )
        if len(units) > 1:
            units = units[:1]
        scaled = points * radius + np.array([center_x, center_y])
        counts = count_points(
            scaled[:, 0], scaled[:, 1], units[:, 1], units[:, 2], units[:, 3], units[:, 4]
        )
        vector = np.zeros(grid.shape[-1])
        for index, count in enumerate(counts):
            if count == 0:
                continue
            xmin, xmax = int(units[index, 1] / stride), int(units[index, 2] / stride)
            ymin, ymax = int(units[index, 3] / stride), int(units[index, 4] / stride)
            xmax, ymax = max(xmin + 1, xmax), max(ymin + 1, ymax)
            vector += grid[ymin:ymax, xmin:xmax].sum(axis=(0, 1)) * (count / point_count)
        vectors.append(vector)
    return sp.csr_matrix(np.nan_to_num(np.asarray(vectors), nan=0.0))


def plot_clusters(adata, labels, image, path: Path):
    clusters = np.sort(np.unique(labels))
    cmap = plt.get_cmap("tab20")
    norm = mcolors.Normalize(vmin=0, vmax=clusters.max())
    figure, axis = plt.subplots(figsize=(10, 9))
    axis.imshow(image, extent=[0, image.shape[1], image.shape[0], 0])
    axis.scatter(
        adata.obs["tl_yn_down"],
        adata.obs["tl_xn_down"],
        c=labels,
        cmap=cmap,
        norm=norm,
        s=adata.obs["radius"],
        alpha=0.8,
        edgecolors="none",
    )
    handles = [
        Line2D(
            [0], [0], marker="o", linestyle="none", markerfacecolor=cmap(norm(cluster)),
            markeredgecolor="none", markersize=10, label=f"c{int(cluster) + 1}"
        )
        for cluster in clusters
    ]
    axis.legend(
        handles=handles, loc="upper center", bbox_to_anchor=(0.5, -0.08),
        ncol=5, frameon=False, handletextpad=0.35, columnspacing=0.9
    )
    axis.set_title("SQUALL combined clustering")
    axis.set_aspect("equal")
    axis.axis("off")
    figure.subplots_adjust(bottom=0.16)
    figure.savefig(path, dpi=500, transparent=True, bbox_inches="tight")
    plt.show()


def run_clustering(
    sample_id: str,
    data_root: Path,
    embedding_path: Path,
    config_path: Path,
    offset_path: Path,
    output_dir: Path,
):
    config = load_config(config_path)
    output_dir.mkdir(parents=True, exist_ok=True)
    adata, image, offset = prepare_spots(data_root, sample_id, offset_path)
    expression, rgb = load_embeddings(embedding_path, int(config.STRIDE))
    expression_spots = aggregate(
        expression, adata, int(config.STRIDE), int(config.N_POINTS), int(config.RANDOM_SEED)
    )
    rgb_spots = aggregate(
        rgb, adata, int(config.STRIDE), int(config.N_POINTS), int(config.RANDOM_SEED)
    )

    combined = ad.AnnData(X=sp.hstack([expression_spots, rgb_spots]))
    sc.pp.pca(
        combined, n_comps=int(config.N_PCA_COMPONENTS), random_state=int(config.RANDOM_SEED)
    )
    sc.pp.neighbors(
        combined, n_neighbors=int(config.N_NEIGHBORS), use_rep="X_pca",
        random_state=int(config.RANDOM_SEED)
    )
    model = KMeans(n_clusters=int(config.N_CLUSTERS), random_state=int(config.RANDOM_SEED))
    labels = model.fit_predict(combined.obsm["X_pca"])

    table = pd.DataFrame({"barcode": adata.obs_names, "cluster": labels})
    csv_path = output_dir / f"{sample_id}_clusters.csv"
    figure_path = output_dir / f"{sample_id}_clusters.pdf"
    table.to_csv(csv_path, index=False)
    plot_clusters(adata, labels, image, figure_path)
    return table, csv_path, figure_path, offset


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--sample-id", required=True)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--embedding", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--offset", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    run_clustering(
        args.sample_id, args.data_root, args.embedding,
        args.config, args.offset, args.output_dir
    )


if __name__ == "__main__":
    main()
