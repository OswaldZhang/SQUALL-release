import os
import json
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image, ImageEnhance
from tqdm import tqdm
from matplotlib.colors import LinearSegmentedColormap, Normalize
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.colorbar import ColorbarBase
from scipy.ndimage import gaussian_filter

Image.MAX_IMAGE_PIXELS = None

# ==== 配置路径 ====
survival_json = "/lustre1/zxzeng/bwqin/SQUALL_main/downstream_labels/Survival_five_fold_OS_COX/Survival_TCGA_CESC.json"
expression_base = "/lustre1/zxzeng/bwqin/SQUALL_main/inference/CESC_expression"
svs_base = "/lustre1/zxzeng/bwqin/SQUALL/TCGA_svs_0_5/CESC"
merged_pdf_output = "merged_fast_pdf_select_v6_7_16"
thumb_pdf_output = "merged_fast_pdf_thumbnail_select_v6_7_16"
zoom_output = "zoomed_gene_panels_select_v6_7_16"

os.makedirs(merged_pdf_output, exist_ok=True)
os.makedirs(thumb_pdf_output, exist_ok=True)
os.makedirs(zoom_output, exist_ok=True)

# ==== 设置色图 ====
vortex_cmap = LinearSegmentedColormap.from_list(
    "vortex_cmap", ["#ffffff", "#1c6db7", "#4a8d6e", "#fdb863", "#a91e2c"]#"vortex_cmap", ["#3b0f70", "#1c6db7", "#4a8d6e", "#fdb863", "#a91e2c"]
)
norm = Normalize(vmin=0.01, vmax=1)

# ==== 参数 ====
tile_size = 256
zoom_size = 2000
genes_to_plot = ["CD8A"]

# ==== 加载 survival 信息 ====
with open(survival_json) as f:
    survival_info = json.load(f)

# ==== 主循环 ====
for sample in tqdm(os.listdir(expression_base), desc="Processing"):
    sid = sample.replace("_", "-")
    if sid not in ["TCGA-DS-A1OD","TCGA-DS-A7WF"]:
        continue
    slide_path = os.path.join(svs_base, f"{sid}.tiff")
    if not os.path.exists(slide_path):
        continue

    try:
        base_img = Image.open(slide_path).convert("RGB")
    except:
        continue

    W, H = base_img.size
    panel_images = [base_img]

    for gene in genes_to_plot:
        gene_path = os.path.join(expression_base, sample, f"{gene}.json")
        if not os.path.exists(gene_path):
            panel_images.append(Image.new("RGB", (W, H), (255, 255, 255)))
            continue

        try:
            with open(gene_path) as f:
                gexpr = json.load(f)

            coords, values = [], []
            for tile_name, val in gexpr.items():
                try:
                    x = int(tile_name.split("_")[1])
                    y = int(tile_name.split("_")[3])
                    coords.append((x, y))
                    values.append(val)
                except:
                    continue

            if not values:
                panel_images.append(Image.new("RGB", (W, H), (255, 255, 255)))
                continue

            values = np.array(values)
            norm_values = (values - values.min()) / (values.ptp() + 1e-8)

            x_coords, y_coords = zip(*coords)
            x_min, x_max = min(x_coords), max(x_coords) + tile_size
            y_min, y_max = min(y_coords), max(y_coords) + tile_size
            h_clip, w_clip = y_max - y_min, x_max - x_min

            heatmap = np.zeros((h_clip, w_clip), dtype=np.float32)
            countmap = np.zeros((h_clip, w_clip), dtype=np.uint8)

            for (x, y), v in zip(coords, norm_values):
                x0, y0 = x - x_min, y - y_min
                x1, y1 = x0 + tile_size, y0 + tile_size
                heatmap[y0:y1, x0:x1] += v
                countmap[y0:y1, x0:x1] += 1

            heatmap[countmap > 0] /= countmap[countmap > 0]
            heatmap[countmap == 0] = 0

            # ==== 平滑 + 透明 heatmap ====
            heatmap = gaussian_filter(heatmap, sigma=6)
            colored = vortex_cmap(norm(heatmap))[..., :3]
            rgba = (colored * 255).astype(np.uint8)
            heatmap_img = Image.fromarray(rgba).convert("RGBA")
            alpha = (norm(heatmap) * 255).astype(np.uint8)
            heatmap_img.putalpha(Image.fromarray(alpha))

            # ==== 合成主图 ====
            base_crop = base_img.crop((x_min, y_min, x_max, y_max)).convert("RGBA")
            blended = Image.alpha_composite(base_crop, heatmap_img)
            full_img = Image.new("RGBA", (W, H), (255, 255, 255, 255))
            full_img.paste(blended, (x_min, y_min), blended)
            panel_images.append(full_img.convert("RGB"))
            '''
            # ==== 小图像放大 ====
            top_tiles = sorted(zip(coords, values), key=lambda x: -x[1])
            seen = set()
            fig, axs = plt.subplots(4, 5, figsize=(15, 12))
            axs = axs.flatten()
            n = 0

            for (tx, ty), val in top_tiles:
                if any(abs(tx - sx) < tile_size and abs(ty - sy) < tile_size for sx, sy in seen):
                    continue
                seen.add((tx, ty))

                x0 = max(0, tx - zoom_size // 2)
                y0 = max(0, ty - zoom_size // 2)
                x1 = min(W, x0 + zoom_size)
                y1 = min(H, y0 + zoom_size)

                crop = full_img.crop((x0, y0, x1, y1))
                axs[n].imshow(crop)
                axs[n].set_title(f"{gene} ({tx},{ty})", fontsize=8)
                axs[n].axis("off")
                n += 1
                if n == 20:
                    break

            for i in range(n, 20):
                axs[i].axis("off")

            plt.tight_layout()
            with PdfPages(os.path.join(zoom_output, f"{sid}_{gene}_zoomed.pdf")) as pdf:
                pdf.savefig(fig, dpi=120)
            plt.close(fig)
            '''
        except Exception as e:
            print(f"[Error] {sid} {gene}: {e}")
            panel_images.append(Image.new("RGB", (W, H), (255, 255, 255)))

    # ==== 主图保存 ====
    fig, axs = plt.subplots(1, len(panel_images), figsize=(20, 5))
    titles = ["H&E"] + genes_to_plot
    for i, ax in enumerate(axs):
        ax.imshow(panel_images[i])
        ax.set_title(titles[i] if i < len(titles) else "", fontsize=8)
        ax.axis("off")
        if i == 0:
            scale_bar_length_mm = 1  # 1mm
            microns_per_pixel = 0.5  # 0.5μm/pixel
            pixels_per_mm = int(1000 / microns_per_pixel)  # = 2000 px

            # scale bar 尺寸（单位：图像坐标）
            bar_x = 50  # 离左边 100 px
            bar_height = 50  # 粗细
            bar_y = panel_images[i].height - 30 - bar_height  # 离底部 100 px

            ax.add_patch(plt.Rectangle(
                (bar_x, bar_y), pixels_per_mm, bar_height,
                color='black', linewidth=0
            ))

            ax.text(
                bar_x + pixels_per_mm // 2, bar_y - 20, "1 mm",
                color='black', ha='center', va='bottom', fontsize=8
            )
    cax = fig.add_axes([0.92, 0.25, 0.015, 0.5])
    ColorbarBase(cax, cmap=vortex_cmap, norm=norm).set_label("Normalized Expression", rotation=90)

    plt.tight_layout(rect=[0, 0, 0.9, 1])
    with PdfPages(os.path.join(merged_pdf_output, f"{sid}.pdf")) as pdf:
        pdf.savefig(fig, dpi=300)
    with PdfPages(os.path.join(thumb_pdf_output, f"{sid}_thumb.pdf")) as pdf:
        pdf.savefig(fig, dpi=80)
    plt.close(fig)

