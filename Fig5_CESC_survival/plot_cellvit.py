import os
import json
from PIL import Image, ImageDraw
Image.MAX_IMAGE_PIXELS = None
from tqdm import tqdm

# === 路径设置 ===
tif_dir = "/lustre1/zxzeng/hlxu/OV_TIF"
json_dir = "/lustre1/zxzeng/hlxu/CellViT-plus-plus/OVPannuke_SAM_New"
output_dir = "/lustre1/zxzeng/bwqin/SQUALL_main/clustering/OV/cellvit"
os.makedirs(output_dir, exist_ok=True)

# === 可选：颜色映射（按 type_map）===
type_color_map = {
    "Neoplastic": "red",
    "Inflammatory": "green",
    "Connective": "blue",
    "Epithelial": "orange",
    "Dead": "gray"
}

# === 遍历每个 JSON 文件 ===
for fname in tqdm(os.listdir(json_dir), desc="Processing JSONs"):
    if not fname.endswith("_cells.json"):
        continue

    pid = fname.replace("_he_fnl_cells.json", "")
    if pid not in ["B1587400-1"]:
        continue
    json_path = os.path.join(json_dir, fname)
    tif_path = os.path.join(tif_dir, f"{pid}_he_fnl.tif")
    output_path = os.path.join(output_dir, f"{pid}_cells_overlay.pdf")

    if not os.path.exists(tif_path):
        continue

    # === 读取原始图像 ===
    base_img = Image.open(tif_path).convert("RGB")
    width, height = base_img.size
    downsample = 10
    resized_img = base_img.resize((width // downsample, height // downsample), Image.LANCZOS)
    draw = ImageDraw.Draw(resized_img)

    # === 读取 JSON ===
    with open(json_path, "r") as f:
        data = json.load(f)

    cells = data.get("cells", [])
    type_map = data.get("type_map", {})

    for cell in cells:
        contour = cell.get("contour", [])
        cell_type_id = str(cell.get("type", "0"))
        cell_type_name = type_map.get(cell_type_id, "Unknown")
        color = type_color_map.get(cell_type_name, "black")

        if not contour:
            continue

        # 缩放轮廓坐标
        scaled_contour = [(x // downsample, y // downsample) for x, y in contour]
        if len(scaled_contour) > 1:
            draw.line(scaled_contour + [scaled_contour[0]], fill=color, width=1)

    # === 保存结果图 ===
    resized_img.save(output_path)
