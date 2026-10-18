from pathlib import Path
from time import perf_counter
import csv
import re

import cv2
import numpy as np
import matplotlib.pyplot as plt


# 1. 設定資料夾
BASE_DIR = Path(__file__).resolve().parent
image_dir = BASE_DIR / "images"
output_dir = BASE_DIR / "results"
output_dir.mkdir(exist_ok=True)


# 2. 按檔名中的數字排序，避免 1、10、2 的順序問題
def natural_key(path):
    return [
        int(part) if part.isdigit() else part.lower()
        for part in re.split(r"(\d+)", path.name)
    ]


extensions = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}

image_paths = sorted(
    [
        path
        for path in image_dir.iterdir()
        if path.is_file() and path.suffix.lower() in extensions
    ],
    key=natural_key
)

if not image_paths:
    raise ValueError("images 資料夾裡沒有找到照片")

print(f"找到 {len(image_paths)} 張照片")


# 3. 定義兩種清晰度計算方法
def sobel_score(gray):
    gx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
    gy = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)

    return float(np.mean(gx ** 2 + gy ** 2))


def laplacian_score(gray):
    lap = cv2.Laplacian(gray, cv2.CV_64F, ksize=3)

    return float(np.var(lap))


# 先暖機，再重複計時，降低單次計時的波動
def measure_method(method, gray, repeats=10):
    score = method(gray)
    times = []

    for _ in range(repeats):
        start = perf_counter()
        method(gray)
        elapsed_ms = (perf_counter() - start) * 1000
        times.append(elapsed_ms)

    return score, float(np.median(times))


# 4. 分析所有照片
results = []
expected_shape = None

for index, path in enumerate(image_paths, start=1):
    file_data = np.fromfile(str(path), dtype=np.uint8)
    image = cv2.imdecode(file_data, cv2.IMREAD_COLOR)

    if image is None:
        raise ValueError(f"無法讀取照片：{path.name}")

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # 確認所有照片尺寸相同，讓比較條件一致
    if expected_shape is None:
        expected_shape = gray.shape
    elif gray.shape != expected_shape:
        raise ValueError(f"照片尺寸不一致：{path.name}")

    s_score, s_time = measure_method(sobel_score, gray)
    l_score, l_time = measure_method(laplacian_score, gray)

    results.append({
        "index": index,
        "filename": path.name,
        "sobel_score": s_score,
        "laplacian_score": l_score,
        "sobel_ms": s_time,
        "laplacian_ms": l_time
    })

    print(
        f"{index:02d}. {path.name} | "
        f"Sobel: {s_score:.2f} | "
        f"Laplacian: {l_score:.2f}"
    )


# 5. 儲存 CSV 數據表
csv_path = output_dir / "focus_scores.csv"

with csv_path.open("w", newline="", encoding="utf-8-sig") as file:
    writer = csv.DictWriter(
        file,
        fieldnames=list(results[0].keys())
    )
    writer.writeheader()
    writer.writerows(results)


# 6. 分別找出最高分照片
best_sobel = max(results, key=lambda row: row["sobel_score"])
best_laplacian = max(
    results,
    key=lambda row: row["laplacian_score"]
)

print("\n最佳結果")
print(
    f"Sobel：{best_sobel['filename']}，"
    f"分數 {best_sobel['sobel_score']:.2f}"
)
print(
    f"Laplacian：{best_laplacian['filename']}，"
    f"分數 {best_laplacian['laplacian_score']:.2f}"
)


# 7. 儲存最佳照片，保留原始檔案內容
for method_name, best in [
    ("sobel", best_sobel),
    ("laplacian", best_laplacian)
]:
    source_path = image_dir / best["filename"]
    destination = output_dir / f"best_{method_name}_{source_path.name}"
    destination.write_bytes(source_path.read_bytes())


# 8. 畫出兩種清晰度曲線
indices = [row["index"] for row in results]
filenames = [row["filename"] for row in results]

fig, axes = plt.subplots(2, 1, figsize=(12, 9))

plot_settings = [
    (axes[0], "sobel_score", "Sobel", best_sobel),
    (axes[1], "laplacian_score", "Laplacian", best_laplacian)
]

for ax, score_key, method_name, best in plot_settings:
    scores = [row[score_key] for row in results]

    ax.plot(indices, scores, marker="o")
    ax.scatter(
        best["index"],
        best[score_key],
        color="red",
        s=80,
        zorder=3,
        label=f"Best: {best['filename']}"
    )

    ax.set_title(f"{method_name} Focus Scores")
    ax.set_xlabel("Image filename (sorted)")
    ax.set_ylabel("Focus score")
    ax.set_xticks(indices)
    ax.set_xticklabels(filenames, rotation=45, ha="right")
    ax.grid(True, alpha=0.3)
    ax.legend()

fig.tight_layout()
fig.savefig(
    output_dir / "focus_curves.png",
    dpi=150,
    bbox_inches="tight"
)


# 9. 儲存結果摘要
mean_sobel_ms = np.mean([row["sobel_ms"] for row in results])
mean_laplacian_ms = np.mean([
    row["laplacian_ms"] for row in results
])

summary = (
    f"Image count: {len(results)}\n"
    f"Image size (height, width): {expected_shape}\n"
    f"Sobel best: {best_sobel['filename']}\n"
    f"Sobel best score: {best_sobel['sobel_score']:.6f}\n"
    f"Laplacian best: {best_laplacian['filename']}\n"
    f"Laplacian best score: {best_laplacian['laplacian_score']:.6f}\n"
    f"Sobel mean processing time: {mean_sobel_ms:.3f} ms/image\n"
    f"Laplacian mean processing time: "
    f"{mean_laplacian_ms:.3f} ms/image\n"
    "Timing: median of 10 runs per image; "
    "excludes reading, grayscale conversion, and saving.\n"
)

(output_dir / "summary.txt").write_text(summary, encoding="utf-8")

print(f"\n全部結果已儲存至：{output_dir}")

# 10. 輸出每張照片的原圖與邊緣比較圖
edge_dir = output_dir / "edge_comparisons"
edge_dir.mkdir(exist_ok=True)

for path, row in zip(image_paths, results):
    # 重新讀取照片
    file_data = np.fromfile(str(path), dtype=np.uint8)
    image = cv2.imdecode(file_data, cv2.IMREAD_COLOR)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Sobel 邊緣
    gx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
    gy = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
    magnitude = np.sqrt(gx ** 2 + gy ** 2)

    # Laplacian 邊緣
    lap = cv2.Laplacian(gray, cv2.CV_64F, ksize=3)

    # 使用固定顯示範圍，方便比較不同照片
    # 灰階值為 0～255、ksize=3，以下範圍涵蓋運算結果
    sobel_display_max = np.sqrt(2) * 4 * 255
    laplacian_display_max = 8 * 255

    edge_fig, edge_axes = plt.subplots(1, 3, figsize=(15, 5))

    edge_axes[0].imshow(
        cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    )
    edge_axes[0].set_title("Original")

    edge_axes[1].imshow(
    magnitude,
    cmap="gray"
    )
    
    edge_axes[1].set_title(
        f"Sobel | Score: {row['sobel_score']:.2f}"
    )

    edge_axes[2].imshow(
    np.abs(lap),
    cmap="gray"
    )
    
    edge_axes[2].set_title(
        f"Laplacian | Score: {row['laplacian_score']:.2f}"
    )

    for ax in edge_axes:
        ax.axis("off")

    edge_fig.suptitle(path.name)
    edge_fig.tight_layout()

    edge_fig.savefig(
        edge_dir / f"{path.name}_comparison.png",
        dpi=150,
        bbox_inches="tight"
    )

    # 儲存後關閉，避免同時開啟大量圖形
    plt.close(edge_fig)

print("邊緣比較圖已儲存至：", edge_dir)

plt.show()