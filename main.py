"""
此程式用於生成種族歧視相關的數據視覺化圖表。
包含四個主要圖表：
1. 種族歧視重要性評分分佈
2. 政策支持比例
3. 不同領域歧視影響度比較
4. 政策支持度與重要性感受關係
"""

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
from matplotlib import font_manager
from matplotlib.font_manager import FontProperties

# ———— 1. 動態註冊字型 ————
font_path = os.path.join(os.path.dirname(__file__), "GenKiGothic2TW-M.otf")
assert os.path.isfile(font_path), f"找不到字型檔：{font_path}"

# 把字型加到 Matplotlib 的 font manager
font_manager.fontManager.addfont(font_path)
# 再從這支檔案抓它的字型家族名稱
prop = FontProperties(fname=font_path)
family_name = prop.get_name()

# 設為全域預設
mpl.rcParams["font.family"] = family_name
mpl.rcParams["axes.unicode_minus"] = False

# 設定全域圖表樣式
plt.style.use('bmh')  # 使用 matplotlib 內建的 bmh 樣式
mpl.rcParams['figure.facecolor'] = 'white'
mpl.rcParams['axes.facecolor'] = 'white'
mpl.rcParams['axes.grid'] = True
mpl.rcParams['grid.alpha'] = 0.3
mpl.rcParams['axes.labelsize'] = 12
mpl.rcParams['axes.titlesize'] = 14
mpl.rcParams['xtick.labelsize'] = 10
mpl.rcParams['ytick.labelsize'] = 10

# ———— 2. 讀取資料 ————
CSV_PATH = "data.csv"
df = pd.read_csv(CSV_PATH)

# 把「影響度」文字轉成數值
impact_map = {
    "完全無影響": 0,
    "幾乎無影響": 1,
    "有些影響": 2,
    "比較嚴重": 3,
    "非常嚴重": 4
}
df = df.replace(impact_map)
df = df.replace({"不確定": np.nan})


def plot_importance_boxplot():
    """
    繪製種族歧視重要度的盒狀圖，並加入統計資訊
    """
    # 計算基本統計量
    importance_data = df["種族歧視重要度"]
    stats = {
        "平均數": importance_data.mean(),
        "中位數": importance_data.median(),
        "標準差": importance_data.std(),
        "最小值": importance_data.min(),
        "最大值": importance_data.max()
    }

    # 創建圖表
    plt.figure(figsize=(10, 6))

    # 繪製盒狀圖
    box = plt.boxplot(importance_data, vert=False, widths=0.7,
                      patch_artist=True,
                      boxprops=dict(facecolor='lightblue', alpha=0.7),
                      medianprops=dict(color='red', linewidth=2),
                      whiskerprops=dict(color='gray', linewidth=1.5),
                      capprops=dict(color='gray', linewidth=1.5),
                      flierprops=dict(marker='o', markerfacecolor='gray', markersize=8))

    # 設定標題和標籤
    plt.title("種族歧視重要性評分分佈", pad=20, fontsize=14, fontweight='bold')
    plt.xlabel("重要性評分 (1-5)", labelpad=10)
    plt.yticks([])

    # 添加網格
    plt.grid(True, linestyle='--', alpha=0.3)

    # 添加統計資訊文字
    stats_text = "\n".join([f"{k}: {v:.2f}" for k, v in stats.items()])
    plt.text(0.02, 0.98, stats_text,
             transform=plt.gca().transAxes,
             verticalalignment='top',
             bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

    # 設定x軸範圍
    plt.xlim(0, 6)

    # 調整布局並儲存
    plt.tight_layout()
    plt.savefig("output/種族歧視重要性評分分佈.png", dpi=300, bbox_inches="tight", facecolor='white')
    plt.close()


# 執行繪圖函數
plot_importance_boxplot()

# ———— 5. 圖表二：對政府制定更多種族平等政策的支持比例圓餅圖 ————


def categorize_support(score):
    """
    將支持度分數轉換為類別標籤。

    Args:
        score (float): 支持度分數

    Returns:
        str: 支持度類別（支持/不支持/中立）
    """
    if score >= 4:
        return "支持"
    elif score <= 2:
        return "不支持"
    return "中立"


support_cat = df["政府做更多種族歧視相關政策的支持度"].apply(categorize_support)
pie_counts = support_cat.value_counts()
colors = ['#FF9999', '#66B2FF', '#99FF99']  # 自定義顏色
plt.figure(figsize=(8, 8))
plt.pie(pie_counts.values, labels=pie_counts.index, autopct='%1.1f%%',
        colors=colors, explode=[0.05] * len(pie_counts),  # 稍微分開每個扇形
        shadow=True, startangle=90)
plt.title("對政府制定更多種族平等政策的支持比例", pad=20, fontsize=14, fontweight='bold')
plt.axis('equal')
plt.tight_layout()
plt.savefig("output/政策支持比例.png", dpi=300, bbox_inches="tight", facecolor='white')
plt.close()

# ———— 6. 圖表三：不同領域的歧視影響度比較 並列條形圖 ————
fields = ["教育的歧視影響度", "醫療的歧視影響度",
          "公共服務的歧視影響度", "社交的歧視影響度"]
avg_impacts = df[fields].mean()
plt.figure(figsize=(10, 6))
bars = plt.bar(range(len(fields)), avg_impacts.values,
               tick_label=fields, edgecolor='gray',
               color='lightblue', alpha=0.7)
plt.title("不同領域的歧視影響度比較", pad=20, fontsize=14, fontweight='bold')
plt.xlabel("領域", labelpad=10)
plt.ylabel("平均歧視影響度", labelpad=10)
plt.xticks(rotation=15)
# 在柱狀圖上方添加數值標籤
for bar in bars:
    height = bar.get_height()
    plt.text(bar.get_x() + bar.get_width() / 2., height,
             f'{height:.2f}', ha='center', va='bottom')
plt.grid(True, linestyle='--', alpha=0.3)
plt.tight_layout()
plt.savefig("output/不同領域歧視影響度比較.png", dpi=300, bbox_inches="tight", facecolor='white')
plt.close()

# ———— 7. 圖表四：政策支持度與重要性感受之關係 散點圖／氣泡圖 ————
importance = df["種族歧視重要度"]
support = df["政府做更多種族歧視相關政策的支持度"]
bubble_size = df["教育的歧視影響度"] * 20

plt.figure(figsize=(8, 6))
scatter = plt.scatter(importance, support, s=bubble_size,
                      alpha=0.6, edgecolors='white',
                      c=df["教育的歧視影響度"], cmap='viridis')  # 使用顏色映射
plt.colorbar(scatter, label='教育歧視影響度')
plt.title("政策支持度與重要性感受之關係", pad=20, fontsize=14, fontweight='bold')
plt.xlabel("種族歧視重要度", labelpad=10)
plt.ylabel("政策支持度", labelpad=10)
plt.grid(True, linestyle='--', alpha=0.3)
plt.tight_layout()
plt.savefig("output/政策支持度與重要性感受關係.png", dpi=300, bbox_inches="tight", facecolor='white')
plt.close()
