"""
plot_loss.py

讀取 train_qlora.py 產生的 trainer_state.json，
把 training loss 與 validation loss 畫在同一張圖，並標出最佳 checkpoint。

    python plot_loss.py               # 3B
    python plot_loss.py --model 1.5b  # 1.5B

    outputs/qwen25_3b_lora/trainer_state.json
        ↓
    outputs/qwen25_3b_lora/loss_curve.png
"""

import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")

import matplotlib.pyplot as plt

from train_qlora import DEFAULT_MODEL, MODELS


# =============================================================================
# 1. 路徑設定（由 --model 決定，預設 3B）
# =============================================================================

MODEL_LABEL = DEFAULT_MODEL

OUTPUT_DIR = MODELS[DEFAULT_MODEL][1]

STATE_FILE = OUTPUT_DIR / "trainer_state.json"

PLOT_FILE = OUTPUT_DIR / "loss_curve.png"


# =============================================================================
# 2. 輔助函式
# =============================================================================

def extract_points(logs, key):
    """
    從 log_history 取出 (step, value)。

    遇到含 train_runtime 的那筆（訓練結束的總結）就停止，
    避免把訓練後額外的紀錄畫進去。
    """

    points = []

    for item in logs:
        if "train_runtime" in item:
            break
        if "step" in item and key in item:
            points.append((item["step"], item[key]))

    return points


def find_best_step(state):
    """
    best_model_checkpoint 例如 "outputs/qwen25_3b_lora/checkpoint-140"，取出 140。
    """

    checkpoint = state.get("best_model_checkpoint")
    if not checkpoint:
        return None

    return int(Path(checkpoint).name.split("-")[-1])


# =============================================================================
# 3. 主流程
# =============================================================================

def main():

    if not STATE_FILE.is_file():
        raise FileNotFoundError(
            f"找不到：{STATE_FILE.resolve()}\n請先執行 python train_qlora.py"
        )

    with STATE_FILE.open("r", encoding="utf-8") as file:
        state = json.load(file)

    logs = state.get("log_history", [])
    if not logs:
        raise RuntimeError("trainer_state.json 中沒有 log_history。")

    train_loss = extract_points(logs, "loss")
    valid_loss = extract_points(logs, "eval_loss")
    best_step = find_best_step(state)

    # -------------------------------------------------------------------------
    # 判讀方式和作業一相同：
    #
    #     兩條線一起下降          → 模型持續學到有用的東西
    #     train 降、valid 回升    → 開始 overfitting（背訓練語料）
    # -------------------------------------------------------------------------

    plt.figure(figsize=(10, 6))

    if train_loss:
        x, y = zip(*train_loss)
        plt.plot(x, y, label="Training Loss", linewidth=1.8)

    if valid_loss:
        x, y = zip(*valid_loss)
        plt.plot(x, y, marker="o", label="Validation Loss", linewidth=1.8)

    if best_step is not None:
        plt.axvline(
            best_step,
            color="gray",
            linestyle="--",
            linewidth=1,
            label=f"Best checkpoint (step {best_step})",
        )

    plt.xlabel("Optimizer Step")
    plt.ylabel("Cross-Entropy Loss (completion tokens)")
    plt.title(f"Qwen2.5-{MODEL_LABEL.upper()} QLoRA Training and Validation Loss")
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(PLOT_FILE, dpi=150, bbox_inches="tight")
    plt.close()

    print(f"Training loss points：{len(train_loss)}")
    print(f"Validation loss points：{len(valid_loss)}")
    print(f"Best step：{best_step}")
    print(f"Loss 圖表：{PLOT_FILE.resolve()}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="畫 training / validation loss")
    parser.add_argument("--model", choices=sorted(MODELS), default=DEFAULT_MODEL)
    MODEL_LABEL = parser.parse_args().model
    OUTPUT_DIR = MODELS[MODEL_LABEL][1]
    STATE_FILE = OUTPUT_DIR / "trainer_state.json"
    PLOT_FILE = OUTPUT_DIR / "loss_curve.png"
    main()
