"""
compare_models.py

用「訓練時沒看過」的 validation 對話，公平比較不同模型的多輪表現。

    data/chat.jsonl 裡被分到 validation 的對話（和 prepare_chat_data.py 同樣的切法）
        ↓ 取前 NUM_CONVERSATIONS 段
    每段對話：使用者的每一輪照語料原文，回答交給模型產生
        ↓ 每個模型用同一個 seed
    outputs/compare_models.md（並列：語料裡的參考答案 / 各模型的回答）

為什麼不只看 loss？

    生成任務沒有標準答案，loss 低代表「比較像語料的寫法」，
    不一定代表「比較好笑、比較通順」。最後還是要看實際產生的對話。
"""

import json
import random
import sys
from pathlib import Path

import torch

import infer_lora_chat as chat
from prepare_chat_data import SEED as SPLIT_SEED, VALID_RATIO, load_conversations


# =============================================================================
# 1. 設定
# =============================================================================

CHAT_FILE = Path("data/chat.jsonl")
OUTPUT_FILE = Path("outputs/compare_models.md")

# (顯示名稱, adapter 資料夾, 是否關掉 LoRA)
MODELS = [
    ("Qwen2.5-1.5B 原始模型", "outputs/qwen25_15b_lora", True),
    ("Qwen2.5-1.5B + LoRA", "outputs/qwen25_15b_lora", False),
    ("Qwen2.5-3B + LoRA", "outputs/qwen25_3b_lora", False),
]

NUM_CONVERSATIONS = 6

SEED = 42


# =============================================================================
# 2. 輔助函式
# =============================================================================

def validation_conversations():
    """
    重現 prepare_chat_data.py 的切分，取出 validation 的對話。
    """

    conversations = load_conversations(CHAT_FILE)
    shuffled = conversations.copy()
    random.Random(SPLIT_SEED).shuffle(shuffled)
    valid_size = max(1, int(len(shuffled) * VALID_RATIO))
    return shuffled[:valid_size][:NUM_CONVERSATIONS]


def run_model(adapter_dir, use_base, conversations):
    """
    回傳 replies[i][j] = 第 i 段對話、第 j 輪的模型回答。
    """

    tokenizer, model = chat.load_model(adapter_dir)
    all_replies = []

    for messages in conversations:
        torch.manual_seed(SEED)
        history = [messages[0]]  # system prompt
        replies = []

        for message in messages[1:]:
            if message["role"] != "user":
                continue
            history, reply = chat.chat_turn(
                model, tokenizer, history, message["content"], use_base
            )
            replies.append(reply)

        all_replies.append(replies)

    del model
    torch.cuda.empty_cache()
    return all_replies


# =============================================================================
# 3. 主流程
# =============================================================================

def main():

    sys.stdout.reconfigure(encoding="utf-8")

    conversations = validation_conversations()
    results = {
        name: run_model(adapter_dir, use_base, conversations)
        for name, adapter_dir, use_base in MODELS
    }

    lines = [
        "# 模型比較：validation 對話（訓練時沒看過）",
        "",
        f"每段對話的使用者發言照語料原文；回答由各模型產生（seed={SEED}，"
        f"temperature={chat.TEMPERATURE}）。",
        "",
    ]

    for i, messages in enumerate(conversations):
        lines.append(f"## 對話 {i + 1}")
        users = [m["content"] for m in messages if m["role"] == "user"]
        references = [m["content"] for m in messages if m["role"] == "assistant"]

        for j, user_text in enumerate(users):
            lines.append(f"\n**你：** {user_text}\n")
            lines.append(f"- 📝 語料參考答案：{references[j]}")
            for name in results:
                lines.append(f"- **{name}**：{results[name][i][j]}")

        lines.append("")

    OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT_FILE.write_text("\n".join(lines), encoding="utf-8")
    print("\n".join(lines))
    print(f"→ {OUTPUT_FILE}")


if __name__ == "__main__":
    main()
