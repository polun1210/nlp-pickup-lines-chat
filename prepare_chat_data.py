"""
prepare_chat_data.py

把 data/chat.jsonl 的完整多輪對話，展開成 TRL SFTTrainer 使用的
prompt / completion 格式，並切成 train / validation。

流程：

    data/chat.jsonl（每行一段完整對話）
        ↓
    以「整段對話」為單位打亂、切 90% train / 10% validation
        ↓
    每段對話裡的每一個 assistant 回答，都展開成一筆訓練資料
        ↓
    data/sft/train.jsonl
    data/sft/valid.jsonl

------------------------------------------------------------
展開是什麼意思？
------------------------------------------------------------

一段 3 輪的對話：

    system  S
    user    U1
    assistant A1
    user    U2
    assistant A2
    user    U3
    assistant A3

會變成 3 筆訓練資料，每一筆都是「看到前文 → 寫出下一句」：

    第 1 筆  prompt: S U1          completion: A1
    第 2 筆  prompt: S U1 A1 U2    completion: A2
    第 3 筆  prompt: S U1 … U3     completion: A3

第 2、3 筆就是在教模型「多輪」：
它必須讀懂前面聊過什麼，才寫得出接得上的 A2、A3。

------------------------------------------------------------
為什麼要先切 train / validation 再展開？
------------------------------------------------------------

如果先展開再切，同一段對話的第 1 筆可能在 train、第 2 筆在 validation。
第 2 筆的 prompt 裡就含有 train 看過的 U1、A1，
等於考前看過題目（data leakage），validation loss 會偏低。
以整段對話為單位切，同一段對話的所有展開資料只會在同一邊。
"""

# =============================================================================
# 1. 匯入套件
# =============================================================================

import json
import random
from pathlib import Path


# =============================================================================
# 2. 設定
# =============================================================================

INPUT_FILE = Path("data/chat.jsonl")

OUTPUT_DIR = Path("data/sft")

VALID_RATIO = 0.1

# prompt 最多保留最近幾則 user / assistant 訊息（system 另外保留）。
#
# 本專案的對話最多 4 輪 = 8 則。
# 預測第 4 輪回答時，前文有 7 則（U1 A1 U2 A2 U3 A3 U4），
# 設 6 會把開場的 U1 切掉，而 U1 常常是整段對話的情境（例如「她是咖啡師」）。
# 設 8 就能保留完整前文。
MAX_CONTEXT_MESSAGES = 8

SEED = 42


# =============================================================================
# 3. 輔助函式
# =============================================================================

def build_examples(conversations):
    """
    把每段對話的每一個 assistant 回答展開成一筆 prompt / completion。

    回傳的每一筆：

        {
            "prompt": [system, user, assistant, ..., user],
            "completion": [assistant]
        }
    """

    examples = []

    for messages in conversations:
        has_system = messages[0]["role"] == "system"
        system = messages[:1] if has_system else []

        for index, message in enumerate(messages):
            if message["role"] != "assistant":
                continue

            context = messages[:index]

            # 只收「user 問 → assistant 答」的標準配對。
            if not context or context[-1]["role"] != "user":
                continue

            history = [
                item for item in context if item["role"] != "system"
            ][-MAX_CONTEXT_MESSAGES:]

            # 截斷後如果開頭是 assistant，代表少了它回應的那個問題，
            # 往後刪到從 user 開始。
            while history and history[0]["role"] != "user":
                history = history[1:]

            examples.append({
                "prompt": system + history,
                "completion": [message],
            })

    return examples


def save_jsonl(rows, path: Path):
    with path.open("w", encoding="utf-8") as file:
        for row in rows:
            file.write(json.dumps(row, ensure_ascii=False) + "\n")


def load_conversations(path: Path):
    conversations = []

    with path.open("r", encoding="utf-8") as file:
        for line_number, line in enumerate(file, start=1):
            if not line.strip():
                continue

            try:
                row = json.loads(line)
            except json.JSONDecodeError as error:
                raise ValueError(
                    f"{path} 第 {line_number} 行不是合法 JSON：{error}"
                ) from error

            if "messages" not in row:
                raise KeyError(f"{path} 第 {line_number} 行缺少 messages 欄位")

            conversations.append(row["messages"])

    if not conversations:
        raise RuntimeError(f"{path} 沒有任何對話")

    return conversations


# =============================================================================
# 4. 主流程
# =============================================================================

def main():

    if not INPUT_FILE.is_file():
        raise FileNotFoundError(
            f"找不到：{INPUT_FILE.resolve()}\n請先執行 python build_corpus.py"
        )

    conversations = load_conversations(INPUT_FILE)

    # -------------------------------------------------------------------------
    # 以整段對話為單位切分
    # -------------------------------------------------------------------------

    shuffled = conversations.copy()
    random.Random(SEED).shuffle(shuffled)

    valid_size = max(1, int(len(shuffled) * VALID_RATIO))
    valid_conversations = shuffled[:valid_size]
    train_conversations = shuffled[valid_size:]

    # -------------------------------------------------------------------------
    # 各自展開
    # -------------------------------------------------------------------------

    train_rows = build_examples(train_conversations)
    valid_rows = build_examples(valid_conversations)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    train_path = OUTPUT_DIR / "train.jsonl"
    valid_path = OUTPUT_DIR / "valid.jsonl"

    save_jsonl(train_rows, train_path)
    save_jsonl(valid_rows, valid_path)

    print("=" * 72)
    print(f"對話總數：{len(conversations)}")
    print(
        f"Train：{len(train_conversations)} 段對話 → "
        f"{len(train_rows)} 筆 prompt/completion"
    )
    print(
        f"Valid：{len(valid_conversations)} 段對話 → "
        f"{len(valid_rows)} 筆 prompt/completion"
    )
    print(f"→ {train_path}")
    print(f"→ {valid_path}")
    print("=" * 72)


if __name__ == "__main__":
    main()
