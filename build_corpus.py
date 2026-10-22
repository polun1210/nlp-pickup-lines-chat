"""
build_corpus.py

把 data/raw/ 底下各主題的多輪對話合併成一份 data/chat.jsonl，
並在每段對話最前面加上統一的 system prompt。

土味情話沒有現成的多輪對話資料集，語料是依照 corpus_guidelines.md 自己準備的，
所以這一步不是下載資料，而是「合併 + 檢查」。

    data/raw/campus.jsonl
    data/raw/food.jsonl          每行一段對話，只有 user / assistant
    data/raw/...
        ↓
    檢查格式（角色交錯、從 user 開始、以 assistant 結束、輪數）
        ↓
    移除重複對話（開場問題相同就算重複）
        ↓
    最前面加上 system prompt
        ↓
    data/chat.jsonl

為什麼 system prompt 不直接寫在 raw 資料裡？

    system prompt 只有一份（single source of truth）。
    想調整專家的人設時，只要改這裡的 SYSTEM_PROMPT 再重跑一次，
    不用去改 400 段對話；推論程式 infer_lora_chat.py 也 import 同一份。
"""

# =============================================================================
# 1. 匯入套件
# =============================================================================

import json
from collections import Counter
from pathlib import Path


# =============================================================================
# 2. 設定
# =============================================================================

RAW_DIR = Path("data/raw")

OUTPUT_FILE = Path("data/chat.jsonl")

# 專家的人設。
#
# 訓練時每段對話都以它開頭，
# 推論時也要用同一段，模型才會進入「土味情話高手」的狀態。
SYSTEM_PROMPT = (
    "你是一位土味情話高手，會以親切、溫和、帶點俏皮的口吻回答使用者的問題。"
    "你擅長把日常情境變成讓人會心一笑的情話，並且會根據前面的對話內容接話。"
)

# 至少 2 輪才算多輪對話；語料規範是 3～4 輪。
MIN_TURNS = 2
MAX_TURNS = 6


# =============================================================================
# 3. 輔助函式
# =============================================================================

def validate_messages(messages, source):
    """
    檢查一段對話是否符合格式，不符合就丟出錯誤並指出是哪一行。

    合法的對話：

        user → assistant → user → assistant → ...

    也就是：
        - 從 user 開始
        - user / assistant 交錯
        - 以 assistant 結束（最後一定有專家的回答可以學）
        - content 不是空字串
    """

    if not isinstance(messages, list) or not messages:
        raise ValueError(f"{source}：messages 必須是非空的 list")

    for index, message in enumerate(messages):
        expected_role = "user" if index % 2 == 0 else "assistant"

        if message.get("role") != expected_role:
            raise ValueError(
                f"{source}：第 {index + 1} 則應該是 {expected_role}，"
                f"實際是 {message.get('role')}"
            )

        if not str(message.get("content", "")).strip():
            raise ValueError(f"{source}：第 {index + 1} 則 content 是空的")

    if messages[-1]["role"] != "assistant":
        raise ValueError(f"{source}：最後一則必須是 assistant")

    turns = len(messages) // 2
    if not MIN_TURNS <= turns <= MAX_TURNS:
        raise ValueError(
            f"{source}：輪數 {turns} 不在 {MIN_TURNS}～{MAX_TURNS} 之間"
        )


def load_raw_conversations(raw_dir: Path):
    """
    讀取 raw_dir 底下所有 .jsonl。

    回傳 [(主題名稱, messages), ...]，主題名稱就是檔名（例如 campus）。
    """

    files = sorted(raw_dir.glob("*.jsonl"))
    if not files:
        raise FileNotFoundError(f"{raw_dir.resolve()} 底下沒有任何 .jsonl")

    conversations = []

    for path in files:
        with path.open("r", encoding="utf-8") as file:
            for line_number, line in enumerate(file, start=1):
                if not line.strip():
                    continue

                source = f"{path.name} 第 {line_number} 行"

                try:
                    row = json.loads(line)
                except json.JSONDecodeError as error:
                    raise ValueError(f"{source} 不是合法 JSON：{error}") from error

                messages = row.get("messages")
                validate_messages(messages, source)

                conversations.append((path.stem, messages))

    return conversations


def remove_duplicates(conversations):
    """
    開場問題（第一則 user）完全相同的對話只保留第一段。

    重複的對話如果同時被分到 train 和 valid，validation loss 會被灌水。
    """

    seen = set()
    unique = []

    for topic, messages in conversations:
        key = messages[0]["content"].strip()
        if key in seen:
            continue
        seen.add(key)
        unique.append((topic, messages))

    removed = len(conversations) - len(unique)
    if removed:
        print(f"已移除重複對話：{removed} 段")

    return unique


# =============================================================================
# 4. 主流程
# =============================================================================

def main():

    conversations = load_raw_conversations(RAW_DIR)
    conversations = remove_duplicates(conversations)

    OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)

    with OUTPUT_FILE.open("w", encoding="utf-8") as file:
        for _, messages in conversations:
            row = {
                "messages": [
                    {"role": "system", "content": SYSTEM_PROMPT},
                    *messages,
                ]
            }
            file.write(json.dumps(row, ensure_ascii=False) + "\n")

    # -------------------------------------------------------------------------
    # 統計資訊
    # -------------------------------------------------------------------------

    topic_counts = Counter(topic for topic, _ in conversations)
    turn_counts = Counter(len(messages) // 2 for _, messages in conversations)

    print("=" * 72)
    print(f"對話總數：{len(conversations)} 段")
    print("各主題：")
    for topic, count in sorted(topic_counts.items()):
        print(f"  {topic:<12} {count:>4} 段")
    print("輪數分布：")
    for turns, count in sorted(turn_counts.items()):
        print(f"  {turns} 輪：{count:>4} 段")
    print(f"輸出：{OUTPUT_FILE}")
    print("=" * 72)


if __name__ == "__main__":
    main()
