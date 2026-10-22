"""
infer_lora_chat.py

載入 Qwen2.5-3B-Instruct + 訓練好的 LoRA adapter，和土味情話專家多輪對話。

三種用法：

    python infer_lora_chat.py            互動模式，自己打字聊天（輸入 q 結束）
    python infer_lora_chat.py --demo     跑一段預先寫好的多輪對話（README 的展示）
    python infer_lora_chat.py --demo --base
                                         同一段對話，但關掉 LoRA，用原本的模型回答
                                         → 對照微調前後的差別

加上 --model 1.5b 可以改用 1.5B 的 adapter。

多輪對話的關鍵：

    messages = [system]
        ↓ 使用者說話
    messages = [system, user1]                       → 模型產生 assistant1
        ↓ 把回答也存進去
    messages = [system, user1, assistant1, user2]    → 模型產生 assistant2
        ↓
    ...

模型本身沒有記憶。每一輪都把「到目前為止的整段對話」重新送進去，
它才看得到前面聊過什麼，這就是多輪對話「記得上文」的原理。
"""

# =============================================================================
# 1. 匯入套件
# =============================================================================

import argparse
from contextlib import nullcontext

import torch

from peft import AutoPeftModelForCausalLM
from transformers import AutoTokenizer

# 和訓練語料用同一份 system prompt，模型才會進入訓練時學到的角色。
from build_corpus import SYSTEM_PROMPT
from train_qlora import DEFAULT_MODEL, MODELS


# =============================================================================
# 2. 推論設定
# =============================================================================

ADAPTER_DIR = str(MODELS[DEFAULT_MODEL][1])

MAX_NEW_TOKENS = 200

# temperature 越低越穩定，越高越有變化；情話需要一點創意，取 0.7。
TEMPERATURE = 0.7
TOP_P = 0.9

# 重複懲罰：降低模型一直重複同一句話的機率。
REPETITION_PENALTY = 1.1

# 保留最近幾輪對話，避免聊太久 token 數無限增加。
MAX_HISTORY_TURNS = 8

SEED = 42

# --demo 使用的多輪對話（第一段是作業題目裡的範例情境）。
DEMO_CONVERSATIONS = [
    [
        "教我如何透過長相來認識女性朋友。",
        "長得像你朋友？",
        "那如果她問「你上一任女朋友呢？」",
        "那如果她問「所以很多人報名嗎？」",
    ],
    [
        "我女朋友說她最近胖了，心情很差，我該怎麼說？",
        "她回我「你騙人，你只是在安慰我」。",
        "好，那我晚上要帶什麼給她吃？",
    ],
    [
        "我在圖書館看到一個很可愛的女生，要怎麼開口？",
        "她說她在準備統計學考試耶。",
        "她笑了，說「你好無聊喔」，接下來怎麼辦？",
    ],
]


# =============================================================================
# 3. 輔助函式
# =============================================================================

def load_model(adapter_dir):
    """
    AutoPeftModelForCausalLM 會讀 adapter_config.json，
    自動下載 / 載入原本的 base model（例如 Qwen2.5-3B-Instruct），再把 LoRA 接上去。

    推論不需要省記憶體到 4-bit，直接用 BF16 載入，速度比較快。
    """

    tokenizer = AutoTokenizer.from_pretrained(adapter_dir)

    model = AutoPeftModelForCausalLM.from_pretrained(
        adapter_dir,
        dtype=torch.bfloat16,
        device_map="auto",
    )
    model.eval()

    return tokenizer, model


def trim_history(messages):
    """
    保留 system + 最近 MAX_HISTORY_TURNS 輪（每輪 user + assistant 兩則）。
    """

    limit = 1 + MAX_HISTORY_TURNS * 2
    if len(messages) <= limit:
        return messages

    return [messages[0]] + messages[-MAX_HISTORY_TURNS * 2:]


def generate_reply(model, tokenizer, messages, use_base):
    """
    把整段對話送進模型，回傳這一輪新產生的回答。

    use_base=True 時，用 disable_adapter() 暫時拔掉 LoRA，
    等於直接問原本的 Qwen2.5-Instruct。
    """

    # 套用 Qwen chat template，最後補上「<|im_start|>assistant」讓模型開始回答。
    inputs = tokenizer.apply_chat_template(
        messages,
        add_generation_prompt=True,
        tokenize=True,
        return_dict=True,
        return_tensors="pt",
    ).to(model.device)

    input_length = inputs["input_ids"].shape[-1]

    adapter_context = model.disable_adapter() if use_base else nullcontext()

    with torch.inference_mode(), adapter_context:
        output = model.generate(
            **inputs,
            max_new_tokens=MAX_NEW_TOKENS,
            do_sample=True,
            temperature=TEMPERATURE,
            top_p=TOP_P,
            repetition_penalty=REPETITION_PENALTY,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id,
        )

    # generate() 回傳「原本的 prompt + 新產生的部分」，只取新產生的。
    new_tokens = output[0][input_length:]

    return tokenizer.decode(new_tokens, skip_special_tokens=True).strip()


def chat_turn(model, tokenizer, messages, user_text, use_base):
    """
    加入使用者的話 → 產生回答 → 把回答也存回對話歷史。
    """

    messages = trim_history(
        messages + [{"role": "user", "content": user_text}]
    )
    reply = generate_reply(model, tokenizer, messages, use_base)

    return messages + [{"role": "assistant", "content": reply}], reply


# =============================================================================
# 4. 兩種模式
# =============================================================================

def run_demo(model, tokenizer, use_base):

    title = "原始模型（未微調）" if use_base else "土味情話專家（LoRA）"
    print("=" * 72)
    print(f"展示模式：{title}")
    print("=" * 72)

    for index, user_turns in enumerate(DEMO_CONVERSATIONS, start=1):
        print(f"\n--- 對話 {index} ---")
        messages = [{"role": "system", "content": SYSTEM_PROMPT}]

        for user_text in user_turns:
            messages, reply = chat_turn(
                model, tokenizer, messages, user_text, use_base
            )
            print(f"你：{user_text}")
            print(f"專家：{reply}")


def run_interactive(model, tokenizer, use_base):

    print("=" * 72)
    print("土味情話專家已上線。輸入 /reset 清除對話、q 結束。")
    print("=" * 72)

    messages = [{"role": "system", "content": SYSTEM_PROMPT}]

    while True:
        user_text = input("\n你：").strip()

        if not user_text:
            continue

        if user_text.lower() in {"q", "quit", "exit"}:
            print("\n對話結束。")
            break

        if user_text == "/reset":
            messages = [{"role": "system", "content": SYSTEM_PROMPT}]
            print("\n（已清除對話歷史）")
            continue

        messages, reply = chat_turn(
            model, tokenizer, messages, user_text, use_base
        )
        print(f"專家：{reply}")


# =============================================================================
# 5. 主流程
# =============================================================================

def main():

    parser = argparse.ArgumentParser(description="土味情話專家多輪對話")
    parser.add_argument(
        "--demo", action="store_true", help="跑預先寫好的多輪對話"
    )
    parser.add_argument(
        "--base", action="store_true", help="關掉 LoRA，用原本的模型回答"
    )
    parser.add_argument(
        "--model", choices=sorted(MODELS), default=DEFAULT_MODEL, help="adapter 的模型大小"
    )
    args = parser.parse_args()

    torch.manual_seed(SEED)

    tokenizer, model = load_model(str(MODELS[args.model][1]))

    if args.demo:
        run_demo(model, tokenizer, args.base)
    else:
        run_interactive(model, tokenizer, args.base)


if __name__ == "__main__":
    main()
