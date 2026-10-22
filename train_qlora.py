"""
train_qlora.py

用 4-bit QLoRA 微調 Qwen2.5-3B-Instruct（也可選 1.5B），讓它變成會多輪接話的土味情話專家。

    python train_qlora.py               # 3B（最終版本）
    python train_qlora.py --model 1.5b  # 1.5B（比較用）

在 RTX 5070 Ti（16 GB）上執行，原生支援 BF16，
不需要 Colab T4 這類舊顯卡常見的 FP16 + FP32 修補。

整體流程：

    data/sft/train.jsonl, valid.jsonl（prompt / completion）
        ↓
    Tokenizer 套用 Qwen chat template，把多輪對話排成一條 token 序列
        ↓
    載入 Qwen2.5-3B-Instruct，權重以 4-bit NF4 儲存（Q = Quantization）
        ↓
    在每個 linear layer 旁邊加上 LoRA adapter（只有 adapter 會被訓練）
        ↓
    SFTTrainer：forward → 只對 completion 算 loss → backward → 更新 LoRA
        ↓
    每 EVAL_STEPS 看一次 validation loss，early stopping 挑最佳 checkpoint
        ↓
    outputs/qwen25_3b_lora/（LoRA adapter + tokenizer，約 120 MB）

------------------------------------------------------------
為什麼是 LoRA，不像 BERT 那樣整個模型一起調？
------------------------------------------------------------

    BERT-base：1 億參數，全部更新，存檔 390 MB。
    Qwen 1.5B ：15 億參數，全部更新需要的 GPU 記憶體遠超過 16 GB
                （權重 + gradient + optimizer 狀態，大約是權重的 4～8 倍）。

LoRA 的做法是凍結原本的權重 W，在旁邊加兩個很小的矩陣 A、B：

    輸出 = W·x + B·A·x
           ↑     ↑
         凍結   只訓練這裡（rank r = 16，參數量不到原本的 2%）

訓練完只要存 A、B（adapter），推論時再和原模型組合起來。

QLoRA 再進一步：把凍結的 W 用 4-bit 存，記憶體又省下約 3/4。
"""

# =============================================================================
# 1. 匯入套件
# =============================================================================

import argparse
from pathlib import Path

import torch
import transformers

from datasets import load_dataset
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    EarlyStoppingCallback,
)
from trl import SFTConfig, SFTTrainer


# =============================================================================
# 2. 模型與路徑設定
# =============================================================================

# Instruct 版本：已經學會「聽指令、照對話格式回答」，
# 我們只需要教它新的說話風格，不用從頭教它怎麼對話。
#
# 兩個大小都訓練過並比較（compare_models.py），最終採用 3B：
#
#     python train_qlora.py              → 3B（預設）
#     python train_qlora.py --model 1.5b → 1.5B
#
# 每個模型的 checkpoint-* 和最終的最佳 adapter 存在各自的 outputs/ 資料夾。
MODELS = {
    "3b": ("Qwen/Qwen2.5-3B-Instruct", Path("outputs/qwen25_3b_lora")),
    "1.5b": ("Qwen/Qwen2.5-1.5B-Instruct", Path("outputs/qwen25_15b_lora")),
}
DEFAULT_MODEL = "3b"

MODEL_ID, OUTPUT_DIR = MODELS[DEFAULT_MODEL]

DATA_DIR = Path("data/sft")
TRAIN_FILE = DATA_DIR / "train.jsonl"
VALID_FILE = DATA_DIR / "valid.jsonl"


# =============================================================================
# 3. 訓練設定
# =============================================================================

SEED = 42

# 一筆資料（system + 最多 4 輪前文 + 回答）大約 200～400 tokens，768 綽綽有餘。
MAX_LENGTH = 768

# 上限 5 個 epochs，實際上通常會被 early stopping 提前停止。
EPOCHS = 5

# LoRA 只訓練很少的參數，常用的 learning rate 比全模型微調大很多
# （BERT 全模型微調常用 3e-5；LoRA 常見 1e-4～2e-4）。
# 語料只有一千多筆，取 1e-4 比較不容易過擬合。
LEARNING_RATE = 1e-4

# 有效 batch size = 8 × 2 = 16。
#
# train 約 1,200 筆 → 每個 epoch 約 75 個 optimizer steps，5 epochs 約 375 steps。
#
# 前 20 steps（約 5%）learning rate 從 0 慢慢升上來，
# 避免剛接上的 LoRA 一開始就大幅改動輸出。
# （transformers 5 已移除 warmup_ratio，只能用 warmup_steps 指定步數。）
WARMUP_STEPS = 20

PER_DEVICE_TRAIN_BATCH_SIZE = 8
PER_DEVICE_EVAL_BATCH_SIZE = 16
GRADIENT_ACCUMULATION_STEPS = 2

# 8-bit 的 AdamW：optimizer 狀態也壓縮，省 GPU 記憶體。
OPTIM = "paged_adamw_8bit"

LOGGING_STEPS = 5

# 每 20 steps 驗證一次，約 1/4 個 epoch，曲線上會有十幾個點。
EVAL_STEPS = 20

CHECKPOINT_LIMIT = 2

# 連續 3 次驗證（60 steps）沒有改善就停止。
EARLY_STOPPING_PATIENCE = 3


# =============================================================================
# 4. LoRA 設定
# =============================================================================

# rank：A、B 矩陣的寬度，越大可學的東西越多，也越容易背答案。
# 常見預設是 8；本專案模型較大、又要學創意寫法，提高到 16。
LORA_R = 16

# 縮放係數，常見設定是 alpha = 2 × r。
LORA_ALPHA = 32

LORA_DROPOUT = 0.05

# 所有 linear layer（attention 的 q/k/v/o 和 MLP 的 gate/up/down）都加 LoRA。
LORA_TARGET_MODULES = "all-linear"


# =============================================================================
# 5. 輔助函式
# =============================================================================

def print_gpu_info():
    if not torch.cuda.is_available():
        raise RuntimeError("找不到 CUDA GPU，QLoRA 需要 NVIDIA GPU。")

    print("=" * 72)
    print(f"PyTorch：{torch.__version__}")
    print(f"Transformers：{transformers.__version__}")
    print(f"GPU：{torch.cuda.get_device_name(0)}")
    print(f"BF16 原生支援：{torch.cuda.is_bf16_supported(including_emulation=False)}")
    print("=" * 72)


def load_quantized_model():
    """
    以 4-bit NF4 載入 base model。

    3B 參數：
        BF16 存 → 3.1B × 2 bytes ≈ 6.2 GB
        4-bit 存 → 3.1B × 0.5 byte ≈ 1.6 GB（再加一點量化用的常數）

    「存」是 4-bit，「算」的時候會暫時還原成 BF16 做矩陣乘法
    （bnb_4bit_compute_dtype），所以精度損失不大。
    """

    quant_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True,
        bnb_4bit_compute_dtype=torch.bfloat16,
    )

    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID,
        quantization_config=quant_config,
        dtype=torch.bfloat16,
        device_map={"": 0},
    )

    # KV cache 是推論時加速逐字生成用的，訓練用不到。
    model.config.use_cache = False

    # 凍結 base model、把 layer norm 等少數層轉成較高精度，
    # 讓量化模型可以穩定地接 LoRA 訓練。
    return prepare_model_for_kbit_training(model)


# =============================================================================
# 6. 主流程
# =============================================================================

def main():

    # -------------------------------------------------------------------------
    # Step 1：GPU 與資料
    # -------------------------------------------------------------------------

    print_gpu_info()

    for path in (TRAIN_FILE, VALID_FILE):
        if not path.is_file():
            raise FileNotFoundError(
                f"找不到：{path.resolve()}\n請先執行 python prepare_chat_data.py"
            )

    data = load_dataset(
        "json",
        data_files={"train": str(TRAIN_FILE), "validation": str(VALID_FILE)},
    )
    print(data)

    # -------------------------------------------------------------------------
    # Step 2：Tokenizer
    # -------------------------------------------------------------------------

    # Qwen 的 tokenizer 內建 chat template，SFTTrainer 會用它把
    #
    #     [{"role": "system", ...}, {"role": "user", ...}, ...]
    #
    # 排成 Qwen 預訓練時看過的格式：
    #
    #     <|im_start|>system\n...<|im_end|>\n<|im_start|>user\n...<|im_end|>\n...
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # -------------------------------------------------------------------------
    # Step 3：4-bit base model + LoRA adapter
    # -------------------------------------------------------------------------

    model = load_quantized_model()

    lora_config = LoraConfig(
        r=LORA_R,
        lora_alpha=LORA_ALPHA,
        lora_dropout=LORA_DROPOUT,
        target_modules=LORA_TARGET_MODULES,
        task_type="CAUSAL_LM",
    )
    model = get_peft_model(model, lora_config)

    # 3B 會印出：trainable params: 29,933,568 || all params: 3,115,872,256 || trainable%: 0.96
    model.print_trainable_parameters()

    # -------------------------------------------------------------------------
    # Step 4：SFT 訓練設定
    # -------------------------------------------------------------------------

    args = SFTConfig(
        output_dir=str(OUTPUT_DIR),
        max_length=MAX_LENGTH,

        num_train_epochs=EPOCHS,
        learning_rate=LEARNING_RATE,
        warmup_steps=WARMUP_STEPS,
        lr_scheduler_type="cosine",
        per_device_train_batch_size=PER_DEVICE_TRAIN_BATCH_SIZE,
        per_device_eval_batch_size=PER_DEVICE_EVAL_BATCH_SIZE,
        gradient_accumulation_steps=GRADIENT_ACCUMULATION_STEPS,
        optim=OPTIM,

        # RTX 50 系列原生支援 BF16，不需要 T4 那種 FP16 + FP32 修補。
        bf16=True,

        logging_steps=LOGGING_STEPS,
        eval_strategy="steps",
        eval_steps=EVAL_STEPS,
        save_strategy="steps",
        save_steps=EVAL_STEPS,
        save_total_limit=CHECKPOINT_LIMIT,

        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        greater_is_better=False,

        # 只對 completion（專家的回答）算 loss。
        #
        # prompt 裡的 system 和使用者的話只當作「題目」，不要求模型學會寫出來；
        # 不然模型會把力氣花在模仿使用者的問法，而不是專家的回答。
        completion_only_loss=True,

        # Qwen chat template 用 <|im_end|> 結束每一則訊息。
        # 模型要學會在回答完時輸出它，推論時才知道什麼時候停。
        eos_token="<|im_end|>",

        report_to="none",
        seed=SEED,
    )

    # -------------------------------------------------------------------------
    # Step 5：訓練
    # -------------------------------------------------------------------------

    trainer = SFTTrainer(
        model=model,
        args=args,
        train_dataset=data["train"],
        eval_dataset=data["validation"],
        processing_class=tokenizer,
        callbacks=[
            EarlyStoppingCallback(early_stopping_patience=EARLY_STOPPING_PATIENCE)
        ],
    )

    trainer.train()

    print(f"最佳 checkpoint：{trainer.state.best_model_checkpoint}")
    print(f"最佳 eval_loss：{trainer.state.best_metric}")

    # -------------------------------------------------------------------------
    # Step 6：儲存最佳 adapter
    # -------------------------------------------------------------------------

    # load_best_model_at_end=True，此時 trainer.model 已經是最佳 checkpoint。
    # PEFT 模型的 save_model 只存 LoRA adapter，不會複製整個 base model。
    trainer.save_model(str(OUTPUT_DIR))
    trainer.save_state()
    tokenizer.save_pretrained(str(OUTPUT_DIR))

    print(f"LoRA adapter：{OUTPUT_DIR.resolve()}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="QLoRA 微調土味情話專家")
    parser.add_argument(
        "--model", choices=sorted(MODELS), default=DEFAULT_MODEL, help="基礎模型大小"
    )
    MODEL_ID, OUTPUT_DIR = MODELS[parser.parse_args().model]
    main()
