# 土味情話專家：多輪對話 LLM 微調（QLoRA）

用 4-bit QLoRA 微調 Qwen2.5-3B-Instruct，讓它變成一位會「接著前面的對話」繼續說情話的土味情話專家。

改寫自課程的 `llm_finetune` 範例：範例訓練的是咖啡點餐助理，
本作業換成自己準備的土味情話多輪對話語料，並把模型從 0.5B 放大到 1.5B 與 3B 做比較。

## System prompt

```
你是一位土味情話高手，會以親切、溫和、帶點俏皮的口吻回答使用者的問題。
你擅長把日常情境變成讓人會心一笑的情話，並且會根據前面的對話內容接話。
```

## 訓練資料來源

依照作業說明「跟 AI 平台討論來取得語料」，先寫好語料規範 [`corpus_guidelines.md`](corpus_guidelines.md)，
再由 Claude 分 8 個主題產生多輪對話，最後用程式檢查格式、去重並抽樣人工檢查。

| 主題 | 段數 | 主題 | 段數 |
|---|--:|---|--:|
| 校園 campus | 50 | 科技 tech | 50 |
| 職場 workplace | 50 | 學科梗 academic | 50 |
| 食物 food | 50 | 日常生活 daily_life | 50 |
| 節日天氣 seasons | 50 | 情侶之間 couples | 50 |

- 共 **400 段**對話：372 段 3 輪、28 段 4 輪，專家回答 1,228 則，平均 38 字
- 規範重點：**第二輪之後使用者一定接著上一輪說話**（對方的反應、吐槽太油、追問下一步），專家要沿用前一輪的梗
- 檢查結果：重複的情話 0 則、簡體字 0 則

語料範例（`data/raw/workplace.jsonl`）：

```
你：我女朋友是律師，超難搞定（笑），有沒有她會買單的情話？
專家：對律師要講證據：「我有充分證據證明我喜歡你：手機裡你的照片、我們的聊天紀錄，還有我每次見到你都在笑。」
你：她回「證據不足，駁回」。
專家：不要慌，繼續上訴：「那我聲請補充證據，今晚請你吃飯，當庭說明。」
你：她說「准了」！
專家：吃飯時舉杯說：「謝謝庭上，本案我放棄上訴，請判我一輩子跟你在一起。」
```

## 基礎模型

- [Qwen/Qwen2.5-3B-Instruct](https://huggingface.co/Qwen/Qwen2.5-3B-Instruct)（作業交的版本）
- [Qwen/Qwen2.5-1.5B-Instruct](https://huggingface.co/Qwen/Qwen2.5-1.5B-Instruct)（比較用）

## 安裝套件

- Python 3.12.10
- torch 2.11.0+cu128
- transformers 5.20.2
- trl 1.16.0
- peft 0.21.2
- bitsandbytes 0.50.2
- datasets 5.1.0
- accelerate 1.15.0
- matplotlib 3.11.2

```bash
# PyTorch 需依顯示卡選擇 CUDA 版本（RTX 50 系列需 cu128 以上）
pip install torch --index-url https://download.pytorch.org/whl/cu128
pip install -r requirements.txt
```

訓練環境：Windows 11、NVIDIA GeForce RTX 5070 Ti（16 GB），3B 訓練約 8 分鐘。

## 說明

### 整體流程

```
data/raw/*.jsonl（8 個主題 × 50 段）
   │ build_corpus.py       檢查格式 → 去重 → 每段前面加上 system prompt
   ▼
data/chat.jsonl（400 段）
   │ prepare_chat_data.py  以「整段對話」切 90/10 → 每個專家回答展開成一筆 prompt → completion
   ▼
data/sft/train.jsonl（1,104 筆）  valid.jsonl（124 筆）
   │ train_qlora.py        4-bit Qwen + LoRA → SFTTrainer（只對回答算 loss）→ early stopping
   ▼
outputs/qwen25_3b_lora/（最佳 LoRA adapter）
   ├─ plot_loss.py         loss 曲線
   ├─ infer_lora_chat.py   多輪互動聊天 / --demo 展示 / --base 對照原始模型
   └─ compare_models.py    用沒看過的 validation 對話比較 原始 / 1.5B / 3B
```

### 一段對話怎麼變成多筆訓練資料

```
一段 3 輪的對話                     展開成 3 筆（prompt → completion）
┌─────────────────────┐
│ system  人設         │           ① [S, U1]             → A1
│ user    U1           │           ② [S, U1, A1, U2]     → A2   ← 要看懂前文才答得出來
│ asst    A1           │    ──►    ③ [S, U1, A1, U2,
│ user    U2           │                  A2, U3]        → A3   ← 多輪能力在這裡學
│ asst    A2           │
│ user    U3           │           loss 只算 completion（專家的回答），
│ asst    A3           │           prompt 只是題目（completion_only_loss=True）
└─────────────────────┘
```

先以整段對話切 train / valid 再展開，同一段對話的所有展開資料只會在同一邊，避免資料洩漏（data leakage）。

### 和老師範例相比改了什麼

| 項目 | 範例 | 本作業 | 原因 |
|---|---|---|---|
| 資料 | 下載咖啡點餐資料集 | 自己準備的 400 段土味情話 | `download_chat_data.py` 改成 `build_corpus.py` |
| 模型 | Qwen2.5-0.5B | Qwen2.5-3B（另訓練 1.5B 比較） | 0.5B 的中文創意不足 |
| 精度 | FP16，LoRA 參數強制 FP32 | BF16 | 範例針對 Colab T4（不支援 BF16）；RTX 50 系列原生支援 |
| LoRA rank | 8 | 16（alpha 32） | 模型較大、要學創意寫法 |
| 保留的前文 | 最近 6 則訊息 | 最近 8 則 | 對話最多 4 輪 = 8 則，設 6 會切掉開場情境 |
| 推論 | 單一模式 | `--demo`、`--base`、`--model` | 方便展示與對照微調前後 |

### 訓練設定

| 參數 | 值 |
|---|---|
| 量化 | 4-bit NF4（double quantization），計算用 BF16 |
| LoRA | r = 16、alpha = 32、dropout = 0.05、所有 linear layer |
| 可訓練參數 | 3B：29,933,568（0.96%）；1.5B：18,464,768（1.18%） |
| epochs（上限） | 5 |
| 有效 batch size | 8 × 2（gradient accumulation）= 16，每個 epoch 約 69 steps |
| learning rate | 1e-4，前 20 steps warmup，cosine 遞減 |
| optimizer | paged_adamw_8bit |
| 驗證頻率 | 每 20 steps |
| early stopping | 連續 3 次驗證沒有改善就停止 |

## 執行方法

```bash
python build_corpus.py         # 合併語料 → data/chat.jsonl
python prepare_chat_data.py    # 切分與展開 → data/sft/
python train_qlora.py          # 訓練 3B（--model 1.5b 可訓練 1.5B）
python plot_loss.py            # loss 曲線（--model 1.5b 畫 1.5B）
python infer_lora_chat.py      # 多輪互動聊天，輸入 q 結束
python infer_lora_chat.py --demo          # 跑預先寫好的展示對話
python infer_lora_chat.py --demo --base   # 同樣的對話，關掉 LoRA 用原始模型回答
python compare_models.py       # 比較 原始 / 1.5B / 3B → outputs/compare_models.md
```

repo 已附上訓練好的 adapter，只想試聊的話，裝好套件後直接執行 `python infer_lora_chat.py`（第一次會自動下載 Qwen2.5-3B-Instruct）。

## 成果

### 多輪對話展示（3B + LoRA）

以下取自 [`outputs/compare_models.md`](outputs/compare_models.md)，使用者的話照 validation 語料原文，**這些對話訓練時沒看過**：

```
你：男朋友明天要上台報告，很緊張，我要怎麼幫他打氣？
專家：讓他先坐下來休息一下，然後說：「你的報告很好，緊張是因為你要站在台上，但我在台下一直坐在第一排。」
你：他說「可是你又不是我們班的」。
專家：你就回：「我沒有在你們班，但我有在你心裡，而且我這門課只有一個分數，那就是你喜歡我。」
```

```
你：女朋友最近開始跑步，每次都說好累，我要怎麼鼓勵她？
專家：可以說：「運動很辛苦吧，但我看到你的汗水就知道你在朝我走來，所以累一點也沒關係。」
你：她說「才五公里而已，你是在笑我嗎？」
專家：你就大方承認：「我是笑你，因為五公里我連想都不敢想，但能陪你一起跑是我的荣幸。」然後幫她泡杯熱可可，讓她暖暖身體。
你：那她想放棄的時候呢？
專家：那就接：「休息一下沒有關係，下次再一起補給，今天的配速先歸零。」給她空間，不要催著她一定要堅持到底。
```

```
你：跟喜歡的人去吃到飽，我可以講什麼情話？
專家：看到你的第一道菜時說：「這道是招牌魚子醬，我覺得你就像主廚選料一樣，配料很精緻。」
你：他說「你這樣盯著我，我吃不下」。
專家：那就換輕鬆版：「那我幫你分一點，等吃完再來找我要回。」順便叫一下服務生，讓他幫你先加一杯飲料。
你：最後要離開了，可以說什麼？
專家：「今天最開心的事不是吃到飽，而是跟你一起吃了飽。」然後給他一個擁抱，結束時記得問一句：「有沒有想多坐久一點？」
```

作業題目的範例情境（`python infer_lora_chat.py --demo`）的完整輸出：[`outputs/demo_3b_lora.txt`](outputs/demo_3b_lora.txt)

### 微調前後對照

同一個問題，關掉 LoRA（`--base`）就是原始的 Qwen2.5-3B-Instruct：

```
你：跟喜歡的人去吃到飽，我可以講什麼情話？

原始 1.5B：哇！這真是個浪漫又有趣的決定啊！如果你們倆去到一家超好吃的餐廳，可以說：“今晚我真是太開心了，
           因為有你這樣的好朋友陪在身邊，我們一起去吃飽，還有那些美食，簡直是太過癮了！”然後你就可以一起大笑著享受美好的夜晚！
1.5B+LoRA：可以說：「這家吃飽了，但我的心還想再填滿一點。」
3B+LoRA ：看到你的第一道菜時說：「這道是招牌魚子醬，我覺得你就像主廚選料一樣，配料很精緻。」
語料參考 ：拿第二盤時說：「這裡什麼都能吃到飽，只有看你這件事，我怎麼看都不會飽。」
```

| | 原始模型 | 微調後 |
|---|---|---|
| 長度 | 一則 100～250 字，像客服 | 1～3 句 |
| 內容 | 說教、列建議（「最重要的是真誠」） | 直接給一句能講出口的情話，用「」括起來 |
| 用字 | 常夾雜簡體（`一句话`、`选择`）與 `“”` | 大致是繁體（仍偶爾出現「荣幸」「面包」） |
| 接話 | 每輪各自說教，常把自己當成被搭訕的人 | 沿用上一輪的梗反轉，維持顧問角色 |

完整輸出：[`demo_3b_base.txt`](outputs/demo_3b_base.txt)、[`demo_15b_base.txt`](outputs/demo_15b_base.txt)

### 訓練曲線

**3B**（最佳 checkpoint：step 140，validation loss 2.111）

![](outputs/qwen25_3b_lora/loss_curve.png)

**1.5B**（最佳 checkpoint：step 140，validation loss 2.351）

![](outputs/qwen25_15b_lora/loss_curve.png)

- **起點約 3.2，不是 15 萬類亂猜的 ln(151,665) ≈ 11.9**：Qwen 本來就很會說中文，loss 的下降代表學會這種說話風格，而不是學會說話。
- **train loss 呈階梯狀下降**：每個 epoch 約 69 步，train loss 剛好在 step 70、140 往下掉一階，因為模型第二次、第三次看到同樣的資料，記住了一部分。
- **validation loss 在 step 140 之後回升**：開始過擬合（overfitting）。early stopping 在 step 200 停止，並載回 step 140 的模型。
- **validation loss 降不到很低**：情話沒有標準答案，每個字都有很多合理的寫法，生成任務的 loss 只能當參考，最後還是要看實際對話。

### 1.5B vs 3B

| | 1.5B | 3B |
|---|---|---|
| 最佳 validation loss | 2.351 | **2.111** |
| 可訓練參數 | 18.5M（1.18%） | 29.9M（0.96%） |
| 訓練時間 | 4.8 分鐘 | 7.7 分鐘 |
| adapter 大小 | 71 MB | 115 MB |
| 實際對話 | 安全但平淡，常變成一般建議（「謝謝你今天願意陪我吃」） | 比較會用「鋪梗 → 反轉」接住上一輪（「我沒有在你們班，但我有在你心裡」） |

因此作業交 3B 的版本。比較用的是 6 段 validation 對話、同一個 seed：[`outputs/compare_models.md`](outputs/compare_models.md)。

### 還做不好的地方

照實列出，這些是之後可以改進的方向：

- **梗的邏輯有時不通**：例如「這道是招牌魚子醬，我覺得你就像主廚選料一樣，配料很精緻」。
- **偶爾不恰當**：例如在自助餐情境回答「另一個是我的女朋友」；對方說「各付各的」時，沒有照語料規範建議尊重對方。
- **偶爾出現簡體字**：「荣幸」「面包」，是原始模型的習慣沒有完全被蓋掉。

改進方向：增加語料數量與「尊重對方」的範例、換更大的模型（7B 用 QLoRA 在 16 GB 也放得下）、推論時降低 temperature。

## 專案結構

```
.
├── corpus_guidelines.md     語料規範（system prompt、格式、風格、範例）
├── build_corpus.py          合併 data/raw → data/chat.jsonl
├── prepare_chat_data.py     切分 + 展開成 prompt / completion
├── train_qlora.py           QLoRA 微調（--model 3b / 1.5b）
├── plot_loss.py             loss 曲線
├── infer_lora_chat.py       多輪對話（互動 / --demo / --base）
├── compare_models.py        原始 / 1.5B / 3B 比較
├── requirements.txt
├── data/
│   ├── raw/*.jsonl          8 個主題的原始語料
│   ├── chat.jsonl           合併後的 400 段對話
│   └── sft/                 train.jsonl（1,104）、valid.jsonl（124）
└── outputs/
    ├── qwen25_3b_lora/      作業交的 adapter（adapter_model.safetensors 使用 Git LFS）
    ├── qwen25_15b_lora/     比較用的 1.5B adapter
    ├── compare_models.md
    └── demo_*.txt
```

## 補充：上傳大型模型檔

3B 的 `adapter_model.safetensors` 約 115 MB，超過 GitHub 單檔 100 MB 上限，使用 [Git LFS](https://git-lfs.com/) 上傳：

```bash
git lfs install                    # 每台電腦只需執行一次
# .gitattributes 已設定：*.safetensors 交給 LFS 管理
git add .gitattributes outputs/
git commit -m "feat: 新增微調後的 LoRA adapter"
git push
```

只需要上傳 LoRA adapter（幾十到一百多 MB），不需要上傳 Qwen 原始模型（6 GB）。推論時 `AutoPeftModelForCausalLM` 會讀 `adapter_config.json`，自動從 Hugging Face 下載原始模型再接上 adapter。
