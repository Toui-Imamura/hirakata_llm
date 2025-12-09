"""Fine-Tuningの実行スクリプト"""
import json
import random
from unsloth import FastLanguageModel
from trl import SFTTrainer
from transformers import TrainingArguments
import torch
from datasets import Dataset
from config import *

FINE_TUNING_PROMPT = """\
あなたは枚方市に関する質問に答えるアシスタントです。
市の公式情報やFAQに基づき、正確かつ丁寧に回答してください。

### 指示:
「{input}」について、{instruction}

### 応答:
{output}"""

# データセットの読み込み（JSONL形式を行単位で）
corpus_data = []
with open(JSON_DATASET_FILE, "r", encoding="utf-8") as f:
    for line in f:
        line = line.strip()
        if not line:
            continue
        corpus_data.append(json.loads(line))

# データサイズ制限
if MAX_DATASET_SIZE < len(corpus_data):
    random.shuffle(corpus_data)
    corpus_data = corpus_data[:MAX_DATASET_SIZE]

print(f"データセット件数: {len(corpus_data)}")

# モデルとトークナイザの読み込み --- (*3)
model, tokenizer = FastLanguageModel.from_pretrained(
    model_name=MODEL_NAME,
    max_seq_length=MAX_SEQ_LENGTH,
    dtype=None,
    load_in_4bit=True,
)
# PEFTのLoRAを適用 --- (*4)
target_modules = ["q_proj", "k_proj", "v_proj", "up_proj", "down_proj", "o_proj", "gate_proj"]
model = FastLanguageModel.get_peft_model(
    model,
    r = 16, # Suggested 8, 16, 32, 64, 128
    target_modules=target_modules,
    lora_alpha = 16,
    lora_dropout = 0,
    bias="none",
    # unslothモデルの場合は勾配チェックポイントを有効化
    use_gradient_checkpointing = "unsloth" in MODEL_NAME,
    use_rslora = False,
    loftq_config = None,
)
# トークナイザーのEOSトークンを確認 --- (*5)
EOS_TOKEN = tokenizer.eos_token
print(f"EOS_TOKEN={EOS_TOKEN}")

# データセットの整形を適用 --- (*6)
def format_example(example):
    """テキストを整形する関数"""
    instruction = example["instruction"].strip()
    input_s = example["input"].strip()
    output_s = example["output"].strip()

    text = FINE_TUNING_PROMPT.format(
        instruction=instruction,
        input=input_s,
        output=output_s
    ) + EOS_TOKEN

    return {"text": text}

corpus_data = list(map(format_example, corpus_data)) # 整形を適用
print(f"データセットのサイズ: {len(corpus_data)}")
dataset = Dataset.from_list(corpus_data) # Dataset形式に変換
print(dataset[:3]) # データセットの最初の3件を表示

# 学習のための設定 --- (*7)
training_args = TrainingArguments(
    output_dir=DIR_MODEL,
    per_device_train_batch_size=2,
    gradient_accumulation_steps=4,
    warmup_steps=5,
    max_steps=MAX_STEPS,
    learning_rate=2e-4,
    optim="adamw_8bit",
    logging_dir="./logs",
    weight_decay=0.01,
    lr_scheduler_type="linear",
    fp16 = not torch.cuda.is_bf16_supported(),
    bf16 = torch.cuda.is_bf16_supported(),
    report_to="none",  # W&Bなどの外部ログを無効化
    save_strategy="no"
)
# 学習を実行 --- (*8)
trainer = SFTTrainer(
    model=model,
    tokenizer=tokenizer,
    train_dataset=dataset,  # 学習に使うデータセットを指定
    dataset_text_field="text",  # データセットのフィールド名を指定
    dataset_num_proc = 2,
    packing = False,  # Packingを無効化
    args=training_args,
)
trainer.train()

# 学習したモデルを保存 --- (*9)
model.save_pretrained(MODEL_OUTPUT_DIR)
tokenizer.save_pretrained(TOKENIZER_DIR)
