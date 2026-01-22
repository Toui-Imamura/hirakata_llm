"""Fine-Tuningの実行スクリプト"""
import os
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"  # CUDAメモリの断片化を抑制

import json
import random
from unsloth import FastLanguageModel
from trl import SFTTrainer
from transformers import TrainingArguments, TrainerCallback
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
with open("FAQdataset.json", "r", encoding="utf-8") as f:
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
    r = 32, # Suggested 8, 16, 32, 64, 128
    target_modules=target_modules,
    lora_alpha = 32,
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

    if input_s:
        text = FINE_TUNING_PROMPT.format(
            instruction=instruction,
            input=input_s,
            output=output_s
        ) + EOS_TOKEN
    else:
        # Prompt without input section
        no_input_prompt = """\
あなたは枚方市に関する質問に答えるアシスタントです。
市の公式情報やFAQに基づき、正確かつ丁寧に回答してください。

### 指示:
{instruction}

### 応答:
{output}"""
        text = no_input_prompt.format(
            instruction=instruction,
            output=output_s
        ) + EOS_TOKEN

    return {"text": text}

corpus_data = list(map(format_example, corpus_data)) # 整形を適用
print(f"データセットのサイズ: {len(corpus_data)}")
dataset = Dataset.from_list(corpus_data) # Dataset形式に変換

# データセットを訓練用と検証用に分割 --- (*6.5)
split_dataset = dataset.train_test_split(test_size=0.2, seed=42)
train_dataset = split_dataset["train"]
eval_dataset = split_dataset["test"]
print(f"訓練用データセット: {len(train_dataset)}件")
print(f"検証用データセット: {len(eval_dataset)}件")
print(train_dataset[:3]) # データセットの最初の3件を表示

# 学習のための設定 --- (*7)
training_args = TrainingArguments(
    output_dir=DIR_MODEL,
    per_device_train_batch_size=8,  # メモリ節約のため16から8に削減
    gradient_accumulation_steps=8,  # 実質的なバッチサイズ = 8 * 8 = 64 (元の16 * 4 = 64と同じ)
    warmup_steps=5,
    max_steps=MAX_STEPS,
    learning_rate=2e-4,
    optim="adamw_8bit",
    logging_dir="./logs",
    logging_steps=10,  # 10ステップごとにログを出力
    weight_decay=0.01,
    lr_scheduler_type="linear",
    fp16 = not torch.cuda.is_bf16_supported(),
    bf16 = torch.cuda.is_bf16_supported(),
    report_to="none",  # W&Bなどの外部ログを無効化
    save_strategy="no",
    eval_strategy="steps",  # 検証を定期的に実行
    eval_steps=100,  # 100ステップごとに検証を実行
    save_total_limit=1,  # 最良のモデルのみ保存
    per_device_eval_batch_size=4,  # 検証時のバッチサイズも削減
    gradient_checkpointing=True,  # メモリ効率を改善
    log_level="info",  # ログレベルを設定
)
# 学習を実行 --- (*8)
trainer = SFTTrainer(
    model=model,
    tokenizer=tokenizer,
    train_dataset=train_dataset,  # 訓練用データセットを指定
    eval_dataset=eval_dataset,  # 検証用データセットを指定
    dataset_text_field="text",  # データセットのフィールド名を指定
    dataset_num_proc = 2,
    packing = False,  # Packingを無効化
    args=training_args,
)
trainer.train()

# 学習したモデルを保存 --- (*9)
model.save_pretrained(DIR_MODEL)
tokenizer.save_pretrained(DIR_MODEL)
