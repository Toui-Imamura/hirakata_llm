"""Fine-Tuningの実行スクリプト"""
import os
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"  # CUDAメモリの断片化を抑制

import numpy as np # 計算用に必要
import pandas as pd
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

# --- 追加コードここから ---

def preprocess_logits_for_metrics(logits, labels):
    """
    評価時にGPUメモリを節約するため、全確率(float)ではなく
    予測したトークンID(int)だけをCPUに送る関数
    """
    torch.cuda.empty_cache()

    if isinstance(logits, tuple):
        # モデルによってはタプルで返ってくる場合があるため
        logits = logits[0]
    # 最も確率が高いトークンのIDを取得 (argmax)
    return logits.argmax(dim=-1)

def compute_metrics(eval_pred):
    """
    予測結果と正解ラベルを比較してAccuracyを計算する関数
    """
    preds, labels = eval_pred
    
    # ラベルが -100 の場所は「学習対象外（プロンプトの指示部分など）」なので無視する
    # -100以外の場所（モデルが生成すべき回答部分）だけを評価対象にするマスクを作成
    mask = labels != -100
    
    # マスクを適用して、評価対象のトークンのみ抽出
    active_preds = preds[mask]
    active_labels = labels[mask]
    
    # 正解率の計算 (一致している数 / 全体数)
    accuracy = (active_preds == active_labels).mean()
    
    return {"accuracy": accuracy}

# 学習のための設定 --- (*7)
training_args = TrainingArguments(
    output_dir=DIR_MODEL,
    per_device_train_batch_size=8,  # メモリ節約のため16から8に削減
    gradient_accumulation_steps=8,  # 実質的なバッチサイズ = 8 * 8 = 64 (元の16 * 4 = 64と同じ)
    warmup_steps=5,
    max_steps=1500,
    learning_rate=2e-4,
    optim="adamw_8bit",
    logging_dir="./logs",
    logging_steps=10,  # 10ステップごとにログを出力
    weight_decay=0.01,
    lr_scheduler_type="linear",
    fp16 = not torch.cuda.is_bf16_supported(),
    bf16 = torch.cuda.is_bf16_supported(),
    report_to="none",  # W&Bなどの外部ログを無効化
    save_strategy="steps",
    save_steps=100,
    eval_strategy="steps",  # 検証を定期的に実行
    eval_steps=10,  # 10ステップごとに検証を実行
    save_total_limit=1,  # 最良のモデルのみ保存
    per_device_eval_batch_size=1,  # 検証時のバッチサイズも削減
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
    # ↓↓↓ ここに以下の2行を追加 ↓↓↓
    compute_metrics=compute_metrics,
    preprocess_logits_for_metrics=preprocess_logits_for_metrics,
)
trainer.train()

# 学習したモデルを保存 --- (*9)
model.save_pretrained(DIR_MODEL)
tokenizer.save_pretrained(DIR_MODEL)

# ---------------------------------------------------------
# 追加コード: 学習ログをCSVとして出力する
# ---------------------------------------------------------
import pandas as pd

# 1. ログ履歴の取得 (Trainerが自動で記録しているデータ)
log_history = trainer.state.log_history
df = pd.DataFrame(log_history)

# --- 修正ポイント ---
# Trainerのログは「学習だけの行」と「評価だけの行」に分かれていることが多いので、
# 'step' をキーにして、学習ログと評価ログを1つの行に結合（マージ）します。

# 学習ログのみ抽出 (lossがある行)
train_df = df[df["loss"].notna()][["step", "epoch", "learning_rate", "loss"]]
train_df = train_df.rename(columns={"loss": "train_loss"})

# 評価ログのみ抽出 (eval_lossがある行)
# ★ここで eval_accuracy も取得します
eval_df = df[df["eval_loss"].notna()][["step", "eval_loss", "eval_accuracy"]]

# stepを基準に結合 (how='outer'にすることで、評価がないstepも残す)
df_merged = pd.merge(train_df, eval_df, on="step", how="outer")

# epochやlearning_rateがNaNになっている行（評価だけの行）を埋める（前方穴埋め）
df_merged["epoch"] = df_merged["epoch"].ffill()
df_merged["learning_rate"] = df_merged["learning_rate"].ffill()

# step順にソート
df_formatted = df_merged.sort_values("step").reset_index(drop=True)

# 4. CSVファイルとして保存
csv_path = os.path.join(DIR_MODEL, "training_log.csv")
df_formatted.to_csv(csv_path, index=False)

# --- 追加コードここまで ---

print(f"学習ログを保存しました: {csv_path}")
print("--- データの先頭5行 ---")
print(df_formatted.head())
