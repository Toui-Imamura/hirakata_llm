import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import os
from config import DIR_MODEL  # configから保存先フォルダ名を読み込む

# ---------------------------------------------------------
# 1. データの読み込みと前処理
# ---------------------------------------------------------
# 絶対パスでCSVを指定（前回の修正箇所）
# 絶対パス（ファイルの住所）を直接指定する
csv_path = "/home/j23039/llama31_3b_model/training_log.csv"

try:
    df = pd.read_csv(csv_path)
except FileNotFoundError:
    print(f"エラー: {csv_path} が見つかりません。")
    print("fine_turning.py が最後まで完了しているか、パスが合っているか確認してください。")
    exit()

# 学習データ(Train)と検証データ(Eval)を分ける
df_train = df.dropna(subset=['train_loss']).copy()
df_eval = df.dropna(subset=['eval_loss']).copy()

# ★ここが修正ポイント★
# マージするときに、必要なカラム('step'と'loss')だけを抽出してから結合します。
# こうすることで、不要なカラムの衝突（名前変更）を防ぎます。
df_merge = pd.merge_asof(
    df_eval[['step', 'eval_loss']].sort_values('step'),   # 検証データからはこの2列だけ
    df_train[['step', 'train_loss']].sort_values('step'), # 学習データからはこの2列だけ
    on='step',
    direction='nearest'
)

# これで KeyError にならず計算できます
df_merge['gen_gap'] = df_merge['eval_loss'] - df_merge['train_loss']

# ---------------------------------------------------------
# 2. グラフの描画設定
# ---------------------------------------------------------
plt.style.use('seaborn-v0_8-whitegrid') 

# 3行2列のレイアウト
fig, axes = plt.subplots(3, 2, figsize=(16, 15))
fig.suptitle('Training Metrics Analysis', fontsize=16)

# --- (1行目 左) Training Loss History ---
ax = axes[0, 0]
ax.plot(df_train['step'], df_train['train_loss'], label='Train Loss', color='#1f77b4', alpha=0.8)
ax.scatter(df_eval['step'], df_eval['eval_loss'], label='Eval Loss', color='#d62728', zorder=5)
ax.set_title('Training Loss History', fontweight='bold')
ax.set_xlabel('Step')
ax.set_ylabel('Loss')
ax.legend()
ax.grid(True, which='both', linestyle='--', alpha=0.5)

# --- (1行目 右) Loss vs Epoch ---
ax = axes[0, 1]
ax.plot(df_train['epoch'], df_train['train_loss'], label='Train Loss', color='#1f77b4', alpha=0.8)
ax.scatter(df_eval['epoch'], df_eval['eval_loss'], label='Eval Loss', color='#d62728', zorder=5)
ax.set_title('Loss vs Epoch', fontweight='bold')
ax.set_xlabel('Epoch')
ax.set_ylabel('Loss')
ax.legend()
ax.grid(True, which='both', linestyle='--', alpha=0.5)

# --- (2行目 左) Learning Rate Schedule ---
ax = axes[1, 0]
df_lr = df.dropna(subset=['learning_rate'])
ax.plot(df_lr['step'], df_lr['learning_rate'], color='#9467bd', linewidth=2)
ax.set_title('Learning Rate Schedule', fontweight='bold')
ax.set_xlabel('Step')
ax.set_ylabel('Learning Rate')
ax.ticklabel_format(style='sci', axis='y', scilimits=(0,0))
ax.grid(True, which='both', linestyle='--', alpha=0.5)

# --- (2行目 右) Generalization Gap ---
ax = axes[1, 1]
if len(df_merge) > 1:
    width = (df_merge['step'].max() - df_merge['step'].min()) / len(df_merge) * 0.8
else:
    width = 10
ax.bar(df_merge['step'], df_merge['gen_gap'], width=width, color='#ff7f0e', alpha=0.8, align='center')
ax.set_title('Generalization Gap (Eval - Train)', fontweight='bold')
ax.set_xlabel('Step')
ax.set_ylabel('Eval Loss - Train Loss')
ax.axhline(0, color='black', linewidth=0.8)
ax.grid(True, axis='y', linestyle='--', alpha=0.5)

# --- (3行目 左) Accuracy History ---
ax = axes[2, 0]
if 'eval_accuracy' in df_eval.columns:
    ax.plot(df_eval['step'], df_eval['eval_accuracy'], label='Eval Accuracy', color='#2ca02c', marker='o', markersize=4)
    
    # 最大精度の地点に注釈
    if not df_eval.empty:
        max_acc = df_eval['eval_accuracy'].max()
        max_step = df_eval.loc[df_eval['eval_accuracy'].idxmax(), 'step']
        ax.annotate(f'Max Acc: {max_acc:.4f}', 
                    xy=(max_step, max_acc), 
                    xytext=(max_step, max_acc * 1.05),
                    arrowprops=dict(facecolor='black', shrink=0.05))

    ax.set_title('Evaluation Accuracy History', fontweight='bold')
    ax.set_xlabel('Step')
    ax.set_ylabel('Accuracy')
    ax.legend(loc='lower right')
    ax.grid(True, which='both', linestyle='--', alpha=0.5)
else:
    ax.text(0.5, 0.5, 'No Accuracy Data Found', ha='center', va='center', fontsize=12)

# --- (3行目 右) Summary Text ---
ax = axes[2, 1]
ax.axis('off')
if 'eval_accuracy' in df_eval.columns and not df_eval.empty:
    best_loss = df_eval['eval_loss'].min()
    best_acc = df_eval['eval_accuracy'].max()
    final_loss = df_eval.iloc[-1]['eval_loss']
    final_acc = df_eval.iloc[-1]['eval_accuracy']
    
    text_str = (
        f"--- Training Summary ---\n\n"
        f"Best Eval Loss: {best_loss:.4f}\n"
        f"Best Accuracy:  {best_acc:.4f}\n\n"
        f"Final Loss:     {final_loss:.4f}\n"
        f"Final Accuracy: {final_acc:.4f}\n\n"
        f"Total Steps:    {df['step'].max()}"
    )
    ax.text(0.1, 0.5, text_str, fontsize=14, family='monospace', va='center')

# ---------------------------------------------------------
# 3. 表示と保存
# ---------------------------------------------------------
plt.tight_layout(rect=[0, 0.03, 1, 0.95]) 
plt.savefig("training_analysis_with_accuracy.png", dpi=300) 
print("グラフを 'training_analysis_with_accuracy.png' として保存しました。")