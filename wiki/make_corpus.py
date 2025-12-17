"""LLMを使ってWikipediaから枚方コーパスを自動生成するスクリプト"""
import json
import requests
from config import *

# 使用モデルと API URL
MODEL_NAME = "llama3.2:3b"  # 軽量版 LLaMA3
OLLAMA_API_URL = "http://localhost:11434/api/chat"

# プロンプト（枚方市に関する文章を300字以内に要約）
GENERATE_PROMPT = """
「{title}」に関する以下の文章を日本語で300字以内に要約してください。
地理、歴史、文化、施設などを簡潔にまとめてください。
情報が不足している場合は「記載なし」と答えてください。

文章:
{text}
"""

# Ollama に問い合わせる関数（chat API対応）
def ask_ollama(prompt):
    for _ in range(3):  # 3回までリトライ
        try:
            response = requests.post(
                OLLAMA_API_URL,
                json={
                    "model": MODEL_NAME,
                    "messages": [{"role": "user", "content": prompt}],
                    "stream": False
                },
                timeout=120  # 重いモデルは長めに設定
            )
            response.raise_for_status()
            # chat API の場合は "message.content" に出力がある
            result = response.json()["message"]["content"].strip()
            return result
        except Exception as e:
            print(f"*** APIエラー: {e} - 再試行します...")
    return None

# Alpaca形式のJSONを生成
def generate_alpaca_format(file_list):
    alpaca_data = []

    for i, filepath in enumerate(file_list):
        with open(filepath, "r", encoding="utf-8") as f:
            input_text = f.read()

        # 入力文字数制限
        MAX_INPUT_CHARS = 1000
        input_text = input_text[:MAX_INPUT_CHARS]

        # タイトル取得
        title = input_text.split("\n")[0].lstrip("# ").strip()

        print("------------------------------")
        print(f"*** タイトル: {title} ({i+1}/{len(file_list)})")

        # プロンプト生成
        prompt = GENERATE_PROMPT.format(title=title, text=input_text)
        summary = ask_ollama(prompt)

        print(f"*** 要約結果: {summary}")

        if summary is None or "記載なし" in summary:
            continue

        alpaca_data.append({
            "instruction": "枚方市について説明してください。",
            "input": title,
            "output": summary
        })

    # JSON に出力
    with open(CORPUS_FILE_JSON, "w", encoding="utf-8") as fp:
        json.dump({"train": alpaca_data}, fp, ensure_ascii=False, indent=2)

    print("枚方用Alpaca形式のJSONを出力しました。")

# 実行
if __name__ == "__main__":
    with open(CORPUS_TARGET_FILES, "r", encoding="utf-8") as f:
        target_files = json.load(f)
        generate_alpaca_format(target_files)
