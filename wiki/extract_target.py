"""Wikipediaコーパスから『枚方』に関するテキストファイルを抽出する"""
import json
import os
import re

import config

def extract_files_from_corpus():
    """corpusディレクトリから枚方関連テキストを抽出する関数"""

    file_list = []


    for root, _dirs, files in os.walk(config.DIR_CORPUS_TEXT):
        for file in files:
            if not file.endswith(".txt"):
                continue

            path = os.path.join(root, file)
            with open(path, "r", encoding="utf-8") as f:
                text_raw = f.read()

            title = text_raw.split("\n")[0]

            # ノイズ除去（Categoryなど）
            text = re.sub(r'Category:\S+', '', text_raw)
            text = re.sub(r'\s+', ' ', text)

            # 短すぎる記事は除外
            if len(text) < 300:
                continue
            
            # ===== 枚方判定（ここだけが本質）=====
            if "枚方" in text:
                file_list.append(path)
                print(f"- 対象ファイル: {title} ({path})")

    print(f"対象のファイル数: {len(file_list)}")

    # 対象ファイル一覧をJSONに保存
    with open(config.CORPUS_TARGET_FILES, "w", encoding="utf-8") as f:
        json.dump(file_list, f, ensure_ascii=False, indent=2)

if __name__ == "__main__":
    extract_files_from_corpus()
