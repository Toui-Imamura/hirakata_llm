import json
import pandas as pd

# 同じ階層の config.py を読み込む
from config import *

# 改行コード・余計なコードをクリーニングする関数
def clean_text(s):
    if isinstance(s, str):
        return (
            s.replace("_x000D_", "")  # Excel 特有の CR 表記
             .replace("\r", "")       # CR
             .strip()
        )
    return s

# Excelファイルを読み込む
df = pd.read_excel(EXCEL_FILE, index_col=0)

records = []

for _, row in df.iterrows():
    record = {
        "instruction": clean_text(row["質問内容"]),
        "input": "",
        "output": clean_text(row["回答内容"]),
    }

    records.append(record)
# JSON で保存
with open(JSON_XLSX_DATASET_FILE, "w", encoding="utf-8") as f:
    for r in records:
        f.write(json.dumps(r, ensure_ascii=False) + "\n")

print("JSON を保存しました:", JSON_XLSX_DATASET_FILE)




