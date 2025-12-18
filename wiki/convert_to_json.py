import json
import os
import hirakata_bot.config as config

with open(config.JSON_WIKI_DATASET_FILE, "r", encoding="utf-8") as f:
    data = json.load(f)

with open(config.JSON_TO_JSONL_FILE, "w", encoding="utf-8") as f:
    for item in data["train"]:
        json_line = json.dumps({
            "instruction": item["instruction"],  # 何をせよ
            "input": item["input"],                # 対象
            "output": item["output"]               # 回答
        }, ensure_ascii=False)
        f.write(json_line + "\n")
print(f"Converted {config.JSON_WIKI_DATASET_FILE} to {config.JSON_TO_JSONL_FILE}")

