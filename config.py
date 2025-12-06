"""プロジェクト設定""" 
import os # ディレクトリ設定 
DIR_CORPUS =  "corpus"
DIR_DATA =  "data"
DIR_MODEL = "model"

# 入力ファイル 
EXCEL_FILE = os.path.join(DIR_DATA, "FAQlist.xlsx")

# 生成ファイル 
JSON_DATASET_FILE = os.path.join(DIR_CORPUS, "FAQdataset.json") 

# モデル出力 
MODEL_OUTPUT_DIR = os.path.join(DIR_MODEL, "finetuned") 
TOKENIZER_DIR = os.path.join(DIR_MODEL, "tokenizer") 

# パラメータ 
MODEL_NAME = "unsloth/Meta-Llama-3.1-8B-bnb-4bit" 
OLLAMA_TIMEOUT = 30 
MAX_SEQ_LENGTH = 2048 
MAX_STEPS = 500 
MAX_DATASET_SIZE = 3000