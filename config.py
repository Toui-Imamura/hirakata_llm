"""プロジェクト設定""" 
import os # ディレクトリ設定 
DIR_CORPUS = "corpus"
DIR_DATA = "/home/aichatbot25/data"
DIR_MODEL = "model"

# 入力ファイル 
EXCEL_FILE = os.path.join(DIR_DATA, "FAQlist.xlsx")

# 生成ファイル 
JSON_DATASET_FILE = os.path.join(DIR_CORPUS, "FAQdataset.json") 


# パラメータ 
MODEL_NAME = "unsloth/Llama-3.2-3B-Instruct" 
OLLAMA_TIMEOUT = 30 
MAX_SEQ_LENGTH = 2048 
MAX_STEPS = 500 
MAX_DATASET_SIZE = 3000