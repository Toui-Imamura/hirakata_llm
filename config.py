"""プロジェクト設定""" 
import os # ディレクトリ設定 
DIR_CORPUS = "corpus"
DIR_DATA = "/home/j23039/data"
DIR_MODEL = "/home/j23039/llama31_3b_model/"

# 入力ファイル 
EXCEL_FILE = os.path.join(DIR_DATA, "FAQlist.xlsx")

# 生成ファイル 
JSON_DATASET_FILE = os.path.join(DIR_CORPUS, "FAQdataset.json") 


# パラメータ 
MODEL_NAME = "unsloth/gpt-oss-20b-unsloth-bnb-4bit" 
OLLAMA_TIMEOUT = 30 
MAX_SEQ_LENGTH = 2048 
MAX_STEPS = 1500 
MAX_DATASET_SIZE = 3000

# RAG Settings
DIR_VECTOR_DB = "vector_db"
EMBEDDING_MODEL = "granite-embedding:278m"
LLM_MODEL = "qwen3:8b" # User pulled this
RETRIEVAL_K = 3
TEMPERATURE = 0.1
CHUNK_SIZE = 500
CHUNK_OVERLAP = 50
QA_TEMPLATE = """
あなたは役に立つアシスタントです。以下の「参照ドキュメント」だけを元にして、質問に答えてください。
答えがわからない場合は、「わかりません」と答えてください。

# 参照ドキュメント:
{context}

# 質問:
{question}

# 回答:
"""
