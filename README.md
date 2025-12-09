ファインチューニングプログラム

h_bot/hirakata_bot  
├── __pycache__  
├── .venv  
├── corpus
├── data(ここ直下にexcelファイル置く)
├── model
│   ├── finetuned
│   ├── model_merged
│   └── tokenizer
├── config.py
├── convert_gguf.py
├── excel_to_alpaca.py
├── fine_turning.py
├── generate.py
└── requirements.txt


config.py<br>
ファインチューニングするための設定ファイル

requirements.txt<br>
必要なパッケージのリスト。できない可能性があるからantigravity君に。。。

excel_to_alpaca.py<br>
ExcelファイルをAlpaca形式のJSONLファイルに変換するスクリプト

fine_turning.py<br>
ファインチューニングを行うスクリプト

generate.py<br>
ファインチューニングしたモデルを元にテキスト生成するテストスクリプト
