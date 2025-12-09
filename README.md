ファインチューニングプログラム

root   
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


config.py  
ファインチューニングするための設定ファイル

requirements.txt  
必要なパッケージのリスト。できない可能性があるからantigravity君に。。。

excel_to_alpaca.py  
ExcelファイルをAlpaca形式のJSONLファイルに変換するスクリプト

fine_turning.py  
ファインチューニングを行うスクリプト

generate.py  
ファインチューニングしたモデルを元にテキスト生成するテストスクリプト
