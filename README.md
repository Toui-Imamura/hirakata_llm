hirakata_bot/  
├── config.py  
│   └── ファインチューニングに関する各種設定をまとめた設定ファイル  
│  
├── requirements.txt  
│   └── 本プロジェクトで使用する必要なPythonパッケージの一覧  
│  
├── excel_to_alpaca.py  
│   └── ExcelファイルをAlpaca形式のJSONLデータに変換するスクリプト  
│  
├── fine_tuning.py  
│   └── データセットを用いてモデルのファインチューニングを行うスクリプト  
│  
├── generate.py  
│   └── ファインチューニング済みモデルを用いてテキスト生成を行うテスト用スクリプト  
│  
├── corpus/  
│   └── 学習用データ（JSON / JSONL など）を格納するディレクトリ  
│  
├── model/  
│   └── ファインチューニング後のモデルおよび重みファイルを保存するディレクトリ  
│  
├── wiki/  
│   ├── extract_text.py  
│   │   └── Wikipediaダンプから本文テキストを抽出するスクリプト  
│   ├── extract_target.py  
│   │   └── 枚方市に関連する記事を抽出するスクリプト  
│   ├── convert_to_json.py  
│   │   └── 抽出したデータをJSON形式に変換するスクリプト  
│   └── make_corpus.py  
│       └── Wikipedia由来データから学習用コーパスを作成するスクリプト  
│  
└── __init__.py  
    └── 本ディレクトリをPythonパッケージとして認識させるためのファイル  
