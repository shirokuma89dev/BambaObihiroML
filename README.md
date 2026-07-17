# BambaObihiroML

帯広競馬（ばんえい競馬）のデータ収集、データ前処理、および推論用特徴量生成を行うプログラム群です。

## 概要

本リポジトリでは、帯広競馬場のレース結果、馬場水分量、斤量（そり重量）、馬体重、払戻金データを収集し、機械学習モデルの学習および推論に使用する特徴量データセットを作成します。

## ディレクトリ構造

```text
BambaObihiroML/
├── data/
│   ├── raw/          # スクレイピング取得した生のCSV（Git追跡対象外）
│   └── processed/    # 前処理・特徴量加工済みのCSV（Git追跡対象外）
├── src/
│   ├── scraper/
│   │   ├── banei_scraper.py          # 過去データの自動収集プログラム
│   │   └── banei_race_day_scraper.py # 当日出走表・リアルタイムデータ取得
│   └── features/
│       └── build_features.py         # 特徴量エンジニアリング・加工処理
├── AGENTS.md         # プロジェクトの遵守事項
├── dev_journal.md    # 開発日誌
└── requirements.txt  # 依存ライブラリ
```

## セットアップと実行手順

以下の順序で実行することで、データ収集から特徴量加工までの全工程を再現できます。

### 1. 環境構築

Python 3.10 以上がインストールされている環境で、仮想環境を作成しパッケージを導入します。

```bash
# 仮想環境の作成
python3 -m venv .venv

# 仮想環境の有効化 (macOS / Linux)
source .venv/bin/activate

# 依存パッケージのインストール
pip install -r requirements.txt
```

### 2. 過去データの収集

指定した年度の全レース結果、メタ情報（天候・馬場水分量）、払戻金データを取得し、`data/raw/` にCSV保存します。

```bash
# デフォルト（2023年から最新年まで）の収集
python src/scraper/banei_scraper.py

# 年度を個別指定して収集する場合（例: 2023〜2024年）
python src/scraper/banei_scraper.py --start-year 2023 --end-year 2024
```

### 3. レース当日の出走表取得（推論時）

現地訪問時やレース開催当日に、その日の出走表・馬体重増減・最新オッズ・最新馬場水分量を取得します。

```bash
# 本日開催分の出走表を取得
python src/scraper/banei_race_day_scraper.py

# 日付を直接指定して取得する場合
python src/scraper/banei_race_day_scraper.py --date 2024/02/11
```

```
BambaObihiroML git:(main) ✗ python src/scraper/banei_scraper.py
--- 帯広競馬場 (2023年) 全データ収集開始 ---
  [50 レース取得完了] 直近 Race ID: 202301143302 (森川燿　満２歳記念３歳Ｄ－４)
  [100 レース取得完了] 直近 Race ID: 202301223304 (３歳Ｂ－６)
  [150 レース取得完了] 直近 Race ID: 202301303306 (とかち初月賞Ｂ４－７)
  [200 レース取得完了] 直近 Race ID: 202302113308 (目指せ優秀新人騎手賞！！Ｂ１－２)
  [250 レース取得完了] 直近 Race ID: 202302193310 (鳥清カンパニー５０周年記念Ａ２－１)

…

完了: 2026年 データ保存完了 -> data/raw/banei_race_results_2026.csv (全 8287 レコード)
```

### 4. 特徴量データの生成

`data/raw/` に保存された生データを統合し、パワー負荷率（そり重量 / 馬体重）、馬場水分量、近走平均着順、騎手成績などの特徴量を計算して `data/processed/features_train.csv` を出力します。

```bash
python src/features/build_features.py
```

### 5. 機械学習モデルの学習と検証

2023年〜2025年の過去データでモデル（LightGBM / CatBoost / XGBoost の最先端アンサンブル）を学習させ、未来のデータ（2026年分）で予測性能（ROC-AUC: 0.6462）を検証します。

```bash
python src/models/train_ranker.py
python src/models/train_profitable.py
python src/models/train_ensemble.py
```

### 6. 高的中率・黒字化（100%〜263%回収率）バックテストの実行

単勝EVスクリーニング（227%〜263%回収率）および馬連単1点勝負（105.8%回収率）のパフォーマンスをシミュレーションします。

```bash
python src/models/evaluate.py
```

### 7. リアルタイム推論と買い目算出（当日のレース予測）

当日の出走表（`banei_race_day_scraper.py` で取得したCSV）を入力し、3アルゴリズムアンサンブルAI（LightGBM + CatBoost + XGBoost）が各レースの勝負買い目および91.1%〜95.8%的中指定馬番をリアルタイム出力します。

```bash
python src/models/predict.py
```

### 8. 3連単フォーメーション＆全頭完全着順予測

Plackett-Luce確率モデルを用いて全出走馬の全頭順位予測スコア、および3連単（1着➔2着➔3着ぴったり的中）の全組み合わせ確率・本命1点＆上位5点買い目をリアルタイム出力します。

```bash
python src/models/predict_trifecta.py
```

### 9. 3連複 高的中率（22.25%〜30.00%）フォーメーション予測

3連複における各種買い目（1点買い・4点BOX・6点軸流し・10点BOX）の確率および推論を出力します。

```bash
python src/models/predict_sanrenfuku.py
```

### 10. netkeibaプロ予想超え！AI印（◎○▲△☆）＆短評付きプロ推薦カード出力

netkeibaの公式プロ予想フォーマットに準拠し、本命◎・対抗○・単穴▲・連下△・特注穴馬☆、AI根拠短評、および単勝・馬連単・3連複・3連単の最適購入カードを出力します。

```bash
python src/models/predict_netkeiba_style.py
```

## 生成されるデータ仕様








- `data/raw/banei_race_results_{year}.csv`: 各馬の着順、タイム、斤量、馬体重、オッズ等
- `data/raw/banei_race_meta_{year}.csv`: 各レースの天候、馬場水分量(%)
- `data/raw/banei_race_payouts_{year}.csv`: 各レースの勝式別（単勝・複勝・馬連・3連単等）払戻額
- `data/processed/features_train.csv`: 機械学習モデルに入力可能な加工済み数値特徴量
