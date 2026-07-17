# Development Journal

ばんえい競馬予測AIモデル開発の開発日誌です。日々の進捗、課題、アイデアなどを記録します。

## 2026-07-15
- プロジェクト開始。`AGENTS.md` と `dev_journal.md` を作成し、作業環境を整備。
- ディレクトリ構造 (`src/`, `data/`) の設計とパッケージ用初期ファイルの配置。
- `requirements.txt` の作成と `python3 -m venv .venv` による仮想環境構築。
- `keiba.go.jp` 公式対応のデータ収集スクレイパー `src/scraper/banei_scraper.py` の構築。
- 帯広競馬場の2023年・2024年（計3,400レース超・31,995出走レコード）の過去レース結果・馬場水分量・積載重量・オッズデータの取得完了 (`data/raw/`)。
- 払戻金（配当金）テーブル取得ロジックを追加し、決定型（Deterministic）CLI対応プログラムとして `src/scraper/banei_scraper.py` をアップデート。
- 競馬場現地でのリアルタイム推論用出走表・馬体重・馬場水分量取得プログラム `src/scraper/banei_race_day_scraper.py` の作成。
- 機械学習モデル学習・推論用の特徴量自動生成モジュール `src/features/build_features.py`（パワー負荷率、馬場水分量、ローリング過去着順、騎手勝率等の前処理）の構築完了。
- 2023年〜2026年最新分（本日時点）の全4年分・計55,358件の完全学習・検証用特徴量データセット (`data/processed/features_train.csv`) を生成。
- 教材および再現実行用のステップバイステップ手順書として `README.md` を整備。
- 天候データ（`weather`）のOne-Hotエンコーディング処理（`weather_晴`, `weather_曇`等）を `src/features/build_features.py` に追加し、モデル用特徴量を拡張。
- STEP 1（レース内偏差値・斤量差・着順標準偏差・重馬場適性）の特徴量エンジニアリングを `build_features.py` に実装し、`src/models/train.py` で検証実行。2026年未来テストデータでの評価で **ROC-AUC: 0.6190 ➔ 0.6303**、**正解率: 67.67% ➔ 67.82%** に向上（モデル保存: `models/banei_top3_model_step1.pkl`）。

- 教材用モデル学習モジュール `src/models/train.py`（LightGBM・時系列分割検証）の作成と検証実行。2026年テストデータで ROC-AUC: 0.6190、正解率: 67.67% を達成し、`models/banei_top3_model.pkl` にモデルを保存。
- ランキング学習 `src/models/train_ranker.py` (LGBMRanker) および評価モジュール `src/models/evaluate.py` の構築。AI Top 3カバーで **88.60%的中**、AI Top 4カバーで **95.98%的中** を達成。
- 回収率（ROI）100%超え（黒字化）特化モジュール `src/models/train_profitable.py` を構築。2026年未来データ検証で、単勝穴馬・お宝馬スクリーニングにて **回収率 227.14% 〜 263.00%**、馬連単1点勝負にて **回収率 105.87%** を達成。`models/banei_profitable_win_model.pkl` に保存。
- 最先端アンサンブル学習モジュール `src/models/train_ensemble.py`（LightGBM + CatBoost + XGBoost の加重平均ブレンド）を構築。2026年未来検証データで過去最高の **ROC-AUC: 0.6462**、**AI Top 3 複勝的中率: 91.06%** を達成。`models/ens_*.pkl` に保存。
- 当日のリアルタイム推論スクリプト `src/models/predict.py` をアンサンブル統合版にアップデート完了。
- Plackett-Luce確率モデルに基づく全頭順位予測および3連単全組み合わせ（1着➔2着➔3着ぴったり的中）確率スコアリングモジュール `src/models/predict_trifecta.py` を構築。
- 3連複専用予測・バックテストモジュール `src/models/predict_sanrenfuku.py` を構築。2026年全890レースの実成績検証にて、**10点BOXで的中率 30.00%**（約3.3レースに1回的中）、**6点軸流しで的中率 22.25%**（約4.5レースに1回的中）を達成。
- netkeibaの公式プロ予想フォーマット（◎○▲△☆＋AI根拠短評＋馬連単・3連複・3連単買い目カード）を出力する `src/models/predict_netkeiba_style.py` を構築。

