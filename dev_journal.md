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






