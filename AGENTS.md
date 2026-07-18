# AI Agents & Tools

ばんえい競馬予測AIモデル開発に関連するエージェントやツールの役割、構成などをまとめるファイルです。

## プロジェクト遵守事項

1. **データの非公開**: 今後スクレイピングしたデータはGitHubで公開せず、追跡対象外（`.gitignore`）のディレクトリに読み込んでから使用する（スクレイピングコードの公開は可）。
2. **指示ベースの進行**: プロジェクトはユーザーのペースで進行する。未指示の作業を勝手に進めたり、先の工程を推測して提案するようなおせっかいは厳禁。
3. **作業前の事前通知**: ファイルの書き込み等の作業を開始する前に、必ず「何をしようとしているか」を文章で通知する。事後報告のみの通知は禁止。
4. **コミットの提案**: リポジトリはGit管理のため、AI自身がコミットする必要はないが、適切なタイミングでのコミット提案とコミットメッセージの例文提示は行う（実際のコミット操作はユーザーが行う）。
5. **簡潔な回答**: AIトークンの浪費を防ぐため、相槌や枕詞的な文章は使用しない。

---

## プロジェクト概要
帯広競馬場で開催される「ばんえい競馬」に特化した高精度着順予測AIモデルの開発プロジェクト。競馬プロ予想家（netkeiba等）の予測精度を凌駕し、期待回収率をプラスに持っていくための「ガチ性能」を目指す。

## 現在のモデル構成と検証結果

### 1. 特徴量システム（計90次元 = 84 + 気象one-hot）
`src/features/build_features.py` の `engineer_features(df)` が一括生成。正規リストは同ファイルの `FEATURE_COLS`。
- **馬場水分物理相互作用**: `power_moisture_interaction`、`sled_weight_moisture_interaction`。
- **夏競馬（猛暑乾燥・夏バテ）物理**: `is_summer_season`, `summer_heat_stress`（気温÷水分×パワー比）, `summer_heat_fatigue`（真夏日馬体重減）, `sand_drying_index`（乾燥摩擦）。
- **冬競馬（凍結・氷結滑走）物理**: `is_freezing_temp`, `freezing_ice_friction`（氷結滑走摩擦）, `snow_ice_moisture_interaction`（氷雪複合）。
- **調子・疲労・勢い (EWM)**: `horse_ewm_speed`（直近5走スピード加重平均）, `jockey_ewm_win_rate`/`jockey_ewm_top3_rate`（直近15戦騎手調子ウェーブ）, `fatigue_index`, `recent_form_score`。
- **ハンデ軽量化**: `sled_weight_relief`（近走平均からの軽量化度合い）。
- **クラス別実績**: `horse_class_win_rate`, `horse_class_top3_rate`（馬の現クラス実績）。
- **Eloレーティング**: `horse_elo_pre`, `jockey_elo_pre`（対戦相手の強さを織り込む動的能力指標）。
- **斤量・馬場補正タイム指数**: `horse_speed_figure`（時計を条件で補正した実力）。
- **気象庁外部データ**: `data/raw/weather/jma_obihiro_weather.csv`。
- **市場(人気順)**: `popularity_num`, `pop_is_fav`, `pop_inv`, `pop_zscore`。

### 2. 予測アーキテクチャ
- **LightGBM LambdaRank**（ランキング学習）。レース内で全馬を着順順に並べ替えることを直接最適化し、競争構造（1レース1勝ち馬・排他）に適合。
- ハイパラは **Optuna × 複数シーズンのウォークフォワード** で最適化（過学習を防ぐため、浅い決定木 `max_depth: 3` × 強正則化 `reg_alpha: 7.26` に制限）。

### 3. 検証結果（ウォークフォワード：学習<年 → 検証=年）※誠実な実測値
- **Optuna探索 平均1着的中率 (平均top1)**: **35.10%** （90次元モデルにおける歴代最高値）
- **1着的中率**: 2024 33.7% / 2025 35.1% / 2026 35.5% ≒ **市場（1番人気）と互角以上**。
- **回収率(ROI)**: 三連単1点勝負で 2026通年 **82.5%** まで改善（前半 73.2% / 後半 92.1%）。依然として控除率の壁（約20%の負期待値）があるため、少額娯楽用途を前提とする。

---

## プロジェクトの主要スクリプト（教材品質・コア5本）

| スクリプトパス | 役割 |
| :--- | :--- |
| `src/scraper/banei_scraper.py` | 帯広の過去レース結果・払戻を取得しCSV化。 |
| `src/scraper/scrape_trifecta_payouts.py` | 三連単・三連複の実配当を取得（漢字「三連単」表記・rowspan対応）。 |
| `src/features/build_features.py` | 全特徴量生成（`engineer_features`）と正規リスト `FEATURE_COLS`。 |
| `src/models/train.py` | LambdaRank学習（`--tune N` でOptuna再探索）。→ `models/banei_ranker.pkl` |
| `src/models/evaluate.py` | ウォークフォワード精度 vs 市場 ＋ 三連単ROIの誠実な検証。 |
| `src/models/predict.py` | 出走表から印(◎○▲△)・本命短評・三連単1点・AI確信度を出力。 |

標準的な実行順: `build_features.py` → `train.py` → `evaluate.py` → `predict.py`。

---

## 次回作業への引き継ぎ事項

コード整理統合（実験20本→コア3本）とREADME教育ドキュメント化は完了済み（詳細は `dev_journal.md` 冒頭の「引き継ぎ要約」参照）。

1. **軽微な既知の割り切り**: `compute_speed_figure` の回帰は全期間で係数フィット（物理関係のため軽微なリーク）。厳密には学習期間のみで係数推定するのが理想。
2. **精度の頭打ち**: 予測精度は市場水準（≈35%）が天井。これ以上はデータ・手法では伸びにくい。挑戦するなら新規データ源（調教評価・血統・当日気配）の追加が方向性。
3. **未解消の緊張**: ユーザーはREADME冒頭を自ら「初めての競馬でお金を増やしたい」に書き換えている。技術的には「儲かるAIは作れない」と繰り返し確定済みだが、根本動機は解消されていない可能性がある。新たな根拠なく「儲かる」と誇張しないこと。
