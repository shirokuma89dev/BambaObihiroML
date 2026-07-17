"""
実戦用 レース当日予想。

出走表(card)を過去全レース結果に連結し、build_features と同一パイプラインで
Elo・累積勝率・タイム指数まで各馬の実履歴から算出したうえで、
学習済みランカー(banei_ranker.pkl)で着順を予測する。
末尾に置いた出走表行は「そのレース開始前」の値を受け取るためリークしない。

出力: 各レースの印(◎○▲△)・本命の根拠・三連単1点・AI確信度。

使い方:
  python src/models/predict.py                    # 最新の出走表CSVを自動選択
  python src/models/predict.py --card path.csv
"""
import os
import re
import sys
import glob
import argparse
import numpy as np
import pandas as pd
import joblib

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "features"))
from build_features import engineer_features  # noqa: E402

MODEL_PATH = "models/banei_ranker.pkl"
FEATURES_PATH = "models/banei_ranker_features.pkl"
MARKS = ["◎", "○", "▲", "△"]


def parse_card_odds(val):
    """出走表の '5.4(3)' -> (オッズ5.4, 人気3)。"""
    if pd.isna(val):
        return np.nan, np.nan
    m = re.search(r"([\d.]+)\s*\((\d+)\)", str(val))
    if m:
        return float(m.group(1)), float(m.group(2))
    m2 = re.search(r"[\d.]+", str(val))
    return (float(m2.group(0)), np.nan) if m2 else (np.nan, np.nan)


def card_to_results_schema(card):
    """出走表の列を、過去結果CSVと同じスキーマに変換(未確定の着順・タイムはNaN)。"""
    odds_pop = card["popularity"].apply(parse_card_odds)
    return pd.DataFrame({
        "race_id": card["race_id"], "date": card["date"], "race_no": card["race_no"],
        "race_name": card["race_name"], "weather": card["weather"],
        "track_moisture": card["track_moisture"], "rank": np.nan,
        "waku": card["waku"], "umaban": card["umaban"], "horse_name": card["horse_name"],
        "sex_age": np.nan, "sled_weight": card.get("sled_weight", np.nan),
        "jockey_name": card["jockey_name"], "trainer_name": np.nan,
        "horse_weight": card.get("horse_weight", np.nan), "time": np.nan, "margin": np.nan,
        "popularity": [p for _, p in odds_pop], "odds": [o for o, _ in odds_pop],
    })


def build_comment(row):
    """特徴量から本命の短評を自動生成する。"""
    tags = []
    tm = row.get("track_moisture_num", np.nan)
    if pd.notna(tm):
        tags.append("軽い馬場" if tm < 1.0 else ("重い力馬場" if tm > 3.0 else "標準馬場"))
    if row.get("elo_gap_to_top", -1) >= 0:
        tags.append("地力最上位")
    if row.get("recent_form_score", 9) <= 3.0:
        tags.append("近走好調")
    if row.get("horse_cum_win_rate", 0) >= 0.25:
        tags.append("勝率優秀")
    if row.get("pop_is_fav", 0) == 1:
        tags.append("市場も支持")
    return " / ".join(tags[:3]) if tags else "特筆なし"


def main():
    ap = argparse.ArgumentParser(description="ばんえいAI レース当日予想")
    ap.add_argument("--card", default="", help="出走表CSV(省略時は最新を自動選択)")
    args = ap.parse_args()
    card_path = args.card or sorted(glob.glob("data/raw/banei_race_card_*.csv"))[-1]

    model = joblib.load(MODEL_PATH)
    feats = joblib.load(FEATURES_PATH)

    card = pd.read_csv(card_path)
    card["race_id"] = card["race_id"].astype(str)
    card_ids = set(card["race_id"])

    hist = pd.concat([pd.read_csv(f) for f in glob.glob("data/raw/banei_race_results_*.csv")],
                     ignore_index=True)
    hist["race_id"] = hist["race_id"].astype(str)
    hist = hist[~hist["race_id"].isin(card_ids)]  # 既に結果があれば重複排除
    combined = pd.concat([hist, card_to_results_schema(card)], ignore_index=True)

    fdf = engineer_features(combined)
    fdf["race_id"] = fdf["race_id"].astype(str)
    for c in feats:
        if c not in fdf.columns:
            fdf[c] = 0.0
    fdf[feats] = fdf[feats].fillna(0)

    today = fdf[fdf["race_id"].isin(card_ids)].copy()
    today["score"] = model.predict(today[feats])

    print("=" * 60)
    print(f"  ばんえいAI レース当日予想   出走表: {os.path.basename(card_path)}")
    print("=" * 60)
    best = None
    for rid, r in today.sort_values(["race_no", "score"], ascending=[True, False]).groupby("race_no", sort=True):
        r = r.sort_values("score", ascending=False)
        # レース内softmaxで勝率化 -> 三連単1点の確信度
        e = np.exp(r["score"] - r["score"].max())
        pwin = (e / e.sum()).values
        trio_conf = pwin[0] * pwin[1] * pwin[2] if len(pwin) >= 3 else pwin[0]
        top = r.head(4).reset_index(drop=True)
        combo = "-".join(str(int(x)) for x in r.head(3)["umaban"].values)
        print(f"\n■ {int(rid)}R  {str(r.iloc[0]['race_name'])[:24]}")
        for i, row in top.iterrows():
            mk = MARKS[i] if i < len(MARKS) else "  "
            print(f"   {mk} {int(row['umaban']):>2} {str(row['horse_name'])[:10]:<10} "
                  f"人気{int(row['popularity_num']):>2}  勝率{row['horse_cum_win_rate']*100:4.0f}%")
        print(f"   └ 本命短評: {build_comment(top.iloc[0])}")
        print(f"   └ 三連単1点: {combo}   AI確信度 {trio_conf*100:.2f}%")
        if best is None or trio_conf > best[2]:
            best = (int(rid), combo, trio_conf)

    print("\n" + "-" * 60)
    print(f"▼ 本日の最推奨(最高確信): {best[0]}R 三連単 {best[1]} ({best[2]*100:.2f}%)")
    print("\n【誠実な注記】三連単1点の的中率は約3〜5%。多くは外れ、当たれば高配当という")
    print(" 戦略で、期待値プラスの保証はない。余剰資金で少額・娯楽として楽しむこと。")


if __name__ == "__main__":
    main()
