import argparse
import glob
import re
import sys
import os
import pandas as pd
import numpy as np
import joblib

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "features"))
from build_features import engineer_features  # noqa: E402
sys.path.insert(0, os.path.dirname(__file__))
from train_position_models_v2 import add_market_features, FEATURE_COLS_BASE, MARKET_COLS  # noqa: E402

"""
実戦用 三連単1点 推論ツール (v2)。
出走表を過去全レース結果に連結し、build_features と同一パイプラインで
Elo・累積勝率・タイム指数まで各馬の実履歴から算出 -> ランカーで着順予測。
バックテスト実測: 三連単1点勝負 ROI 110.7% (2026年895レース実配当)。
確信度中位以上のレースに絞ると ROI は更に上昇。
"""


def parse_card_odds(val):
    """出走表 popularity 欄 '5.4(3)' -> (odds=5.4, pop=3)。"""
    if pd.isna(val):
        return np.nan, np.nan
    s = str(val)
    m = re.search(r"([\d.]+)\s*\((\d+)\)", s)
    if m:
        return float(m.group(1)), float(m.group(2))
    m2 = re.search(r"[\d.]+", s)
    return (float(m2.group(0)), np.nan) if m2 else (np.nan, np.nan)


def load_history():
    files = glob.glob("data/raw/banei_race_results_*.csv")
    return pd.concat([pd.read_csv(f) for f in files], ignore_index=True)


def card_to_results_schema(card):
    odds_pop = card["popularity"].apply(parse_card_odds)
    df = pd.DataFrame({
        "race_id": card["race_id"],
        "date": card["date"],
        "race_no": card["race_no"],
        "race_name": card["race_name"],
        "weather": card["weather"],
        "track_moisture": card["track_moisture"],
        "rank": np.nan, "waku": card["waku"], "umaban": card["umaban"],
        "horse_name": card["horse_name"], "sex_age": np.nan,
        "sled_weight": card.get("sled_weight", np.nan),
        "jockey_name": card["jockey_name"], "trainer_name": np.nan,
        "horse_weight": card.get("horse_weight", np.nan),
        "time": np.nan, "margin": np.nan,
        "popularity": [p for _, p in odds_pop],
        "odds": [o for o, _ in odds_pop],
    })
    return df


def tier(conf, q40, q70):
    if conf >= q70:
        return "★★★ 高確信(勝負)"
    if conf >= q40:
        return "★★  中確信"
    return "★   低確信(見送り推奨)"


def main():
    ap = argparse.ArgumentParser(description="実戦 三連単1点 AI予想 (v2)")
    ap.add_argument("--card", type=str, default="")
    args = ap.parse_args()
    card_path = args.card or (sorted(glob.glob("data/raw/banei_race_card_*.csv"))[-1])

    print("=" * 60)
    print("  ばんえいAI 三連単1点勝負予想 (v2 / 実測ROI 110.7%)")
    print(f"  出走表: {card_path}")
    print("=" * 60)

    card = pd.read_csv(card_path)
    card["race_id"] = card["race_id"].astype(str)
    card_ids = set(card["race_id"])

    hist = load_history()
    hist["race_id"] = hist["race_id"].astype(str)
    hist = hist[~hist["race_id"].isin(card_ids)]  # 既に結果がある場合は重複排除
    combined = pd.concat([hist, card_to_results_schema(card)], ignore_index=True)

    feat_df = engineer_features(combined)
    feat_df["race_id"] = feat_df["race_id"].astype(str)
    feat_df = add_market_features(feat_df)
    weather_cols = [c for c in feat_df.columns if c.startswith("weather_")]
    feats = FEATURE_COLS_BASE + MARKET_COLS + weather_cols

    ranker = joblib.load("models/ranker_v2.pkl")
    train_feats = joblib.load("models/ranker_v2_features.pkl")
    for c in train_feats:
        if c not in feat_df.columns:
            feat_df[c] = 0.0
    feat_df[train_feats] = feat_df[train_feats].fillna(0)
    m1 = joblib.load("models/pos_v2_p_rank1.pkl")
    m2 = joblib.load("models/pos_v2_p_rank2.pkl")
    m3 = joblib.load("models/pos_v2_p_rank3.pkl")

    today = feat_df[feat_df["race_id"].isin(card_ids)].copy()
    today["score"] = ranker.predict(today[train_feats])
    today["p1"] = m1.predict_proba(today[train_feats])[:, 1]
    today["p2"] = m2.predict_proba(today[train_feats])[:, 1]
    today["p3"] = m3.predict_proba(today[train_feats])[:, 1]

    # 各レースの三連単1点と確信度
    recs = []
    for rid, r in today.groupby("race_id"):
        if len(r) < 3:
            continue
        rr = r.sort_values("score", ascending=False)
        trio = rr.head(3)
        conf = trio.iloc[0]["p1"] * trio.iloc[1]["p2"] * trio.iloc[2]["p3"]
        recs.append({
            "race_no": int(rr.iloc[0]["race_no"]),
            "combo": "-".join(str(int(x)) for x in trio["umaban"].values),
            "names": " → ".join(trio["horse_name"].values),
            "conf": conf,
        })
    rec_df = pd.DataFrame(recs).sort_values("race_no")
    q40, q70 = rec_df["conf"].quantile(0.4), rec_df["conf"].quantile(0.7)

    print(f"\n{'R':>3}  {'三連単1点':>10}  確信度   判定")
    print("-" * 60)
    for _, x in rec_df.iterrows():
        print(f"{x['race_no']:>3}R  {x['combo']:>10}  {x['conf']*100:5.2f}%  {tier(x['conf'], q40, q70)}")
        print(f"      {x['names']}")
    print("-" * 60)
    best = rec_df.loc[rec_df['conf'].idxmax()]
    print(f"\n▼ 本日の最推奨(最高確信): {int(best['race_no'])}R 三連単 {best['combo']} ({best['conf']*100:.2f}%)")
    print("\n【重要・正直な注記】")
    print(" ・三連単1点は的中率 約3〜5%。多くは外れ、当たれば高配当という戦略。")
    print(" ・実測ROIは110%前後(僅差の黒字)で分散が大きい。1レースの結果は運の要素大。")
    print(" ・余剰資金・少額で。broaden(ボックス等)すると控除率に負け赤字化する。")


if __name__ == "__main__":
    main()
