import random
import pandas as pd
import numpy as np
import lightgbm as lgb
import joblib

from train_position_models_v2 import add_market_features, FEATURE_COLS_BASE, MARKET_COLS

"""
ランカーのハイパラ最適化 + エッジ堅牢性検証。
- 選択: 2023-2024学習 -> 2025で三連単ドンピシャ的中率を最大化 (test非汚染)
- 最終: 最良paramで2023-2025再学習 -> 2026実配当で三連単1点ROI
- 安定性: 2026を前半/後半に割り実測ROIが両期間で黒字かを確認
"""


def paynum(s):
    return float(str(s).replace("円", "").replace(",", "").strip())


def load():
    df = pd.read_csv("data/processed/features_train.csv")
    df["date"] = pd.to_datetime(df["date"])
    df = add_market_features(df)
    wc = [c for c in df.columns if c.startswith("weather_")]
    feats = FEATURE_COLS_BASE + MARKET_COLS + wc
    df[feats] = df[feats].fillna(0)
    df = df.dropna(subset=["rank_num"]).copy()
    df["race_id"] = df["race_id"].astype(str)
    df["field"] = df.groupby("race_id")["rank_num"].transform("count")
    df["rel"] = (df["field"] - df["rank_num"]).clip(lower=0).astype(int)
    return df, feats


def fit_ranker(tr, feats, params):
    tr = tr.sort_values("race_id")
    grp = tr.groupby("race_id", sort=True).size().values
    r = lgb.LGBMRanker(objective="lambdarank", metric="ndcg", random_state=42, verbose=-1, **params)
    r.fit(tr[feats], tr["rel"], group=grp)
    return r


def trifecta_hit_rate(model, test, feats):
    """三連単ドンピシャ的中率(配当不要・resultsのみで計算可)。"""
    test = test.copy()
    test["score"] = model.predict(test[feats])
    hit = tot = 0
    for _, r in test.groupby("race_id"):
        if len(r) < 3:
            continue
        order = r.sort_values("score", ascending=False)["umaban"].values[:3]
        a = r.sort_values("rank_num")["umaban"].values[:3]
        tot += 1
        if list(order) == list(a):
            hit += 1
    return hit / tot * 100 if tot else 0


def trifecta_roi(model, test, feats, tri_lut):
    test = test.copy()
    test["score"] = model.predict(test[feats])
    staked = returned = hits = races = 0
    for rid, r in test.groupby("race_id"):
        if len(r) < 3 or rid not in tri_lut:
            continue
        races += 1
        order = [int(x) for x in r.sort_values("score", ascending=False)["umaban"].values[:3]]
        staked += 100
        wc, p = tri_lut[rid]
        if "-".join(str(x) for x in order) == wc:
            returned += p
            hits += 1
    roi = returned / staked * 100 if staked else 0
    return roi, hits, races


def main():
    df, feats = load()
    train = df[df["date"] < "2025-01-01"]
    val = df[(df["date"] >= "2025-01-01") & (df["date"] < "2026-01-01")]
    test = df[df["date"] >= "2026-01-01"]
    trainval = df[df["date"] < "2026-01-01"]
    print(f"学習{len(train)} / 検証(2025){len(val)} / テスト(2026){len(test)}")

    pay = pd.read_csv("data/raw/banei_trifecta_payouts_2026.csv")
    tri = pay[pay.ticket_type == "三連単"].copy(); tri["pay"] = tri.payout.apply(paynum)
    tri_lut = dict(zip(tri.race_id.astype(str), zip(tri.combination, tri.pay)))

    random.seed(0)
    space = {
        "n_estimators": [400, 600, 800, 1000, 1200],
        "learning_rate": [0.01, 0.02, 0.03, 0.05],
        "num_leaves": [15, 23, 31, 47, 63],
        "min_child_samples": [20, 30, 50, 80],
        "subsample": [0.7, 0.8, 0.9],
        "colsample_bytree": [0.6, 0.7, 0.8],
    }
    configs = []
    for _ in range(20):
        configs.append({k: random.choice(v) for k, v in space.items()})

    best = (-1, None)
    print("\n=== ランダム探索(選択指標: 2025三連単ドンピシャ的中率) ===")
    for i, p in enumerate(configs):
        m = fit_ranker(train, feats, p)
        hr = trifecta_hit_rate(m, val, feats)
        if hr > best[0]:
            best = (hr, p)
        print(f" [{i+1:2}/20] val三連単的中 {hr:.2f}%  {p}")
    print(f"\n最良: val {best[0]:.2f}%  params={best[1]}")

    # 最終評価
    base = joblib.load("models/ranker_v2.pkl")
    roi_base, h0, n0 = trifecta_roi(base, test, feats, tri_lut)
    m_best = fit_ranker(trainval, feats, best[1])
    roi_new, h1, n1 = trifecta_roi(m_best, test, feats, tri_lut)
    print(f"\n=== 2026 三連単1点ROI 比較 ===")
    print(f" 既存ranker_v2 : ROI {roi_base:.1f}% (的中{h0}/{n0})")
    print(f" 最適化版      : ROI {roi_new:.1f}% (的中{h1}/{n1})")

    # 時期安定性(2026前半/後半)
    test2 = test.copy(); test2["score"] = m_best.predict(test2[feats])
    mid = test2["date"].quantile(0.5)
    for lab, sub in [("2026前半", test2[test2.date <= mid]), ("2026後半", test2[test2.date > mid])]:
        r, h, n = trifecta_roi(m_best, sub, feats, tri_lut)
        print(f" {lab}: ROI {r:.1f}% (的中{h}/{n}) {'★黒字' if r>=100 else '(赤字)'}")

    if roi_new >= roi_base:
        joblib.dump(m_best, "models/ranker_v2.pkl")
        joblib.dump(feats, "models/ranker_v2_features.pkl")
        print("\n最適化版を ranker_v2.pkl に採用(全期間再学習済)。")
    else:
        print("\n既存版が優位のため据え置き。")


if __name__ == "__main__":
    main()
