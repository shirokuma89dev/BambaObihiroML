import pandas as pd
import numpy as np
import joblib
from itertools import permutations

from train_position_models_v2 import add_market_features

"""
三連単の真のROI検証。ランカー(models/ranker_v2.pkl)の順位スコアで買い目を組み、
実際の三連単配当(banei_trifecta_payouts_2026.csv)と照合。
市場が最も非効率な高配当プールで、モデルの優位が利益になるかを判定。
"""
STAKE = 100.0


def paynum(s):
    return float(str(s).replace("円", "").replace(",", "").strip())


def main():
    pay = pd.read_csv("data/raw/banei_trifecta_payouts_2026.csv")
    tri = pay[pay["ticket_type"] == "三連単"].copy()
    tri["pay"] = tri["payout"].apply(paynum)
    tri_lut = dict(zip(tri["race_id"].astype(str), zip(tri["combination"], tri["pay"])))
    print(f"三連単 配当データ: {len(tri_lut)} レース")

    df = pd.read_csv("data/processed/features_train.csv")
    df["date"] = pd.to_datetime(df["date"])
    df = add_market_features(df)
    feats = joblib.load("models/ranker_v2_features.pkl")
    df[feats] = df[feats].fillna(0)
    test = df[df["date"] >= "2026-01-01"].copy()
    ranker = joblib.load("models/ranker_v2.pkl")
    test["score"] = ranker.predict(test[feats])
    test["race_id"] = test["race_id"].astype(str)

    def bet_straight(order):        # 1点: top1-2-3
        return ["-".join(str(int(x)) for x in order[:3])]

    def bet_formation_1x3x3(order):  # 1着=top1固定, 2-3着=top2..4
        first = int(order[0]); pool = [int(x) for x in order[1:4]]
        return [f"{first}-{a}-{b}" for a in pool for b in pool if a != b]

    def bet_box4(order):            # top4 ボックス(24点)
        p = [int(x) for x in order[:4]]
        return ["-".join(str(x) for x in c) for c in permutations(p, 3)]

    def bet_box5(order):            # top5 ボックス(60点)
        p = [int(x) for x in order[:5]]
        return ["-".join(str(x) for x in c) for c in permutations(p, 3)]

    strategies = [
        ("三連単 1点(top1-2-3)", bet_straight),
        ("三連単 フォーメーション1x3x3(6点)", bet_formation_1x3x3),
        ("三連単 top4ボックス(24点)", bet_box4),
        ("三連単 top5ボックス(60点)", bet_box5),
    ]

    for name, fn in strategies:
        staked = returned = hits = races = 0
        for rid, r in test.groupby("race_id"):
            if len(r) < 3 or rid not in tri_lut:
                continue
            races += 1
            order = r.sort_values("score", ascending=False)["umaban"].values
            bets = fn(order)
            staked += STAKE * len(bets)
            win_combo, p = tri_lut[rid]
            if win_combo in bets:
                returned += p
                hits += 1
        roi = returned / staked * 100 if staked else 0
        hr = hits / races * 100 if races else 0
        flag = "★黒字★" if roi >= 100 else "(赤字)"
        print(f"\n[{name}]  {races}レース")
        print(f"  投資 {staked:,.0f}円 / 払戻 {returned:,.0f}円 / 的中 {hits}本 ({hr:.1f}%)")
        print(f"  回収率(ROI): {roi:.1f}% {flag}")


if __name__ == "__main__":
    main()
