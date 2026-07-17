"""
モデルの誠実な評価。2つの真実を提示する。

(1) 予測精度: ウォークフォワード(学習<年 → 検証=年)で 1着的中率 を計測し、
    「市場(1番人気ベタ買い)」と比較する。単一splitの偶然を排し、複数季節で安定して
    市場を超えるかを見る。→ 結論: ほぼ市場と互角(僅かに上回る季節あり)。

(2) 回収率(ROI): 2026年の実配当で、三連単1点勝負のROIを計測する。
    さらに前半/後半に分けて、見かけの黒字が「本物のエッジ」か「分散(運)」かを検証する。
    → 結論: 単一シーズンで黒字に見えても時期分割で消えることがあり、再現性ある+ROIは確認できない。
    競馬は控除率(約20%)の負期待値であり、これは手法改善では覆せない構造。

使い方: python src/models/evaluate.py
"""
import os
import sys
import pandas as pd
import numpy as np

sys.path.insert(0, os.path.dirname(__file__))
from train import load_dataset, fit_ranker, BEST_PARAMS, WALK_FORWARD_YEARS  # noqa: E402

TRIFECTA_PAYOUTS = "data/raw/banei_trifecta_payouts_2026.csv"


def _paynum(s):
    return float(str(s).replace("円", "").replace(",", "").strip())


def race_metrics(model, val_df, feats):
    """1着的中率・3連単ズバリ率・勝ち馬Top3内カバー率を返す。"""
    val_df = val_df.copy()
    val_df["score"] = model.predict(val_df[feats])
    t1 = t3 = cover = total = 0
    for _, r in val_df.groupby("race_id"):
        if len(r) < 3:
            continue
        total += 1
        order = list(r.sort_values("score", ascending=False)["umaban"].values)
        actual = list(r.sort_values("rank_num")["umaban"].values[:3])
        t1 += order[0] == actual[0]
        t3 += order[:3] == actual
        cover += actual[0] in set(order[:3])
    return dict(top1=t1 / total * 100, trifecta=t3 / total * 100,
                cover=cover / total * 100, races=total)


def market_top1(val_df):
    """1番人気を買い続けた場合の1着的中率。"""
    hit = total = 0
    for _, r in val_df.groupby("race_id"):
        if len(r) < 3 or not (r["popularity_num"] == 1).any():
            continue
        total += 1
        hit += r[r["popularity_num"] == 1].iloc[0]["rank_num"] == 1
    return hit / total * 100 if total else 0.0


def evaluate_accuracy(df, feats):
    print("=" * 62)
    print("  (1) 予測精度: ウォークフォワード検証 vs 市場(1番人気)")
    print("=" * 62)
    print(f"{'季節':>6} {'1着的中':>8} {'市場':>8} {'差':>7} {'3連単':>7} {'勝馬Top3内':>10}")
    for y in WALK_FORWARD_YEARS:
        tr, va = df[df["date"].dt.year < y], df[df["date"].dt.year == y]
        m = fit_ranker(tr, feats, BEST_PARAMS)
        met = race_metrics(m, va, feats)
        mk = market_top1(va)
        d = met["top1"] - mk
        print(f"{y:>6} {met['top1']:>7.2f}% {mk:>7.2f}% {d:>+6.2f} "
              f"{'★' if d > 0 else ' '} {met['trifecta']:>6.2f}% {met['cover']:>9.2f}%")
    print("→ 3季節とも市場とほぼ互角。効率的市場を明確に超えるのは不可能。\n")


def evaluate_roi(df, feats):
    print("=" * 62)
    print("  (2) 回収率(ROI): 三連単1点勝負 @ 2026実配当")
    print("=" * 62)
    if not os.path.exists(TRIFECTA_PAYOUTS):
        print(f"配当データが無いためスキップ: {TRIFECTA_PAYOUTS}")
        return
    pay = pd.read_csv(TRIFECTA_PAYOUTS)
    tri = pay[pay["ticket_type"] == "三連単"].copy()
    tri["pay"] = tri["payout"].apply(_paynum)
    lut = dict(zip(tri["race_id"].astype(str), zip(tri["combination"], tri["pay"])))

    tr = df[df["date"].dt.year < 2026]
    test = df[df["date"].dt.year == 2026].copy()
    model = fit_ranker(tr, feats, BEST_PARAMS)
    test["score"] = model.predict(test[feats])

    def roi(sub):
        staked = returned = hits = races = 0
        for rid, r in sub.groupby("race_id"):
            if len(r) < 3 or rid not in lut:
                continue
            races += 1
            staked += 100
            order = [int(x) for x in r.sort_values("score", ascending=False)["umaban"].values[:3]]
            wc, p = lut[rid]
            if "-".join(str(x) for x in order) == wc:
                returned += p
                hits += 1
        return returned / staked * 100 if staked else 0, hits, races

    r_all, h, n = roi(test)
    print(f" 2026通年 : ROI {r_all:.1f}% (的中 {h}/{n})  {'★黒字' if r_all >= 100 else '(赤字)'}")
    mid = test["date"].quantile(0.5)
    r1, h1, n1 = roi(test[test["date"] <= mid])
    r2, h2, n2 = roi(test[test["date"] > mid])
    print(f" 2026前半 : ROI {r1:.1f}% (的中 {h1}/{n1})  {'★黒字' if r1 >= 100 else '(赤字)'}")
    print(f" 2026後半 : ROI {r2:.1f}% (的中 {h2}/{n2})  {'★黒字' if r2 >= 100 else '(赤字)'}")
    print("→ 通年で黒字に見えても前半/後半で符号が割れるなら、それは分散(運)。")
    print("  再現性ある+ROIは確認できない。賭けは娯楽・少額で、期待値プラスとは考えない。\n")


def main():
    df, feats = load_dataset()
    evaluate_accuracy(df, feats)
    evaluate_roi(df, feats)


if __name__ == "__main__":
    main()
