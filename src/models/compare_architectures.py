import os
import json
import warnings
import pandas as pd
import numpy as np
import lightgbm as lgb
from sklearn.calibration import CalibratedClassifierCV

from train_position_models_v2 import add_market_features, FEATURE_COLS_BASE, MARKET_COLS
from optuna_walkforward import load, make_folds, fit_ranker, eval_metrics, market_top1

warnings.filterwarnings("ignore")

"""
アーキテクチャ比較 + 市場ブレンド。
ランカー / 1着分類器 / 市場(人気) を個別評価し、
「モデルと市場のブレンド」が市場単体のtop-1を超えるか(=市場に直交するスキルの有無)を
ウォークフォワードで検証。α(市場重み)をスイープ。
"""


def pctl(s):
    """レース内で高いほど1(良い)になる順位パーセンタイル。"""
    return s.rank(pct=True)


def score_top1(va, col):
    hit = tot = 0
    for _, r in va.groupby("race_id"):
        if len(r) < 3:
            continue
        tot += 1
        pick = r.loc[r[col].idxmax(), "umaban"]
        actual = r.loc[r["rank_num"].idxmin(), "umaban"]
        if pick == actual:
            hit += 1
    return hit / tot * 100 if tot else 0


def main():
    df, feats = load()
    folds = make_folds(df)

    params = {"n_estimators": 800, "learning_rate": 0.02, "num_leaves": 31,
              "min_child_samples": 30, "subsample": 0.7, "colsample_bytree": 0.6, "subsample_freq": 1}
    if os.path.exists("models/ranker_tuned_params.json"):
        params = json.load(open("models/ranker_tuned_params.json"))
        print("Optuna最適paramsを使用")

    alphas = [0.0, 0.2, 0.4, 0.5, 0.6, 0.7, 0.8, 1.0]  # 市場重み(1.0=市場のみ)
    rows = []
    for tr, va, y in folds:
        va = va.copy()
        rk = fit_ranker(tr, feats, params)
        va["s_rank"] = rk.predict(va[feats])
        # 1着分類器
        base = lgb.LGBMClassifier(n_estimators=600, learning_rate=0.025, num_leaves=35,
                                  subsample=0.8, colsample_bytree=0.8, random_state=42, verbose=-1)
        clf = CalibratedClassifierCV(estimator=base, cv=3, method="sigmoid")
        clf.fit(tr[feats], (tr["rank_num"] == 1).astype(int))
        va["s_clf"] = clf.predict_proba(va[feats])[:, 1]
        va["s_mkt"] = -va["popularity_num"]

        # レース内パーセンタイル化
        for c in ["s_rank", "s_clf", "s_mkt"]:
            va[c + "_p"] = va.groupby("race_id")[c].transform(pctl)

        # モデル(ランカー+分類器の平均)と市場をαでブレンド
        va["s_model_p"] = (va["s_rank_p"] + va["s_clf_p"]) / 2
        res = {"year": y, "market": market_top1(va),
               "ranker": score_top1(va, "s_rank"), "clf": score_top1(va, "s_clf")}
        for a in alphas:
            va["blend"] = a * va["s_mkt_p"] + (1 - a) * va["s_model_p"]
            res[f"a{a}"] = score_top1(va, "blend")
        rows.append(res)

    rep = pd.DataFrame(rows)
    print("\n=== 季節別 1着的中率 (%) ===")
    show = ["year", "market", "ranker", "clf"] + [f"a{a}" for a in alphas]
    print(rep[show].to_string(index=False, float_format=lambda x: f"{x:5.2f}"))

    print("\n=== 平均(3シーズン) ===")
    means = rep[[c for c in rep.columns if c != "year"]].mean()
    mkt = means["market"]
    for c in means.index:
        d = means[c] - mkt
        tag = "★市場超" if d > 0.01 else ("≒" if abs(d) <= 0.01 else "")
        lbl = {"market": "市場(1番人気)", "ranker": "ランカー単体", "clf": "分類器単体"}.get(c, f"ブレンド市場重み{c[1:]}")
        print(f"  {lbl:>22}: {means[c]:5.2f}%  ({d:+.2f}) {tag}")
    print("\n注: α=1.0は市場のみ, α=0.0はモデルのみ。市場超えの安定性を平均で判断。")


if __name__ == "__main__":
    main()
