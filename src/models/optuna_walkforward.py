import json
import argparse
import warnings
import pandas as pd
import numpy as np
import lightgbm as lgb
import optuna
import joblib

from train_position_models_v2 import add_market_features, FEATURE_COLS_BASE, MARKET_COLS

warnings.filterwarnings("ignore")
optuna.logging.set_verbosity(optuna.logging.WARNING)

"""
A案: 精度を極限まで追求。
ランカーのハイパラを Optuna で最適化。ただし単一splitへの過学習を避けるため、
複数シーズンのウォークフォワード(2024/2025/2026)で平均top-1的中率を最大化する。
選択は各foldの「学習より未来の検証期間」のみで行い、test汚染を排除。
"""

FOLDS = [
    ("2024-01-01", "2024"),  # 学習: <2024, 検証: 2024年
    ("2025-01-01", "2025"),  # 学習: <2025, 検証: 2025年
    ("2026-01-01", "2026"),  # 学習: <2026, 検証: 2026年
]


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


def make_folds(df):
    out = []
    for split, year in FOLDS:
        tr = df[df["date"] < split]
        va = df[(df["date"].dt.year == int(year))]
        out.append((tr, va, year))
    return out


def fit_ranker(tr, feats, params):
    tr = tr.sort_values("race_id")
    grp = tr.groupby("race_id", sort=True).size().values
    m = lgb.LGBMRanker(objective="lambdarank", metric="ndcg",
                       random_state=42, verbose=-1, n_jobs=-1, **params)
    m.fit(tr[feats], tr["rel"], group=grp)
    return m


def eval_metrics(model, va, feats):
    va = va.copy()
    va["score"] = model.predict(va[feats])
    t1 = t3exact = cover = tot = 0
    for _, r in va.groupby("race_id"):
        if len(r) < 3:
            continue
        tot += 1
        order = list(r.sort_values("score", ascending=False)["umaban"].values)
        actual = list(r.sort_values("rank_num")["umaban"].values[:3])
        if order[0] == actual[0]:
            t1 += 1
        if order[:3] == actual:
            t3exact += 1
        if actual[0] in set(order[:3]):
            cover += 1
    return dict(top1=t1 / tot * 100, top3exact=t3exact / tot * 100,
                cover=cover / tot * 100, races=tot)


def market_top1(va):
    """その季節に1番人気を買い続けた時の1着的中率。"""
    hit = tot = 0
    for _, r in va.groupby("race_id"):
        if len(r) < 3 or not (r["popularity_num"] == 1).any():
            continue
        tot += 1
        fav = r[r["popularity_num"] == 1].iloc[0]
        if fav["rank_num"] == 1:
            hit += 1
    return hit / tot * 100 if tot else 0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--trials", type=int, default=40)
    args = ap.parse_args()

    df, feats = load()
    folds = make_folds(df)
    for _, va, y in folds:
        print(f"fold {y}: 学習外検証 {va['race_id'].nunique()} レース")

    def objective(trial):
        params = dict(
            n_estimators=trial.suggest_int("n_estimators", 300, 1400, step=100),
            learning_rate=trial.suggest_float("learning_rate", 0.005, 0.08, log=True),
            num_leaves=trial.suggest_int("num_leaves", 15, 127),
            min_child_samples=trial.suggest_int("min_child_samples", 10, 120),
            subsample=trial.suggest_float("subsample", 0.6, 1.0),
            subsample_freq=1,
            colsample_bytree=trial.suggest_float("colsample_bytree", 0.5, 1.0),
            reg_alpha=trial.suggest_float("reg_alpha", 1e-3, 10.0, log=True),
            reg_lambda=trial.suggest_float("reg_lambda", 1e-3, 10.0, log=True),
            max_depth=trial.suggest_int("max_depth", 3, 14),
        )
        scores = []
        for tr, va, _ in folds:
            m = fit_ranker(tr, feats, params)
            scores.append(eval_metrics(m, va, feats)["top1"])
        return float(np.mean(scores))

    study = optuna.create_study(direction="maximize",
                                sampler=optuna.samplers.TPESampler(seed=42))
    study.optimize(objective, n_trials=args.trials, show_progress_bar=False)

    best = study.best_params
    best["subsample_freq"] = 1
    print(f"\n最良 平均top1: {study.best_value:.2f}%")
    print(f"最良params: {json.dumps(best, ensure_ascii=False)}")

    # 最終・季節別レポート(市場と比較)
    print("\n" + "=" * 66)
    print("  最適モデル 季節別精度 vs 市場(1番人気ベタ買い)")
    print("=" * 66)
    print(f"{'季節':>6} {'1着的中':>8} {'市場1番人気':>10} {'差':>7} {'3連単ズバリ':>10} {'勝馬Top3内':>10}")
    fold_models = {}
    for tr, va, y in folds:
        m = fit_ranker(tr, feats, best)
        fold_models[y] = m
        met = eval_metrics(m, va, feats)
        mk = market_top1(va)
        diff = met["top1"] - mk
        mark = "★超" if diff > 0 else "×負"
        print(f"{y:>6} {met['top1']:>7.2f}% {mk:>9.2f}% {diff:>+6.2f} {mark} {met['top3exact']:>8.2f}% {met['cover']:>9.2f}%")

    # 本番用: 全期間(〜2026)で再学習して保存
    full = fit_ranker(df, feats, best)
    joblib.dump(full, "models/ranker_tuned.pkl")
    joblib.dump(feats, "models/ranker_tuned_features.pkl")
    with open("models/ranker_tuned_params.json", "w", encoding="utf-8") as f:
        json.dump(best, f, ensure_ascii=False, indent=2)
    print("\n保存: models/ranker_tuned.pkl (全期間学習) / *_params.json")


if __name__ == "__main__":
    main()
