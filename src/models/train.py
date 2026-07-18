"""
ばんえい競馬 着順予測モデルの学習。

このプロジェクトの最終モデルは LightGBM の LambdaRank(ランキング学習)。
「1レース内で全馬を着順の良い順に並べる」ことを直接最適化するため、
競馬の競争構造(1レースに勝ち馬は1頭・馬同士は排他)に適合する。

学習の考え方:
  - 目的変数 relevance = そのレースで打ち負かした頭数 (field_size - 着順)。大きいほど上位。
  - group  = レースごとの出走頭数。ランカーはこの単位で並べ替えを学習する。
  - ハイパラは Optuna を「複数シーズンのウォークフォワード」で最適化した値を採用
    (単一splitへの過学習を避け、未来データでの安定性を担保。--tune で再探索可能)。

使い方:
  python src/models/train.py            # 既定の最適パラメータで全期間学習し保存
  python src/models/train.py --tune 40  # Optunaで40試行チューニングしてから学習
"""
import os
import sys
import json
import argparse
import numpy as np
import pandas as pd
import lightgbm as lgb
import joblib

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "features"))
from build_features import model_feature_cols  # noqa: E402

DATA_PATH = "data/processed/features_train.csv"
MODEL_PATH = "models/banei_ranker.pkl"
FEATURES_PATH = "models/banei_ranker_features.pkl"
PARAMS_PATH = "models/banei_ranker_params.json"

# Optuna×ウォークフォワードで得た最適パラメータ(浅い木×強正則化で過学習を抑制)。
BEST_PARAMS = {
    "n_estimators": 300,
    "learning_rate": 0.00965377328885368,
    "num_leaves": 45,
    "min_child_samples": 38,
    "subsample": 0.8915046079722405,
    "colsample_bytree": 0.7396752184286884,
    "reg_alpha": 2.7698623616220774,
    "reg_lambda": 4.4317860991437215,
    "max_depth": 3,
    "subsample_freq": 1,
}

# ウォークフォワード検証の区切り(学習は各年より前、検証はその年)。
WALK_FORWARD_YEARS = [2024, 2025, 2026]


def load_dataset():
    """特徴量CSVを読み、ランキング学習用のラベルとメタ列を付けて返す。"""
    df = pd.read_csv(DATA_PATH)
    df["date"] = pd.to_datetime(df["date"])
    df["race_id"] = df["race_id"].astype(str)
    df = df.dropna(subset=["rank_num"]).copy()
    feats = model_feature_cols(df)
    df[feats] = df[feats].fillna(0)
    df["field_size"] = df.groupby("race_id")["rank_num"].transform("count")
    df["relevance"] = (df["field_size"] - df["rank_num"]).clip(lower=0).astype(int)
    return df, feats


def fit_ranker(train_df, feats, params):
    """レース単位(group)でLambdaRankを学習する。"""
    train_df = train_df.sort_values("race_id")
    groups = train_df.groupby("race_id", sort=True).size().values
    model = lgb.LGBMRanker(
        objective="lambdarank", metric="ndcg",
        random_state=42, verbose=-1, n_jobs=-1, **params,
    )
    model.fit(train_df[feats], train_df["relevance"], group=groups)
    return model


def _top1_accuracy(model, val_df, feats):
    """検証: レースごとにスコア最上位馬が実際に1着だった割合。"""
    val_df = val_df.copy()
    val_df["score"] = model.predict(val_df[feats])
    hit = total = 0
    for _, r in val_df.groupby("race_id"):
        if len(r) < 3:
            continue
        total += 1
        if r.loc[r["score"].idxmax(), "umaban"] == r.loc[r["rank_num"].idxmin(), "umaban"]:
            hit += 1
    return hit / total * 100 if total else 0.0


def walk_forward_tune(df, feats, n_trials):
    """複数シーズンのウォークフォワードで平均top-1的中率を最大化する(Optuna)。"""
    import optuna
    optuna.logging.set_verbosity(optuna.logging.WARNING)
    folds = [(df[df["date"].dt.year < y], df[df["date"].dt.year == y]) for y in WALK_FORWARD_YEARS]

    def objective(trial):
        params = dict(
            n_estimators=trial.suggest_int("n_estimators", 300, 1500, step=100),
            learning_rate=trial.suggest_float("learning_rate", 0.003, 0.05, log=True),
            num_leaves=trial.suggest_int("num_leaves", 15, 63),
            min_child_samples=trial.suggest_int("min_child_samples", 30, 150),
            subsample=trial.suggest_float("subsample", 0.7, 1.0), subsample_freq=1,
            colsample_bytree=trial.suggest_float("colsample_bytree", 0.5, 0.9),
            reg_alpha=trial.suggest_float("reg_alpha", 1e-2, 20.0, log=True),
            reg_lambda=trial.suggest_float("reg_lambda", 1e-2, 20.0, log=True),
            max_depth=trial.suggest_int("max_depth", 3, 6),
        )
        return float(np.mean([_top1_accuracy(fit_ranker(tr, feats, params), va, feats)
                              for tr, va in folds]))

    study = optuna.create_study(direction="maximize",
                                sampler=optuna.samplers.TPESampler(seed=42))
    study.optimize(objective, n_trials=n_trials, show_progress_bar=False)
    print(f"チューニング完了: 平均top1 {study.best_value:.2f}%")
    best = study.best_params
    best["subsample_freq"] = 1
    return best


def main():
    ap = argparse.ArgumentParser(description="ばんえい着順予測 LambdaRank 学習")
    ap.add_argument("--tune", type=int, default=0, metavar="N",
                    help="Optunaの試行回数(0=既定パラメータを使用)")
    args = ap.parse_args()

    df, feats = load_dataset()
    print(f"データ: {len(df)} 出走 / {df['race_id'].nunique()} レース / 特徴量 {len(feats)} 次元")

    params = walk_forward_tune(df, feats, args.tune) if args.tune > 0 else dict(BEST_PARAMS)
    print(f"使用パラメータ: {json.dumps(params, ensure_ascii=False)}")

    model = fit_ranker(df, feats, params)  # 全期間で最終学習
    os.makedirs("models", exist_ok=True)
    joblib.dump(model, MODEL_PATH)
    joblib.dump(feats, FEATURES_PATH)
    json.dump(params, open(PARAMS_PATH, "w", encoding="utf-8"), ensure_ascii=False, indent=2)
    print(f"保存完了: {MODEL_PATH}")


if __name__ == "__main__":
    main()
