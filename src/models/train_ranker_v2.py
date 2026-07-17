import os
import itertools
import pandas as pd
import numpy as np
import lightgbm as lgb
import joblib

from train_position_models_v2 import FEATURE_COLS_BASE, MARKET_COLS, add_market_features

"""
Phase 2: LambdaRank による競争構造直接学習。
レース内の全馬を「着順」で相対ランキング最適化 -> top-1 を直接狙う。
独立二値分類3本(24.80% -> 35.14%)を、構造的に正しいランカーで超える。
目標: 市場1番人気 35.75% を明確に上回る。
"""


def evaluate_ranker(test_df, score_col, tag):
    """スコア上位を 1着/2着/3着 とみなして的中を測定。"""
    races = test_df["race_id"].unique()
    c1 = c2 = c3 = ct2 = ct3 = cover3 = total = 0
    for r_id in races:
        r = test_df[test_df["race_id"] == r_id]
        if len(r) < 3:
            continue
        act1 = r[r["rank_num"] == 1]["umaban"].values
        act2 = r[r["rank_num"] == 2]["umaban"].values
        act3 = r[r["rank_num"] == 3]["umaban"].values
        if len(act1) == 0 or len(act2) == 0 or len(act3) == 0:
            continue
        total += 1
        a1, a2, a3 = act1[0], act2[0], act3[0]
        order = r.sort_values(score_col, ascending=False)["umaban"].values
        p1, p2, p3 = order[0], order[1], order[2]
        top3 = set(order[:3])
        if p1 == a1: c1 += 1
        if p2 == a2: c2 += 1
        if p3 == a3: c3 += 1
        if p1 == a1 and p2 == a2: ct2 += 1
        if p1 == a1 and p2 == a2 and p3 == a3: ct3 += 1
        if a1 in top3: cover3 += 1
    print(f"\n===== [{tag}] 検証結果 (2026年 {total}レース) =====")
    print(f" 1着 ドンピシャ的中率 : {c1/total*100:.2f} %   (市場1番人気=35.75%)")
    print(f" 2着 ドンピシャ的中率 : {c2/total*100:.2f} %")
    print(f" 3着 ドンピシャ的中率 : {c3/total*100:.2f} %")
    print(f" 馬単 完全ピタリ      : {ct2/total*100:.2f} %")
    print(f" 3連単 ズバリ         : {ct3/total*100:.2f} %")
    print(f" 勝ち馬 Top3内カバー率 : {cover3/total*100:.2f} %")
    return c1 / total * 100


def main():
    print("=== Phase 2: LambdaRank 競争構造学習モデル ===")
    df = pd.read_csv("data/processed/features_train.csv")
    df["date"] = pd.to_datetime(df["date"])
    df = add_market_features(df)

    weather_cols = [c for c in df.columns if c.startswith("weather_")]
    feature_cols = FEATURE_COLS_BASE + MARKET_COLS + weather_cols
    df[feature_cols] = df[feature_cols].fillna(0)

    # 着順が有効な行のみ(ランク学習にラベルが必要)
    df = df.dropna(subset=["rank_num"]).copy()
    # レース内で「打ち負かした頭数」= relevance (大きいほど上位)
    df["field_size"] = df.groupby("race_id")["rank_num"].transform("count")
    df["relevance"] = (df["field_size"] - df["rank_num"]).clip(lower=0).astype(int)

    split_date = "2026-01-01"
    train_df = df[df["date"] < split_date].sort_values("race_id").copy()
    test_df = df[df["date"] >= split_date].copy()

    X_train = train_df[feature_cols]
    y_train = train_df["relevance"]
    group_train = train_df.groupby("race_id", sort=True).size().values

    print(f"特徴量: {len(feature_cols)} 次元 | 学習: {len(X_train)} ({len(group_train)}レース) | 検証: {len(test_df)}")

    ranker = lgb.LGBMRanker(
        objective="lambdarank",
        metric="ndcg",
        n_estimators=800,
        learning_rate=0.02,
        num_leaves=31,
        subsample=0.8,
        colsample_bytree=0.8,
        min_child_samples=30,
        random_state=42,
        verbose=-1,
    )
    ranker.fit(X_train, y_train, group=group_train)

    test_df["rank_score"] = ranker.predict(test_df[feature_cols])
    acc = evaluate_ranker(test_df, "rank_score", "LambdaRank (全特徴量)")

    # 市場(人気)単体ベースラインも同じ評価系で比較
    test_df["pop_score"] = -test_df["popularity_num"]  # 人気が高い(数字が小さい)ほど高スコア
    evaluate_ranker(test_df, "pop_score", "参考: 市場人気順そのまま")

    print(f"\n>>> 1着的中: 旧24.80% → v2分類35.14% → ランカー {acc:.2f}% (目標市場 35.75%)")

    joblib.dump(ranker, "models/ranker_v2.pkl")
    joblib.dump(feature_cols, "models/ranker_v2_features.pkl")
    print("ランカー保存完了: models/ranker_v2.pkl")


if __name__ == "__main__":
    main()
