import os
import itertools
import pandas as pd
import numpy as np
import lightgbm as lgb
from sklearn.calibration import CalibratedClassifierCV
import joblib

"""
Phase 1: 市場情報(人気順)の投入 + レース内確率正規化。
目標マイルストーン: 市場「1番人気機械買い」の1着的中率 35.75% を超える。
既存 train_position_models.py (24.80%) との直接比較用。
"""

FEATURE_COLS_BASE = [
    "sled_weight_num", "sled_weight_change", "sled_weight_zscore",
    "horse_body_weight", "horse_weight_change", "power_ratio",
    "horse_body_weight_zscore", "power_ratio_zscore", "sled_weight_diff_max",
    "track_moisture_num", "days_since_last_race", "class_level", "class_diff",
    "horse_avg_speed", "horse_max_speed", "speed_zscore", "momentum_score",
    "horse_cum_win_rate", "horse_cum_top3_rate", "pair_top3_rate",
    "trainer_win_rate", "trainer_top3_rate", "jt_pair_top3_rate",
    "horse_past_3_avg_rank", "horse_past_5_avg_rank", "horse_rank_std",
    "horse_past_3_avg_margin", "horse_best_time_sec", "horse_best_time_zscore",
    "horse_dry_avg_rank", "horse_wet_avg_rank", "track_specialist_factor",
    "jockey_win_rate", "jockey_top3_rate",
    "precip_total_mm", "temp_avg_c", "temp_max_c", "temp_min_c",
    "humidity_avg_pct", "wind_avg_mps", "sunlight_hours", "snowfall_cm", "snow_depth_cm",
    "power_moisture_interaction", "sled_weight_moisture_interaction",
    "jockey_upgrade_factor", "recent_form_score", "fatigue_index",
    "horse_elo_pre", "jockey_elo_pre", "horse_elo_zscore", "elo_gap_to_top",
    "horse_speed_figure", "horse_speed_figure_zscore",
]

# Phase 1 で追加する市場(人気)由来の特徴量
MARKET_COLS = ["popularity_num", "pop_is_fav", "pop_inv", "pop_zscore"]


def add_market_features(df):
    """人気順から市場特徴量を生成。学習・検証で共通利用可能(欠損<1%)。"""
    # 欠損は「最も人気薄」相当として大きめの値で補完
    df["popularity_num"] = df["popularity_num"].fillna(20.0)
    df["pop_is_fav"] = (df["popularity_num"] == 1).astype(int)
    df["pop_inv"] = 1.0 / df["popularity_num"]
    df["pop_zscore"] = df.groupby("race_id")["popularity_num"].transform(
        lambda x: (x - x.mean()) / (x.std() + 1e-6) if len(x) > 1 else 0.0
    )
    return df


def normalize_within_race(df, col):
    """レース内で確率を合計1に正規化(排他・被りなし構造を近似)。"""
    s = df.groupby("race_id")[col].transform("sum")
    return df[col] / (s + 1e-9)


def evaluate_assignment(test_df, tag):
    races = test_df["race_id"].unique()
    c1 = c2 = c3 = ct2 = ct3 = total = 0
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
        horses = r["umaban"].values
        p1s, p2s, p3s = r["p1n"].values, r["p2n"].values, r["p3n"].values
        best, pr1, pr2, pr3 = -1, None, None, None
        for i, j, k in itertools.permutations(range(len(horses)), 3):
            sc = p1s[i] * p2s[j] * p3s[k]
            if sc > best:
                best, pr1, pr2, pr3 = sc, horses[i], horses[j], horses[k]
        if pr1 == a1: c1 += 1
        if pr2 == a2: c2 += 1
        if pr3 == a3: c3 += 1
        if pr1 == a1 and pr2 == a2: ct2 += 1
        if pr1 == a1 and pr2 == a2 and pr3 == a3: ct3 += 1
    print(f"\n===== [{tag}] 検証結果 (2026年 {total}レース) =====")
    print(f" 1着 ドンピシャ的中率 : {c1/total*100:.2f} %   (市場1番人気=35.75%)")
    print(f" 2着 ドンピシャ的中率 : {c2/total*100:.2f} %")
    print(f" 3着 ドンピシャ的中率 : {c3/total*100:.2f} %")
    print(f" 馬単 完全ピタリ      : {ct2/total*100:.2f} %")
    print(f" 3連単 ズバリ         : {ct3/total*100:.2f} %")
    return c1 / total * 100


def main():
    print("=== Phase 1: 市場情報投入 + レース内正規化モデル (v2) ===")
    data_path = "data/processed/features_train.csv"
    df = pd.read_csv(data_path)
    df["date"] = pd.to_datetime(df["date"])
    df = add_market_features(df)

    df["is_rank1"] = (df["rank_num"] == 1).astype(int)
    df["is_rank2"] = (df["rank_num"] == 2).astype(int)
    df["is_rank3"] = (df["rank_num"] == 3).astype(int)

    weather_cols = [c for c in df.columns if c.startswith("weather_")]
    feature_cols = FEATURE_COLS_BASE + MARKET_COLS + weather_cols
    df[feature_cols] = df[feature_cols].fillna(0)

    split_date = "2026-01-01"
    train_df = df[df["date"] < split_date].copy()
    test_df = df[df["date"] >= split_date].copy()
    X_train, X_test = train_df[feature_cols], test_df[feature_cols]
    print(f"特徴量: {len(feature_cols)} 次元 | 学習: {len(X_train)} | 検証: {len(X_test)}")

    specs = [
        ("is_rank1", 600, 35, "p_rank1"),
        ("is_rank2", 500, 31, "p_rank2"),
        ("is_rank3", 400, 31, "p_rank3"),
    ]
    models = {}
    for target, n_est, leaves, pcol in specs:
        print(f"\n[{target}] 学習中...")
        base = lgb.LGBMClassifier(n_estimators=n_est, learning_rate=0.025, num_leaves=leaves,
                                  subsample=0.8, colsample_bytree=0.8, random_state=42, verbose=-1)
        cal = CalibratedClassifierCV(estimator=base, cv=3, method="sigmoid")
        cal.fit(X_train, train_df[target])
        test_df[pcol] = cal.predict_proba(X_test)[:, 1]
        models[pcol] = cal

    # レース内正規化あり/なしを比較
    for pcol, ncol in [("p_rank1", "p1n"), ("p_rank2", "p2n"), ("p_rank3", "p3n")]:
        test_df[ncol] = test_df[pcol]  # 正規化なし用に生値をコピー
    acc_raw = evaluate_assignment(test_df, "正規化なし(生確率)")

    for pcol, ncol in [("p_rank1", "p1n"), ("p_rank2", "p2n"), ("p_rank3", "p3n")]:
        test_df[ncol] = normalize_within_race(test_df, pcol)
    acc_norm = evaluate_assignment(test_df, "レース内正規化あり")

    print(f"\n>>> 1着的中: 旧24.80% → v2生 {acc_raw:.2f}% → v2正規化 {acc_norm:.2f}% (目標市場 35.75%)")

    model_dir = "models"
    os.makedirs(model_dir, exist_ok=True)
    for pcol, cal in models.items():
        joblib.dump(cal, os.path.join(model_dir, f"pos_v2_{pcol}.pkl"))
    print("v2モデル保存完了: models/pos_v2_*.pkl")


if __name__ == "__main__":
    main()
