import os
import glob
import pandas as pd
import numpy as np
import lightgbm as lgb
import joblib

def main():
    print("=== 全30特徴量完全拡張版 ランキングモデル (LGBMRanker) 学習 ===")
    
    data_path = "data/processed/features_train.csv"
    if not os.path.exists(data_path):
        print(f"エラー: {data_path} が存在しません。")
        return
        
    df = pd.read_csv(data_path)
    df["date"] = pd.to_datetime(df["date"])
    
    def get_relevance(rank):
        if rank == 1:
            return 3
        elif rank == 2:
            return 2
        elif rank == 3:
            return 1
        else:
            return 0
            
    df["relevance"] = df["rank_num"].apply(get_relevance)
    
    feature_cols = [
        "sled_weight_num", "sled_weight_change", "horse_body_weight", "horse_weight_change", "power_ratio",
        "horse_body_weight_zscore", "power_ratio_zscore", "sled_weight_diff_max",
        "track_moisture_num", "days_since_last_race", "class_level", "class_diff",
        "horse_cum_win_rate", "horse_cum_top3_rate", "pair_top3_rate",
        "horse_past_3_avg_rank", "horse_past_5_avg_rank", "horse_rank_std",
        "horse_past_3_avg_margin", "horse_best_time_sec",
        "horse_dry_avg_rank", "horse_wet_avg_rank",
        "jockey_win_rate", "jockey_top3_rate"
    ]
    weather_cols = [c for c in df.columns if c.startswith("weather_")]
    feature_cols.extend(weather_cols)
    
    df[feature_cols] = df[feature_cols].fillna(0)
    
    split_date = "2026-01-01"
    train_df = df[df["date"] < split_date].copy()
    test_df = df[df["date"] >= split_date].copy()
    
    train_df = train_df.sort_values(by=["date", "race_id", "umaban"]).reset_index(drop=True)
    test_df = test_df.sort_values(by=["date", "race_id", "umaban"]).reset_index(drop=True)
    
    train_groups = train_df.groupby("race_id", sort=False).size().values
    test_groups = test_df.groupby("race_id", sort=False).size().values
    
    X_train = train_df[feature_cols]
    y_train = train_df["relevance"]
    X_test = test_df[feature_cols]
    y_test = test_df["relevance"]
    
    print(f"全特徴量数: {len(feature_cols)} 個 | 学習レース数: {len(train_groups)}, 検証: {len(test_groups)}")
    
    ranker = lgb.LGBMRanker(
        objective="lambdarank",
        metric="ndcg",
        eval_at=[1, 3],
        n_estimators=700,
        learning_rate=0.03,
        num_leaves=45,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42,
        verbose=-1
    )
    
    ranker.fit(
        X_train, y_train,
        group=train_groups,
        eval_set=[(X_test, y_test)],
        eval_group=[test_groups],
        callbacks=[lgb.early_stopping(stopping_rounds=50, verbose=False)]
    )
    
    model_dir = "models"
    os.makedirs(model_dir, exist_ok=True)
    joblib.dump(ranker, os.path.join(model_dir, "banei_ranker_pure_model.pkl"))
    print("最新ランキングモデル保存完了: models/banei_ranker_pure_model.pkl")

if __name__ == "__main__":
    main()
