import os
import glob
import pandas as pd
import numpy as np
import lightgbm as lgb
from sklearn.calibration import CalibratedClassifierCV
import joblib

def load_all_payouts(data_dir="data/raw"):
    payout_files = glob.glob(os.path.join(data_dir, "banei_race_payouts_*.csv"))
    if not payout_files:
        return pd.DataFrame()
    dfs = [pd.read_csv(f) for f in payout_files]
    return pd.concat(dfs, ignore_index=True)

def main():
    print("=== 性能進化版 モデル精度＆回収率評価モジュール ===")
    
    data_path = "data/processed/features_train.csv"
    model_path = "models/banei_ranker_pure_model.pkl"
    
    if not os.path.exists(data_path) or not os.path.exists(model_path):
        print("エラー: 処理データまたはモデルファイルが見つかりません。")
        return
        
    df = pd.read_csv(data_path)
    df["date"] = pd.to_datetime(df["date"])
    ranker = joblib.load(model_path)
    
    split_date = "2026-01-01"
    test_df = df[df["date"] >= split_date].copy()
    test_df = test_df.sort_values(by=["date", "race_id", "umaban"]).reset_index(drop=True)
    
    feature_cols = [
        "sled_weight_num", "horse_body_weight", "horse_weight_change", "power_ratio",
        "horse_body_weight_zscore", "power_ratio_zscore", "sled_weight_diff_max",
        "track_moisture_num", "class_level", "class_diff",
        "horse_past_3_avg_rank", "horse_past_5_avg_rank", "horse_rank_std",
        "horse_past_3_avg_margin", "horse_best_time_sec",
        "horse_dry_avg_rank", "horse_wet_avg_rank",
        "jockey_win_rate", "jockey_top3_rate"
    ]
    weather_cols = [c for c in df.columns if c.startswith("weather_")]
    feature_cols.extend(weather_cols)
    
    test_df[feature_cols] = test_df[feature_cols].fillna(0)
    
    X_test = test_df[feature_cols]
    test_df["ai_score"] = ranker.predict(X_test)
    test_df["ai_rank"] = test_df.groupby("race_id")["ai_score"].rank(ascending=False, method="min")
    
    print("\n--- 【1】進化版モデルによるAI本命馬＆カバー的中率 (2026年全 895 レース) ---")
    top1_win = test_df[test_df["ai_rank"] == 1]["is_win"].mean() * 100
    top1_fuku = test_df[test_df["ai_rank"] == 1]["is_top3"].mean() * 100
    top2_fuku = test_df[test_df["ai_rank"] <= 2].groupby("race_id")["is_top3"].max().mean() * 100
    top3_fuku = test_df[test_df["ai_rank"] <= 3].groupby("race_id")["is_top3"].max().mean() * 100
    top4_fuku = test_df[test_df["ai_rank"] <= 4].groupby("race_id")["is_top3"].max().mean() * 100
    
    print(f" - AI 第1推奨馬の単勝的中率 (1着的中): ★ {top1_win:.2f} % ★ (旧: 24.25 %)")
    print(f" - AI 第1推奨馬の複勝的中率 (3着内的中): ★ {top1_fuku:.2f} % ★ (旧: 50.06 %)")
    print(f" - AI 上位2頭カバー (Top 2) 複勝的中率: {top2_fuku:.2f} %")
    print(f" - AI 上位3頭カバー (Top 3) 複勝的中率: {top3_fuku:.2f} %")
    print(f" - AI 上位4頭カバー (Top 4) 複勝的中率: {top4_fuku:.2f} %")

    # 確信度（上位スコア差）別
    test_df["score_diff_2nd"] = test_df.groupby("race_id")["ai_score"].transform(lambda s: s.max() - s.nlargest(2).iloc[-1] if len(s) > 1 else 0)
    top1_df = test_df[test_df["ai_rank"] == 1].copy()
    
    print("\n--- 【2】AI確信度（1位と2位のスコア離れ度）別 第1推奨馬の精度 ---")
    for q in [0.50, 0.70, 0.85, 0.90]:
        thresh = top1_df["score_diff_2nd"].quantile(q)
        high_conf = top1_df[top1_df["score_diff_2nd"] >= thresh]
        win_acc = high_conf["is_win"].mean() * 100
        fuku_acc = high_conf["is_top3"].mean() * 100
        print(f" [確信度 上位{int((1-q)*100):2d}% (全 {len(high_conf)} レース)] 単勝的中率: ★ {win_acc:.2f} % ★ | 複勝的中率: ★ {fuku_acc:.2f} % ★")

if __name__ == "__main__":
    main()
