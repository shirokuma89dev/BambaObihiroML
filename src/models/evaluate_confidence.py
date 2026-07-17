import os
import pandas as pd
import numpy as np
import joblib

def main():
    print("=== AIガチ予測：【超・鉄板レース厳選】確信度別 的中率検証 ===")
    
    data_path = "data/processed/features_train.csv"
    if not os.path.exists(data_path):
        print(f"エラー: {data_path} が見つかりません。")
        return
        
    df = pd.read_csv(data_path)
    df["date"] = pd.to_datetime(df["date"])
    
    # ターゲット生成
    df["is_rank1"] = (df["rank_num"] == 1).astype(int)
    
    feature_cols = [
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
        "humidity_avg_pct", "wind_avg_mps", "sunlight_hours", "snowfall_cm", "snow_depth_cm"
    ]
    weather_cols = [c for c in df.columns if c.startswith("weather_")]
    feature_cols.extend(weather_cols)
    
    df[feature_cols] = df[feature_cols].fillna(0)
    split_date = "2026-01-01"
    test_df = df[df["date"] >= split_date].copy()
    
    m1_cal = joblib.load("models/pos_m1.pkl")
    X_test = test_df[feature_cols]
    
    # AIの1着予測確率（＝確信度）
    test_df["ai_confidence"] = m1_cal.predict_proba(X_test)[:, 1]
    
    # レースごとに一番確信度が高い馬（AI本命馬）を抽出
    idx = test_df.groupby("race_id")["ai_confidence"].idxmax()
    top_picks = test_df.loc[idx].copy()
    
    total_races = len(top_picks)
    overall_win_rate = top_picks["is_rank1"].mean() * 100
    
    print(f"\n[ベースライン] 全 {total_races} レースに毎回賭けた場合の本命勝率: {overall_win_rate:.1f}%")
    print("-" * 60)
    
    # 確信度の閾値（足切りライン）を上げていく
    thresholds = [0.20, 0.30, 0.40, 0.50, 0.60]
    
    for th in thresholds:
        # AIが「勝率 th 以上」と絶対の自信を持ったレースのみ抽出
        strict_picks = top_picks[top_picks["ai_confidence"] >= th]
        n_races = len(strict_picks)
        if n_races == 0:
            continue
            
        win_rate = strict_picks["is_rank1"].mean() * 100
        frequency = n_races / total_races * 100
        
        print(f"【AI確信度 {th*100:.0f}% 以上 の ガチ鉄板レースのみ】")
        print(f"  -> 対象レース数 : {n_races} レース (全体の {frequency:.1f}%)")
        print(f"  -> 👑 1着ズバリ的中率 : ★ {win_rate:.1f} % ★")
        print("-" * 60)

if __name__ == "__main__":
    main()
