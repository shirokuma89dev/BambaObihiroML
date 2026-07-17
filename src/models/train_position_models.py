import os
import glob
import itertools
import pandas as pd
import numpy as np
import lightgbm as lgb
from catboost import CatBoostClassifier
from xgboost import XGBClassifier
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import accuracy_score
import joblib

def main():
    print("=== ガチ性能 1着・2着・3着 個別最適化＆全着順最適割り当てAIモジュール (45次元爆発的改善版) ===")
    
    data_path = "data/processed/features_train.csv"
    if not os.path.exists(data_path):
        print(f"エラー: {data_path} が見つかりません。")
        return
        
    df = pd.read_csv(data_path)
    df["date"] = pd.to_datetime(df["date"])
    
    # ターゲット生成
    df["is_rank1"] = (df["rank_num"] == 1).astype(int)
    df["is_rank2"] = (df["rank_num"] == 2).astype(int)
    df["is_rank3"] = (df["rank_num"] == 3).astype(int)
    
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
    train_df = df[df["date"] < split_date].copy()
    test_df = df[df["date"] >= split_date].copy()
    
    X_train = train_df[feature_cols]
    X_test = test_df[feature_cols]
    
    print(f"特徴量: {len(feature_cols)} 次元 | 学習: {len(X_train)} 件 | 検証 (2026年): {len(X_test)} 件")
    
    # 【モデル1】 1着専用モデル
    print("\n[モデル1] 1着 (勝ち切る力) 専用AIモデルの学習...")
    base_m1 = lgb.LGBMClassifier(n_estimators=600, learning_rate=0.025, num_leaves=35, subsample=0.8, colsample_bytree=0.8, random_state=42, verbose=-1)
    m1_cal = CalibratedClassifierCV(estimator=base_m1, cv=3, method="sigmoid")
    m1_cal.fit(X_train, train_df["is_rank1"])
    
    # 【モデル2】 2着専用モデル
    print("[モデル2] 2着 (連対・粘り力) 専用AIモデルの学習...")
    base_m2 = lgb.LGBMClassifier(n_estimators=500, learning_rate=0.025, num_leaves=31, subsample=0.8, colsample_bytree=0.8, random_state=42, verbose=-1)
    m2_cal = CalibratedClassifierCV(estimator=base_m2, cv=3, method="sigmoid")
    m2_cal.fit(X_train, train_df["is_rank2"])
    
    # 【モデル3】 3着専用モデル
    print("[モデル3] 3着 (複勝圏・差し力) 専用AIモデルの学習...")
    base_m3 = lgb.LGBMClassifier(n_estimators=400, learning_rate=0.025, num_leaves=31, subsample=0.8, colsample_bytree=0.8, random_state=42, verbose=-1)
    m3_cal = CalibratedClassifierCV(estimator=base_m3, cv=3, method="sigmoid")
    m3_cal.fit(X_train, train_df["is_rank3"])
    
    test_df["p_rank1"] = m1_cal.predict_proba(X_test)[:, 1]
    test_df["p_rank2"] = m2_cal.predict_proba(X_test)[:, 1]
    test_df["p_rank3"] = m3_cal.predict_proba(X_test)[:, 1]
    
    # レース単位での1着・2着・3着最適割り当てアルゴリズム
    print("\n全 895 レースにおける 1着・2着・3着の組み合わせ最適化を実行中...")
    
    races = test_df["race_id"].unique()
    correct_1st = 0
    correct_2nd = 0
    correct_3rd = 0
    correct_exact_top2 = 0
    correct_exact_top3 = 0
    total_races = 0
    
    for r_id in races:
        r_df = test_df[test_df["race_id"] == r_id].copy()
        if len(r_df) < 3:
            continue
            
        total_races += 1
        
        # 実際の結果
        act_1 = r_df[r_df["rank_num"] == 1]["umaban"].values
        act_2 = r_df[r_df["rank_num"] == 2]["umaban"].values
        act_3 = r_df[r_df["rank_num"] == 3]["umaban"].values
        
        if len(act_1) == 0 or len(act_2) == 0 or len(act_3) == 0:
            continue
            
        a1_uma, a2_uma, a3_uma = act_1[0], act_2[0], act_3[0]
        
        # 被りのない最適組み合わせ (i=1着, j=2着, k=3着) を探す
        best_score = -1
        pred_1, pred_2, pred_3 = None, None, None
        
        horses = r_df["umaban"].values
        p1s = r_df["p_rank1"].values
        p2s = r_df["p_rank2"].values
        p3s = r_df["p_rank3"].values
        
        n_h = len(horses)
        for i, j, k in itertools.permutations(range(n_h), 3):
            score = p1s[i] * p2s[j] * p3s[k]
            if score > best_score:
                best_score = score
                pred_1, pred_2, pred_3 = horses[i], horses[j], horses[k]
                
        if pred_1 == a1_uma: correct_1st += 1
        if pred_2 == a2_uma: correct_2nd += 1
        if pred_3 == a3_uma: correct_3rd += 1
        if pred_1 == a1_uma and pred_2 == a2_uma: correct_exact_top2 += 1
        if pred_1 == a1_uma and pred_2 == a2_uma and pred_3 == a3_uma: correct_exact_top3 += 1
        
    print("\n=======================================================")
    print("  【爆発的改善版】1着・2着・3着 最適割り当て 検証結果 (2026年)")
    print("=======================================================")
    print(f" - 1着 ドンピシャ的中率: ★ {correct_1st / total_races * 100:.2f} % ★")
    print(f" - 2着 ドンピシャ的中率: ★ {correct_2nd / total_races * 100:.2f} % ★")
    print(f" - 3着 ドンピシャ的中率: ★ {correct_3rd / total_races * 100:.2f} % ★")
    print(f" - 1着-2着 馬単完全ピタリ的中率: ★ {correct_exact_top2 / total_races * 100:.2f} % ★")
    print(f" - 1着-2着-3着 3連単ズバリ的中率: ★ {correct_exact_top3 / total_races * 100:.2f} % ★")
    
    model_dir = "models"
    os.makedirs(model_dir, exist_ok=True)
    joblib.dump(m1_cal, os.path.join(model_dir, "pos_m1.pkl"))
    joblib.dump(m2_cal, os.path.join(model_dir, "pos_m2.pkl"))
    joblib.dump(m3_cal, os.path.join(model_dir, "pos_m3.pkl"))
    print("\n1着・2着・3着専用モデル(45次元版)保存完了: models/pos_m*.pkl")

if __name__ == "__main__":
    main()
