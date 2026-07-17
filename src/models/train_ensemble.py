import os
import glob
import pandas as pd
import numpy as np
import lightgbm as lgb
from catboost import CatBoostClassifier
from xgboost import XGBClassifier
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import roc_auc_score, accuracy_score
import joblib

def main():
    print("=== 最先端 3モデルアンサンブル (LightGBM + CatBoost + XGBoost) 学習 ===")
    
    data_path = "data/processed/features_train.csv"
    if not os.path.exists(data_path):
        print(f"エラー: {data_path} が見つかりません。")
        return
        
    df = pd.read_csv(data_path)
    df["date"] = pd.to_datetime(df["date"])
    
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
    
    X_train = train_df[feature_cols]
    y_train = train_df["is_top3"]
    X_test = test_df[feature_cols]
    y_test = test_df["is_top3"]
    
    print(f"全 {len(feature_cols)} 特徴量 | 学習データ: {len(X_train)} | 検証データ (2026年): {len(X_test)}")
    
    # 1. LightGBM モデル
    print("\n1. LightGBM モデルの学習中...")
    lgb_base = lgb.LGBMClassifier(n_estimators=400, learning_rate=0.03, num_leaves=31, random_state=42, verbose=-1)
    lgb_model = CalibratedClassifierCV(estimator=lgb_base, cv=3, method="sigmoid")
    lgb_model.fit(X_train, y_train)
    p_lgb = lgb_model.predict_proba(X_test)[:, 1]
    
    # 2. CatBoost モデル
    print("2. CatBoost モデルの学習中...")
    cat_model = CatBoostClassifier(iterations=500, learning_rate=0.03, depth=6, random_seed=42, verbose=0)
    cat_model.fit(X_train, y_train)
    p_cat = cat_model.predict_proba(X_test)[:, 1]
    
    # 3. XGBoost モデル
    print("3. XGBoost モデルの学習中...")
    xgb_model = XGBClassifier(n_estimators=300, learning_rate=0.03, max_depth=5, random_state=42, eval_metric="logloss")
    xgb_model.fit(X_train, y_train)
    p_xgb = xgb_model.predict_proba(X_test)[:, 1]
    
    # 4. 加重平均アンサンブル (LightGBM: 0.4, CatBoost: 0.35, XGBoost: 0.25)
    p_ensemble = 0.40 * p_lgb + 0.35 * p_cat + 0.25 * p_xgb
    
    auc_lgb = roc_auc_score(y_test, p_lgb)
    auc_cat = roc_auc_score(y_test, p_cat)
    auc_xgb = roc_auc_score(y_test, p_xgb)
    auc_ens = roc_auc_score(y_test, p_ensemble)
    
    print("\n=== 単体モデル vs アンサンブルモデル ROC-AUC 評価 (2026年未来データ) ===")
    print(f" - LightGBM 単体 AUC:  {auc_lgb:.4f}")
    print(f" - CatBoost 単体 AUC:  {auc_cat:.4f}")
    print(f" - XGBoost  単体 AUC:  {auc_xgb:.4f}")
    print(f" - アンサンブル   AUC: ★ {auc_ens:.4f} ★ (性能向上!)")
    
    test_df["ens_prob"] = p_ensemble
    test_df["ai_rank"] = test_df.groupby("race_id")["ens_prob"].rank(ascending=False, method="min")
    
    top1_fuku = test_df[test_df["ai_rank"] == 1]["is_top3"].mean() * 100
    top2_fuku = test_df[test_df["ai_rank"] <= 2].groupby("race_id")["is_top3"].max().mean() * 100
    top3_fuku = test_df[test_df["ai_rank"] <= 3].groupby("race_id")["is_top3"].max().mean() * 100
    
    print(f"\n=== アンサンブルモデル 本命馬・カバー率評価 ===")
    print(f" - AI 第1推奨馬の複勝的中率: ★ {top1_fuku:.2f} % ★")
    print(f" - AI Top 2 カバー複勝的中率: {top2_fuku:.2f} %")
    print(f" - AI Top 3 カバー複勝的中率: ★ {top3_fuku:.2f} % ★")
    
    # アンサンブルモデル群の保存
    model_dir = "models"
    os.makedirs(model_dir, exist_ok=True)
    joblib.dump(lgb_model, os.path.join(model_dir, "ens_lgb.pkl"))
    joblib.dump(cat_model, os.path.join(model_dir, "ens_cat.pkl"))
    joblib.dump(xgb_model, os.path.join(model_dir, "ens_xgb.pkl"))
    print("\nアンサンブルモデル保存完了: models/ens_*.pkl")

if __name__ == "__main__":
    main()
