import os
import glob
import pandas as pd
import numpy as np
import lightgbm as lgb
from sklearn.metrics import roc_auc_score, accuracy_score, log_loss
import joblib

def main():
    print("=== STEP 1: 高度特徴量込みのモデル学習スクリプト ===")
    
    # 1. 加工済み特徴量データの読み込み
    data_path = "data/processed/features_train.csv"
    if not os.path.exists(data_path):
        print(f"エラー: 学習用特徴量データ {data_path} が見つかりません。")
        return
        
    df = pd.read_csv(data_path)
    df["date"] = pd.to_datetime(df["date"])
    print(f"データ読み込み完了: 全 {len(df)} 件")
    
    # 2. STEP 1 高度特徴量セットの設定
    feature_cols = [
        "sled_weight_num",           # そり重量(kg)
        "horse_body_weight",         # 馬体重(kg)
        "horse_weight_change",       # 馬体重増減(kg)
        "power_ratio",               # パワー負荷率
        # --- STEP 1 追加特徴量 ---
        "horse_body_weight_zscore",  # [新] レース内馬体重偏差値
        "power_ratio_zscore",        # [新] レース内パワー率偏差値
        "sled_weight_diff_max",      # [新] レース内最重量馬との斤量差
        "track_moisture_num",        # 馬場水分量(%)
        "horse_past_3_avg_rank",     # 直近3走の平均着順
        "horse_past_5_avg_rank",     # [新] 直近5走の平均着順
        "horse_rank_std",            # [新] 着順のばらつき（安定性）
        "horse_dry_avg_rank",        # [新] 重馬場（乾燥）での平均着順
        "horse_wet_avg_rank",        # [新] 軽馬場（泥）での平均着順
        "jockey_win_rate",           # 騎手通算勝率
        "jockey_top3_rate"           # 騎手通算3着内率
    ]
    
    # 天候One-Hotフラグ
    weather_cols = [c for c in df.columns if c.startswith("weather_")]
    feature_cols.extend(weather_cols)
    
    target_col = "is_top3"
    
    print(f"\n使用する特徴量数: {len(feature_cols)} 個")
    
    # 3. 時系列分割（Time Series Split）
    split_date = "2026-01-01"
    train_df = df[df["date"] < split_date].copy()
    test_df = df[df["date"] >= split_date].copy()
    
    print(f" - 学習データ: {len(train_df)} 件")
    print(f" - 検証データ (2026年): {len(test_df)} 件")
    
    X_train = train_df[feature_cols]
    y_train = train_df[target_col]
    X_test = test_df[feature_cols]
    y_test = test_df[target_col]
    
    # 4. LightGBM モデル学習
    params = {
        "objective": "binary",
        "metric": "binary_logloss",
        "boosting_type": "gbdt",
        "learning_rate": 0.03,        # 慎重に最適化
        "num_leaves": 45,             # 表現力を強化
        "min_child_samples": 20,
        "subsample": 0.8,
        "colsample_bytree": 0.8,
        "random_state": 42,
        "verbose": -1
    }
    
    model = lgb.LGBMClassifier(**params, n_estimators=500)
    model.fit(
        X_train, y_train,
        eval_set=[(X_test, y_test)],
        callbacks=[lgb.early_stopping(stopping_rounds=40, verbose=False)]
    )
    
    # 5. モデル検証
    y_pred_proba = model.predict_proba(X_test)[:, 1]
    y_pred = (y_pred_proba >= 0.5).astype(int)
    
    auc = roc_auc_score(y_test, y_pred_proba)
    acc = accuracy_score(y_test, y_pred)
    loss = log_loss(y_test, y_pred_proba)
    
    print("\n=== STEP 1 モデル検証結果 (2026年テストデータ) ===")
    print(f" - ROC-AUC (識別能力): {auc:.4f}  (前回: 0.6190)")
    print(f" - 正解率 (Accuracy):   {acc:.4f}  (前回: 0.6767)")
    print(f" - Log Loss (損失):    {loss:.4f}")
    
    # 6. 特徴量重要度
    importance_df = pd.DataFrame({
        "feature": feature_cols,
        "importance": model.feature_importances_
    }).sort_values(by="importance", ascending=False).reset_index(drop=True)
    
    print("\n=== STEP 1 特徴量重要度 TOP 10 ===")
    for idx, row in importance_df.head(10).iterrows():
        print(f" {idx+1:2d}. {row['feature']:<25} : {row['importance']}")
        
    model_dir = "models"
    os.makedirs(model_dir, exist_ok=True)
    model_file = os.path.join(model_dir, "banei_top3_model_step1.pkl")
    joblib.dump(model, model_file)
    print(f"\nモデル保存完了: {model_file}")

if __name__ == "__main__":
    main()
