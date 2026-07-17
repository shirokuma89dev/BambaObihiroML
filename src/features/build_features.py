import argparse
import glob
import os
import re
import pandas as pd
import numpy as np

def clean_numeric(val):
    if pd.isna(val):
        return np.nan
    val_str = str(val).strip()
    match = re.search(r"[-+]?\d*\.\d+|\d+", val_str)
    return float(match.group(0)) if match else np.nan

def extract_horse_weight(val):
    """'1015(2)' から 馬体重 1015.0 と 体重増減 2.0 を抽出"""
    if pd.isna(val):
        return np.nan, np.nan
    val_str = str(val).strip()
    match = re.search(r"(\d{3,4})\s*\(([+-]?\d+)\)", val_str)
    if match:
        return float(match.group(1)), float(match.group(2))
    match_only = re.search(r"(\d{3,4})", val_str)
    if match_only:
        return float(match_only.group(1)), 0.0
    return np.nan, np.nan

def process_raw_data(data_dir="data/raw", out_dir="data/processed"):
    os.makedirs(out_dir, exist_ok=True)
    
    result_files = glob.glob(os.path.join(data_dir, "banei_race_results_*.csv"))
    if not result_files:
        print(f"エラー: {data_dir} にレース結果CSVが存在しません。")
        return None
        
    dfs = [pd.read_csv(f) for f in result_files]
    df = pd.concat(dfs, ignore_index=True)
    
    print(f"--- 生データ処理開始: 全 {len(df)} 件 ---")
    
    # 1. 数値前処理
    df["rank_num"] = df["rank"].apply(clean_numeric)
    df["is_win"] = (df["rank_num"] == 1).astype(int)
    df["is_top3"] = (df["rank_num"] <= 3).astype(int)
    
    df["sled_weight_num"] = df["sled_weight"].apply(clean_numeric)
    df["track_moisture_num"] = df["track_moisture"].apply(clean_numeric)
    df["odds_num"] = df["odds"].apply(clean_numeric)
    df["popularity_num"] = df["popularity"].apply(clean_numeric)
    
    # 馬体重抽出
    hw_tuples = df["horse_weight"].apply(extract_horse_weight)
    df["horse_body_weight"] = [t[0] for t in hw_tuples]
    df["horse_weight_change"] = [t[1] for t in hw_tuples]
    
    # パワー負荷率（そり重量 / 馬体重）
    df["power_ratio"] = df["sled_weight_num"] / df["horse_body_weight"]
    
    # 日付昇順ソート
    df["date"] = pd.to_datetime(df["date"])
    df = df.sort_values(by=["date", "race_no", "umaban"]).reset_index(drop=True)
    
    # 2. ローリング過去成績・集計特徴量（リーク回避のためshift）
    print("特徴量エンジニアリング（ローリング特徴量・集計値計算）中...")
    
    # 馬ごとの直近3走平均着順
    df["horse_past_3_avg_rank"] = df.groupby("horse_name")["rank_num"].transform(
        lambda s: s.shift(1).rolling(3, min_periods=1).mean()
    )
    
    # 騎手の勝率・3着内率
    jockey_stats = df.groupby("jockey_name").agg(
        jockey_win_rate=("is_win", "mean"),
        jockey_top3_rate=("is_top3", "mean")
    ).reset_index()
    
    df = df.merge(jockey_stats, on="jockey_name", how="left")
    
    # 出力カラム整理
    features_df = df[[
        "race_id", "date", "race_no", "race_name", "umaban", "waku",
        "horse_name", "sex_age", "jockey_name", "trainer_name",
        "sled_weight_num", "horse_body_weight", "horse_weight_change", "power_ratio",
        "track_moisture_num", "popularity_num", "odds_num",
        "horse_past_3_avg_rank", "jockey_win_rate", "jockey_top3_rate",
        "is_win", "is_top3", "rank_num"
    ]].copy()
    
    out_path = os.path.join(out_dir, "features_train.csv")
    features_df.to_csv(out_path, index=False, encoding="utf-8-sig")
    print(f"完了: モデル学習用特徴量データ生成 -> {out_path} (全 {len(features_df)} 件)")
    return out_path

def main():
    parser = argparse.ArgumentParser(description="帯広（ばんえい）データ前処理・特徴量自動生成モジュール")
    parser.add_argument("--data-dir", type=str, default="data/raw", help="生データ入力ディレクトリ")
    parser.add_argument("--out-dir", type=str, default="data/processed", help="特徴量出力先ディレクトリ")
    
    args = parser.parse_args()
    process_raw_data(data_dir=args.data_dir, out_dir=args.out_dir)

if __name__ == "__main__":
    main()
