import argparse
import glob
import os
import re
import itertools
import pandas as pd
import numpy as np
from catboost import CatBoostClassifier
from xgboost import XGBClassifier
import lightgbm as lgb
import joblib

def clean_numeric(val):
    if pd.isna(val):
        return np.nan
    val_str = str(val).strip()
    match = re.search(r"[-+]?\d*\.\d+|\d+", val_str)
    return float(match.group(0)) if match else np.nan

def extract_horse_weight(val):
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

def calculate_plackett_luce_trifecta(df_race):
    scores = df_race["ai_score"].values
    horses = df_race["umaban"].astype(int).values
    names = df_race["horse_name"].values
    
    v = np.exp(scores - np.max(scores))
    n = len(v)
    
    trifecta_results = []
    
    for i, j, k in itertools.permutations(range(n), 3):
        p_1st = v[i] / np.sum(v)
        v_remain_1 = np.delete(v, i)
        p_2nd = v[j] / np.sum(v_remain_1)
        v_remain_2 = np.delete(v, [min(i, j), max(i, j)])
        p_3rd = v[k] / np.sum(v_remain_2)
        
        p_trifecta = p_1st * p_2nd * p_3rd
        
        comb_str = f"{horses[i]}-{horses[j]}-{horses[k]}"
        comb_name = f"{names[i]} ➔ {names[j]} ➔ {names[k]}"
        
        trifecta_results.append({
            "combination": comb_str,
            "combination_name": comb_name,
            "uma1": horses[i],
            "uma2": horses[j],
            "uma3": horses[k],
            "prob": p_trifecta,
            "prob_pct": p_trifecta * 100
        })
        
    df_tri = pd.DataFrame(trifecta_results).sort_values(by="prob", ascending=False).reset_index(drop=True)
    return df_tri

def main():
    parser = argparse.ArgumentParser(description="ばんえい競馬AI 3連単・全着順精密予測モジュール")
    parser.add_argument("--card", type=str, default="", help="推論対象の出走表CSVパス")
    args = parser.parse_args()

    card_path = args.card
    if not card_path:
        cards = sorted(glob.glob("data/raw/banei_race_card_*.csv"))
        if cards:
            card_path = cards[-1]
        else:
            print("エラー: 出走表データが見つかりません。")
            return

    print(f"=== ばんえい競馬AI 3連単・着順順位精密予想 (対象: {card_path}) ===")
    df = pd.read_csv(card_path)

    df["sled_weight_num"] = df["sled_weight"].apply(clean_numeric)
    df["track_moisture_num"] = df["track_moisture"].apply(clean_numeric)
    df["odds_num"] = pd.to_numeric(df["odds"], errors="coerce").fillna(999.0)

    hw_tuples = df["horse_weight"].apply(extract_horse_weight)
    df["horse_body_weight"] = [t[0] for t in hw_tuples]
    df["horse_weight_change"] = [t[1] for t in hw_tuples]
    df["power_ratio"] = df["sled_weight_num"] / df["horse_body_weight"]

    df["horse_body_weight_zscore"] = df.groupby("race_id")["horse_body_weight"].transform(
        lambda x: (x - x.mean()) / (x.std() + 1e-6) if len(x) > 1 else 0
    )
    df["power_ratio_zscore"] = df.groupby("race_id")["power_ratio"].transform(
        lambda x: (x - x.mean()) / (x.std() + 1e-6) if len(x) > 1 else 0
    )
    df["sled_weight_diff_max"] = df.groupby("race_id")["sled_weight_num"].transform(
        lambda x: x - x.max()
    )

    def extract_class_level(race_name):
        if pd.isna(race_name): return 0
        name = str(race_name)
        if "オープン" in name or "重賞" in name or "ばんえい記念" in name: return 5
        elif "Ａ" in name or "A" in name: return 4
        elif "Ｂ" in name or "B" in name: return 3
        elif "Ｃ" in name or "C" in name: return 2
        elif "２歳" in name or "3歳" in name or "３歳" in name: return 1
        return 0

    df["class_level"] = df["race_name"].apply(extract_class_level)

    train_path = "data/processed/features_train.csv"
    if os.path.exists(train_path):
        hist_df = pd.read_csv(train_path)
        latest_horse = hist_df.sort_values("date").groupby("horse_name").last().reset_index()
        h_cols = ["horse_name", "class_level", "sled_weight_num", "date",
                  "horse_cum_win_rate", "horse_cum_top3_rate", "horse_past_3_avg_rank",
                  "horse_past_5_avg_rank", "horse_rank_std", "horse_past_3_avg_margin",
                  "horse_best_time_sec", "horse_dry_avg_rank", "horse_wet_avg_rank"]
        
        latest_sub = latest_horse[h_cols].rename(columns={
            "class_level": "prev_class_level",
            "sled_weight_num": "prev_sled_weight",
            "date": "prev_date"
        })
        df = df.merge(latest_sub, on="horse_name", how="left")
        
        df["class_diff"] = (df["class_level"] - df["prev_class_level"]).fillna(0)
        df["sled_weight_change"] = (df["sled_weight_num"] - df["prev_sled_weight"]).fillna(0)
        df["days_since_last_race"] = 14.0

        latest_jockey = hist_df.groupby("jockey_name").agg(
            jockey_win_rate=("is_win", "mean"),
            jockey_top3_rate=("is_top3", "mean")
        ).reset_index()
        df = df.merge(latest_jockey, on="jockey_name", how="left")
        df["pair_top3_rate"] = df["horse_cum_top3_rate"]

    df["horse_past_3_avg_rank"] = df["horse_past_3_avg_rank"].fillna(4.5)
    df["horse_past_5_avg_rank"] = df["horse_past_5_avg_rank"].fillna(4.5)
    df["horse_rank_std"] = df["horse_rank_std"].fillna(1.5)
    df["horse_past_3_avg_margin"] = df["horse_past_3_avg_margin"].fillna(0.0)
    df["horse_best_time_sec"] = df["horse_best_time_sec"].fillna(150.0)
    df["horse_dry_avg_rank"] = df["horse_dry_avg_rank"].fillna(4.5)
    df["horse_wet_avg_rank"] = df["horse_wet_avg_rank"].fillna(4.5)
    df["jockey_win_rate"] = df["jockey_win_rate"].fillna(0.10)
    df["jockey_top3_rate"] = df["jockey_top3_rate"].fillna(0.30)
    df["horse_cum_win_rate"] = df["horse_cum_win_rate"].fillna(0.10)
    df["horse_cum_top3_rate"] = df["horse_cum_top3_rate"].fillna(0.30)
    df["pair_top3_rate"] = df["pair_top3_rate"].fillna(0.30)

    xgb_m = joblib.load("models/ens_xgb.pkl")
    model_feature_names = xgb_m.get_booster().feature_names

    for col in model_feature_names:
        if col not in df.columns:
            if col.startswith("weather_"):
                w_name = col.replace("weather_", "")
                df[col] = (df["weather"] == w_name).astype(int)
            else:
                df[col] = 0.0

    X_predict = df[model_feature_names].fillna(0)

    lgb_m = joblib.load("models/ens_lgb.pkl")
    cat_m = joblib.load("models/ens_cat.pkl")

    p_lgb = lgb_m.predict_proba(X_predict)[:, 1]
    p_cat = cat_m.predict_proba(X_predict)[:, 1]
    p_xgb = xgb_m.predict_proba(X_predict)[:, 1]

    df["ens_prob"] = 0.40 * p_lgb + 0.35 * p_cat + 0.25 * p_xgb
    df["ai_score"] = np.log(df["ens_prob"] / (1.0 - df["ens_prob"]))
    df["ai_rank"] = df.groupby("race_id")["ai_score"].rank(ascending=False, method="min")

    races = df["race_no"].unique()
    print("\n=======================================================")
    print("      ばんえい競馬AI 3連単・全着順予測モデル")
    print("=======================================================")

    for r_no in sorted(races):
        r_df = df[df["race_no"] == r_no].sort_values(by="ai_rank").copy()
        r_name = r_df["race_name"].iloc[0] if "race_name" in r_df.columns else ""
        print(f"\n-------------------------------------------------------")
        print(f"  第 {r_no:2d} レース : {r_name}")
        print(f"-------------------------------------------------------")

        print(" 【AI予測着順ランキング (全頭順位予測)】")
        for idx, row in r_df.iterrows():
            print(f"   第{int(row['ai_rank'])}位 : 馬番 {int(row['umaban']):2d} | {row['horse_name']:<12} (能力スコア: {row['ai_score']:.3f})")

        df_tri = calculate_plackett_luce_trifecta(r_df)

        top1_tri = df_tri.iloc[0]
        top3_umaban = r_df[r_df["ai_rank"] <= 3]["umaban"].astype(int).tolist()
        box6_df = df_tri[df_tri["uma1"].isin(top3_umaban) & df_tri["uma2"].isin(top3_umaban) & df_tri["uma3"].isin(top3_umaban)]
        box6_prob = box6_df["prob_pct"].sum()

        print(f"\n 🎯 3連単 AI本命1点 : 馬番 【 {top1_tri['combination']} 】 (1点的中確率: {top1_tri['prob_pct']:.2f}% | 完全ランダム比: {top1_tri['prob_pct']/0.298:.1f}倍)")
        print(f" 📦 AI Top3 (馬番 {top3_umaban}) 3連単6点BOX : 合計的中確率 ★ {box6_prob:.2f}% ★ (期待オッズ 150〜500倍で高回収)")

        print(f"\n 👑 3連単 厳選高確率買い目 TOP 5 👑")
        for rank_idx, tri_row in df_tri.head(5).iterrows():
            print(f"   {rank_idx+1}位 : 馬番 {tri_row['combination']:<8} (1点確率: {tri_row['prob_pct']:5.2f}%) | {tri_row['combination_name']}")

if __name__ == "__main__":
    main()
