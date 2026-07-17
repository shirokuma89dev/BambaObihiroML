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

def run_backtest_sanrenfuku():
    print("\n=== 2026年全 895 レース 3連複 実際的中率バックテスト ===")
    data_path = "data/processed/features_train.csv"
    if not os.path.exists(data_path):
        return
        
    df = pd.read_csv(data_path)
    df["date"] = pd.to_datetime(df["date"])
    
    split_date = "2026-01-01"
    test_df = df[df["date"] >= split_date].copy()
    
    xgb_m = joblib.load("models/ens_xgb.pkl")
    model_feature_names = xgb_m.get_booster().feature_names
    
    X_test = test_df[model_feature_names].fillna(0)
    
    lgb_m = joblib.load("models/ens_lgb.pkl")
    cat_m = joblib.load("models/ens_cat.pkl")
    
    p_lgb = lgb_m.predict_proba(X_test)[:, 1]
    p_cat = cat_m.predict_proba(X_test)[:, 1]
    p_xgb = xgb_m.predict_proba(X_test)[:, 1]
    
    test_df["ens_prob"] = 0.40 * p_lgb + 0.35 * p_cat + 0.25 * p_xgb
    test_df["ai_rank"] = test_df.groupby("race_id")["ens_prob"].rank(ascending=False, method="min")
    
    races = test_df["race_id"].unique()
    
    strat_results = {
        "AI Top3 (1点勝負)": {"bets": 0, "hits": 0},
        "AI Top4 BOX (4点買い)": {"bets": 0, "hits": 0},
        "AI Top5 BOX (10点買い)": {"bets": 0, "hits": 0},
        "AI 軸1頭-Top2〜5流し (6点買い)": {"bets": 0, "hits": 0},
    }
    
    for r_id in races:
        r_df = test_df[test_df["race_id"] == r_id].sort_values(by="ai_rank")
        if len(r_df) < 5:
            continue
            
        actual_top3 = set(r_df[r_df["rank_num"] <= 3]["umaban"].astype(int).tolist())
        if len(actual_top3) != 3:
            continue
            
        u_rank1 = int(r_df[r_df["ai_rank"] == 1]["umaban"].iloc[0])
        u_top3 = set(r_df[r_df["ai_rank"] <= 3]["umaban"].astype(int).tolist())
        u_top4 = set(r_df[r_df["ai_rank"] <= 4]["umaban"].astype(int).tolist())
        u_top5 = set(r_df[r_df["ai_rank"] <= 5]["umaban"].astype(int).tolist())
        
        # 1. AI Top3 (1点)
        strat_results["AI Top3 (1点勝負)"]["bets"] += 1
        if u_top3 == actual_top3:
            strat_results["AI Top3 (1点勝負)"]["hits"] += 1
            
        # 2. AI Top4 BOX (4点)
        strat_results["AI Top4 BOX (4点買い)"]["bets"] += 1
        if actual_top3.issubset(u_top4):
            strat_results["AI Top4 BOX (4点買い)"]["hits"] += 1

        # 3. AI Top5 BOX (10点)
        strat_results["AI Top5 BOX (10点買い)"]["bets"] += 1
        if actual_top3.issubset(u_top5):
            strat_results["AI Top5 BOX (10点買い)"]["hits"] += 1

        # 4. 軸1頭 流し (6点)
        strat_results["AI 軸1頭-Top2〜5流し (6点買い)"]["bets"] += 1
        if u_rank1 in actual_top3 and actual_top3.issubset(u_top5):
            strat_results["AI 軸1頭-Top2〜5流し (6点買い)"]["hits"] += 1

    for strat, data in strat_results.items():
        count = data["bets"]
        hits = data["hits"]
        hit_rate = (hits / count) * 100 if count > 0 else 0
        print(f" [{strat:<24}] 対象レース: {count:3d} | 的中数: {hits:3d} | 的中率: ★ {hit_rate:5.2f} % ★")

def calculate_sanrenfuku_probs(df_race):
    scores = df_race["ai_score"].values
    horses = df_race["umaban"].astype(int).values
    names = df_race["horse_name"].values
    
    v = np.exp(scores - np.max(scores))
    n = len(v)
    
    sanren_results = []
    
    for combo in itertools.combinations(range(n), 3):
        i, j, k = combo
        p_sum = 0
        for p1, p2, p3 in itertools.permutations([i, j, k]):
            p_1st = v[p1] / np.sum(v)
            v_rem1 = np.delete(v, p1)
            p_2nd = v[p2] / np.sum(v_rem1)
            v_rem2 = np.delete(v, [min(p1, p2), max(p1, p2)])
            p_3rd = v[p3] / np.sum(v_rem2)
            p_sum += p_1st * p_2nd * p_3rd
            
        u_sorted = sorted([horses[i], horses[j], horses[k]])
        comb_str = f"{u_sorted[0]}-{u_sorted[1]}-{u_sorted[2]}"
        comb_name = f"{names[i]} - {names[j]} - {names[k]}"
        
        sanren_results.append({
            "combination": comb_str,
            "combination_name": comb_name,
            "u1": u_sorted[0],
            "u2": u_sorted[1],
            "u3": u_sorted[2],
            "prob": p_sum,
            "prob_pct": p_sum * 100
        })
        
    df_sanren = pd.DataFrame(sanren_results).sort_values(by="prob", ascending=False).reset_index(drop=True)
    return df_sanren

def main():
    parser = argparse.ArgumentParser(description="ばんえい競馬AI 3連複高的中率・黒字化専用推論モジュール")
    parser.add_argument("--card", type=str, default="", help="推論対象の出走表CSVパス")
    args = parser.parse_args()

    run_backtest_sanrenfuku()

    card_path = args.card
    if not card_path:
        cards = sorted(glob.glob("data/raw/banei_race_card_*.csv"))
        if cards:
            card_path = cards[-1]
        else:
            return

    print(f"\n=== 本日レース 3連複 推論出力 (対象: {card_path}) ===")
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
    for r_no in sorted(races):
        r_df = df[df["race_no"] == r_no].sort_values(by="ai_rank").copy()
        r_name = r_df["race_name"].iloc[0] if "race_name" in r_df.columns else ""
        print(f"\n-------------------------------------------------------")
        print(f"  第 {r_no:2d} レース : {r_name}")
        print(f"-------------------------------------------------------")

        df_sanren = calculate_sanrenfuku_probs(r_df)

        top1_uma = int(r_df[r_df["ai_rank"] == 1]["umaban"].iloc[0])
        top4_umas = r_df[r_df["ai_rank"] <= 4]["umaban"].astype(int).tolist()
        top5_umas = r_df[r_df["ai_rank"] <= 5]["umaban"].astype(int).tolist()

        top1_sanren = df_sanren.iloc[0]

        box4_df = df_sanren[df_sanren["u1"].isin(top4_umas) & df_sanren["u2"].isin(top4_umas) & df_sanren["u3"].isin(top4_umas)]
        box4_prob = box4_df["prob_pct"].sum()

        nagashi6_df = df_sanren[(df_sanren["u1"] == top1_uma) | (df_sanren["u2"] == top1_uma) | (df_sanren["u3"] == top1_uma)]
        nagashi6_df = nagashi6_df[nagashi6_df["u1"].isin(top5_umas) & nagashi6_df["u2"].isin(top5_umas) & nagashi6_df["u3"].isin(top5_umas)]
        nagashi6_prob = nagashi6_df["prob_pct"].sum()

        print(f" 🎯 【3連複 AI本命 1点買い】: 馬番 {top1_sanren['combination']} (確率: {top1_sanren['prob_pct']:.1f}%)")
        print(f" 📦 【3連複 AI上位4頭 BOX (4点)】: 馬番 {top4_umas} | 理論的中率 ★ {box4_prob:.1f}% ★")
        print(f" 🚀 【3連複 軸1頭 流し (6点)】: 軸 {top1_uma} ➔ {top5_umas[1:]} | 理論的中率 ★ {nagashi6_prob:.1f}% ★")

        print(f"\n   --- 高確率3連複買い目 TOP 3 ---")
        for rank_idx, s_row in df_sanren.head(3).iterrows():
            print(f"    {rank_idx+1}位 : 馬番 {s_row['combination']:<7} (確率: {s_row['prob_pct']:4.1f}%) | {s_row['combination_name']}")

if __name__ == "__main__":
    main()
