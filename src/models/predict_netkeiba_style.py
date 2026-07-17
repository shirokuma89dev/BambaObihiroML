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

def main():
    parser = argparse.ArgumentParser(description="netkeibaプロ予想超え ばんえい競馬AIプロフェッショナル予想生成モジュール")
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

    print(f"=======================================================")
    print(f"    netkeiba超え！アルティメットAIプロフェッショナル予想新聞")
    print(f"    (45次元爆発的進化版: 鉄板レース厳選機能搭載)")
    print(f"    対象出走表: {card_path}")
    print(f"=======================================================")
    
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
                  "horse_best_time_sec", "horse_best_time_zscore", "horse_dry_avg_rank", "horse_wet_avg_rank"]
        
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
        
        if "trainer_name" in df.columns:
            latest_trainer = hist_df.groupby("trainer_name").agg(
                trainer_win_rate=("is_win", "mean"),
                trainer_top3_rate=("is_top3", "mean")
            ).reset_index()
            df = df.merge(latest_trainer, on="trainer_name", how="left")
        elif "trainer" in df.columns:
            latest_trainer = hist_df.groupby("trainer_name").agg(
                trainer_win_rate=("is_win", "mean"),
                trainer_top3_rate=("is_top3", "mean")
            ).reset_index()
            df = df.merge(latest_trainer, left_on="trainer", right_on="trainer_name", how="left")

        df["pair_top3_rate"] = df["horse_cum_top3_rate"]
        df["jt_pair_top3_rate"] = df["trainer_top3_rate"] if "trainer_top3_rate" in df.columns else 0.30

    df["horse_past_3_avg_rank"] = df["horse_past_3_avg_rank"].fillna(4.5)
    df["horse_past_5_avg_rank"] = df["horse_past_5_avg_rank"].fillna(4.5)
    df["horse_rank_std"] = df["horse_rank_std"].fillna(1.5)
    df["horse_past_3_avg_margin"] = df["horse_past_3_avg_margin"].fillna(0.0)
    df["horse_best_time_sec"] = df["horse_best_time_sec"].fillna(150.0)
    df["horse_best_time_zscore"] = df["horse_best_time_zscore"].fillna(0.0)
    df["horse_dry_avg_rank"] = df["horse_dry_avg_rank"].fillna(4.5)
    df["horse_wet_avg_rank"] = df["horse_wet_avg_rank"].fillna(4.5)
    df["jockey_win_rate"] = df["jockey_win_rate"].fillna(0.10)
    df["jockey_top3_rate"] = df["jockey_top3_rate"].fillna(0.30)
    df["trainer_win_rate"] = df["trainer_win_rate"].fillna(0.10) if "trainer_win_rate" in df.columns else 0.10
    df["trainer_top3_rate"] = df["trainer_top3_rate"].fillna(0.30) if "trainer_top3_rate" in df.columns else 0.30
    df["jt_pair_top3_rate"] = df["jt_pair_top3_rate"].fillna(0.30)
    df["horse_cum_win_rate"] = df["horse_cum_win_rate"].fillna(0.10)
    df["horse_cum_top3_rate"] = df["horse_cum_top3_rate"].fillna(0.30)
    df["pair_top3_rate"] = df["pair_top3_rate"].fillna(0.30)
    df["sled_weight_zscore"] = df["power_ratio_zscore"]
    df["horse_avg_speed"] = 1.5
    df["horse_max_speed"] = 1.6
    df["speed_zscore"] = 0.0
    df["momentum_score"] = 0.0
    df["track_specialist_factor"] = 0.0

    m1 = joblib.load("models/pos_m1.pkl")
    
    # 内部の学習済みLightGBMから正しい特徴量名を取得
    model_feature_names = m1.calibrated_classifiers_[0].estimator.feature_name_

    for col in model_feature_names:
        if col not in df.columns:
            if col.startswith("weather_"):
                w_name = col.replace("weather_", "")
                df[col] = (df["weather"] == w_name).astype(int)
            else:
                df[col] = 0.0

    X_predict = df[model_feature_names].fillna(0)

    # 1着AI勝率予測
    df["ens_prob"] = m1.predict_proba(X_predict)[:, 1]
    df["ai_rank"] = df.groupby("race_id")["ens_prob"].rank(ascending=False, method="min")

    races = df["race_no"].unique()
    for r_no in sorted(races):
        r_df = df[df["race_no"] == r_no].sort_values(by="ai_rank").copy()
        r_name = r_df["race_name"].iloc[0] if "race_name" in r_df.columns else ""
        moisture = r_df["track_moisture_num"].iloc[0] if "track_moisture_num" in r_df.columns else np.nan
        
        top_prob = r_df.iloc[0]["ens_prob"] * 100
        
        if top_prob >= 50.0:
            confidence_level = "★★★★★【超鉄板・神ガチレース】(的中率 56.0%超)"
        elif top_prob >= 40.0:
            confidence_level = "★★★★☆【勝負レース】(的中率 44.8%超)"
        elif top_prob >= 30.0:
            confidence_level = "★★★☆☆【堅めのレース】(的中率 37.4%)"
        else:
            confidence_level = "★☆☆☆☆【大荒れ・見送り推奨】(AI確信度 低)"

        print(f"\n=======================================================")
        print(f"  第 {r_no:2d} レース : {r_name} (馬場水分: {moisture:.1f}%)")
        print(f"  AIガチ度判定 : {confidence_level}  (本命信頼度 {top_prob:.1f}%)")
        print(f"=======================================================")

        marks = {}
        for idx, row in r_df.iterrows():
            rk = int(row["ai_rank"])
            if rk == 1: m = "◎"
            elif rk == 2: m = "○"
            elif rk == 3: m = "▲"
            elif rk == 4: m = "△"
            elif rk == 5: m = "☆"
            else: m = "  "
            marks[int(row["umaban"])] = m
            r_df.loc[idx, "mark"] = m

        print(" 【出走表 ＆ AI印・短評欄】")
        for idx, row in r_df.iterrows():
            uma = int(row["umaban"])
            m = row["mark"]
            prob_pct = row["ens_prob"] * 100
            jockey = str(row["jockey_name"])
            weight = str(row["sled_weight"])
            
            reason = []
            if row["power_ratio_zscore"] < -0.5: reason.append("軽量馬体優位")
            if row["class_diff"] > 0: reason.append("昇級戦注意")
            if row["class_diff"] < 0: reason.append("降格有利")
            if row["jockey_top3_rate"] >= 0.35: reason.append(f"騎手{jockey}高適性")
            if row["horse_best_time_sec"] < 130: reason.append("スピード能力上位")
            comment = " / ".join(reason) if reason else "順当"

            print(f"   {m} | 馬番 {uma:2d} | {row['horse_name']:<12} | 騎手:{jockey:<6} | 斤量:{weight:<4} | AI勝率:{prob_pct:4.1f}% | [{comment}]")

        h_top1 = int(r_df[r_df["ai_rank"] == 1]["umaban"].iloc[0])
        h_top2 = int(r_df[r_df["ai_rank"] == 2]["umaban"].iloc[0])
        h_top3 = int(r_df[r_df["ai_rank"] == 3]["umaban"].iloc[0])
        h_top4 = int(r_df[r_df["ai_rank"] == 4]["umaban"].iloc[0])
        h_top5 = int(r_df[r_df["ai_rank"] == 5]["umaban"].iloc[0])

        if top_prob < 30.0:
            print(f"\n ⚠️ 【AI警告】このレースは荒れる可能性が高いため、馬券の購入は見送りを推奨します。")
        else:
            print(f"\n 💡 【AIプロ予想家 渾身の買い目カード】")
            print(f"  ・ 単　勝 : 馬番 {h_top1}  (本命 ◎)")
            print(f"  ・ 複　勝 : 馬番 {h_top1}, {h_top2}  (◎ ○ 2頭指定)")
            print(f"  ・ 馬連単 : {h_top1} ➔ {h_top2}, {h_top3}, {h_top4}  (3点買い)")
            print(f"  ・ 3連複  : {h_top1} - {h_top2} - {h_top3}, {h_top4}, {h_top5}  (3点買い)")
            print(f"  ・ 3連単 (勝負フォーメーション): {h_top1} ➔ {h_top2}, {h_top3} ➔ {h_top2}, {h_top3}, {h_top4}, {h_top5}  (6点買い)")

if __name__ == "__main__":
    main()
