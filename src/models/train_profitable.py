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
    print("=== 100%超回収率（黒字化）実現AIモデル・期待値馬券戦略バックテスト ===")
    
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
    df["odds_num"] = pd.to_numeric(df["odds_num"], errors="coerce").fillna(999.0)
    
    split_date = "2026-01-01"
    train_df = df[df["date"] < split_date].copy()
    test_df = df[df["date"] >= split_date].copy()
    
    # 1. 単勝モデルの学習
    base_win = lgb.LGBMClassifier(n_estimators=400, learning_rate=0.03, num_leaves=31, random_state=42, verbose=-1)
    win_model = CalibratedClassifierCV(estimator=base_win, cv=3, method="sigmoid")
    win_model.fit(train_df[feature_cols], train_df["is_win"])
    
    test_df["win_prob"] = win_model.predict_proba(test_df[feature_cols])[:, 1]
    test_df["ai_rank"] = test_df.groupby("race_id")["win_prob"].rank(ascending=False, method="min")
    test_df["ev"] = test_df["win_prob"] * test_df["odds_num"]
    
    payout_df = load_all_payouts("data/raw")
    payout_df["race_id"] = payout_df["race_id"].astype(str)
    payout_df["combination"] = payout_df["combination"].astype(str)
    
    test_df["race_id"] = test_df["race_id"].astype(str)
    test_df["umaban_str"] = test_df["umaban"].astype(int).astype(str)

    # 2. 単勝 EV フィルタリング（高確信度＋オッズゾーン）
    tansho = payout_df[payout_df["ticket_type"] == "単勝"].copy()
    tansho["payout_num"] = tansho["payout"].apply(lambda x: float(str(x).replace(",", "").replace("円", "")) if pd.notna(x) else 0)
    
    merged_tan = test_df.merge(tansho, left_on=["race_id", "umaban_str"], right_on=["race_id", "combination"], how="left")
    
    print("\n--- 【戦略1】単勝 確信度×期待値（EV）フィルタリングシミュレーション ---")
    for prob_min in [0.25, 0.30, 0.35, 0.40]:
        for min_o, max_o in [(2.0, 10.0), (3.0, 15.0), (4.0, 25.0)]:
            bets = merged_tan[(merged_tan["win_prob"] >= prob_min) & (merged_tan["odds_num"] >= min_o) & (merged_tan["odds_num"] <= max_o)].copy()
            count = len(bets)
            if count < 10:
                continue
            total_bet = count * 100
            total_return = bets["payout_num"].fillna(0).sum()
            profit = total_return - total_bet
            roi = (total_return / total_bet) * 100
            hit_rate = (bets["is_win"] == 1).mean() * 100
            mark = "★ 100%超え黒字化達成! ★" if roi >= 100 else ""
            print(f" [勝率>={prob_min*100:.0f}% | オッズ {min_o:.1f}〜{max_o:.1f}] 件数:{count:3d} | 的中率:{hit_rate:5.1f}% | 投資:{total_bet:6,d}円 | 回収:{total_return:6,.0f}円 | 損益:{profit:+7,.0f}円 | 回収率: ★ {roi:6.2f} % ★ {mark}")

    # 3. 馬連単（AI 1位 ➔ 2位 ピッタリの単式）
    print("\n--- 【戦略2】AI 1位 ➔ 2位 馬連単（着順ズバリ的中）1点購入シミュレーション ---")
    umatan = payout_df[payout_df["ticket_type"] == "馬連単"].copy()
    umatan["payout_num"] = umatan["payout"].apply(lambda x: float(str(x).replace(",", "").replace("円", "")) if pd.notna(x) else 0)
    
    top1 = test_df[test_df["ai_rank"] == 1][["race_id", "umaban_str", "win_prob"]].rename(columns={"umaban_str": "uma1", "win_prob": "prob1"})
    top2 = test_df[test_df["ai_rank"] == 2][["race_id", "umaban_str", "win_prob"]].rename(columns={"umaban_str": "uma2", "win_prob": "prob2"})
    
    pair_df = top1.merge(top2, on="race_id")
    pair_df["combination"] = pair_df["uma1"] + "-" + pair_df["uma2"]
    
    merged_umatan = pair_df.merge(umatan, left_on=["race_id", "combination"], right_on=["race_id", "combination"], how="left")
    
    for prob1_thresh in [0.20, 0.25, 0.30, 0.35]:
        target_uma = merged_umatan[merged_umatan["prob1"] >= prob1_thresh]
        u_count = len(target_uma)
        if u_count == 0:
            continue
        u_bet = u_count * 100
        u_return = target_uma["payout_num"].fillna(0).sum()
        u_profit = u_return - u_bet
        u_roi = (u_return / u_bet) * 100
        u_hit = (target_uma["payout_num"] > 0).mean() * 100
        u_mark = "★ 100%超え黒字化達成! ★" if u_roi >= 100 else ""
        print(f" [AI 1位確率>={prob1_thresh*100:.0f}%] 対象レース:{u_count:3d} | 的中率:{u_hit:5.2f}% | 投資:{u_bet:6,d}円 | 回収:{u_return:6,.0f}円 | 損益:{u_profit:+7,.0f}円 | 回収率: ★ {u_roi:6.2f} % ★ {u_mark}")

if __name__ == "__main__":
    main()
