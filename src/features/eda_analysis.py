import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import lightgbm as lgb

def set_paper_style():
    plt.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.size": 13,
            "axes.labelsize": 15,
            "xtick.labelsize": 13,
            "ytick.labelsize": 13,
            "axes.linewidth": 1.0,
            "axes.edgecolor": "#222222",
            "xtick.direction": "in",
            "ytick.direction": "in",
            "xtick.major.width": 0.9,
            "ytick.major.width": 0.9,
            "xtick.minor.visible": True,
            "ytick.minor.visible": True,
            "xtick.major.pad": 6.0,
            "ytick.major.pad": 6.0,
            "legend.frameon": True,
            "legend.framealpha": 1.0,
            "legend.fancybox": False,
            "legend.edgecolor": "#444444",
            "savefig.bbox": "tight",
            "savefig.pad_inches": 0.02,
        }
    )

def main():
    print("=== データセット EDA（探索的データ分析）＆特徴量貢献度解析 ===")
    
    data_path = "data/processed/features_train.csv"
    if not os.path.exists(data_path):
        print(f"エラー: {data_path} が存在しません。")
        return
        
    df = pd.read_csv(data_path)
    print(f"解析対象総レコード数: {len(df)} 件")
    
    feature_cols = [
        "sled_weight_num", "sled_weight_change", "horse_body_weight", "horse_weight_change", "power_ratio",
        "horse_body_weight_zscore", "power_ratio_zscore", "sled_weight_diff_max",
        "track_moisture_num", "days_since_last_race", "class_level", "class_diff",
        "horse_cum_win_rate", "horse_cum_top3_rate", "pair_top3_rate",
        "trainer_win_rate", "trainer_top3_rate", "jt_pair_top3_rate",
        "horse_past_3_avg_rank", "horse_past_5_avg_rank", "horse_rank_std",
        "horse_past_3_avg_margin", "horse_best_time_sec", "horse_best_time_zscore",
        "horse_dry_avg_rank", "horse_wet_avg_rank",
        "jockey_win_rate", "jockey_top3_rate"
    ]
    weather_cols = [c for c in df.columns if c.startswith("weather_")]
    feature_cols.extend(weather_cols)
    
    X = df[feature_cols].fillna(0)
    y = df["is_win"]
    
    # 1. LightGBM による特徴量重要度計算
    lgb_clf = lgb.LGBMClassifier(n_estimators=300, learning_rate=0.03, random_state=42, verbose=-1)
    lgb_clf.fit(X, y)
    
    importances = lgb_clf.feature_importances_
    feat_imp = pd.DataFrame({"feature": feature_cols, "importance": importances})
    feat_imp = feat_imp.sort_values("importance", ascending=True).tail(15)
    
    # 2. 特徴量重要度バーチャート生成
    set_paper_style()
    fig, ax = plt.subplots(figsize=(9, 6))
    
    y_pos = np.arange(len(feat_imp))
    ax.barh(
        y_pos, feat_imp["importance"],
        color="#2c3e50", edgecolor="#111111", linewidth=0.9, height=0.65
    )
    ax.set_yticks(y_pos)
    ax.set_yticklabels(feat_imp["feature"])
    ax.set_xlabel("Feature Importance (Split Count)")
    ax.set_title("Banei Racing ML Top 15 Feature Importances", fontsize=15, pad=10)
    
    for spine in ax.spines.values():
        spine.set_linewidth(1.0)
        
    os.makedirs("reports/figures", exist_ok=True)
    save_fig_path = "reports/figures/eda_feature_importance.png"
    plt.tight_layout()
    plt.savefig(save_fig_path, dpi=300)
    plt.close()
    print(f"グラフ保存完了: {save_fig_path}")
    
    # 3. 馬場水分量 × パワー負荷率と勝率の関係（クロス集計解析）
    df["moisture_bin"] = pd.cut(df["track_moisture_num"], bins=[0, 1.5, 3.0, 5.0, 10.0], labels=["Dry (<1.5%)", "Normal (1.5-3%)", "Wet (3-5%)", "Heavy (>5%)"])
    df["power_bin"] = pd.qcut(df["power_ratio"], q=4, labels=["Low Load (Q1)", "Medium (Q2)", "High (Q3)", "Heavy Load (Q4)"])
    
    pivot_win = df.pivot_table(index="power_bin", columns="moisture_bin", values="is_win", aggfunc="mean") * 100
    
    print("\n--- 【EDA分析結果】馬場水分量 × パワー負荷率 勝率クロス集計 (%) ---")
    print(pivot_win.round(2).to_string())
    
    print("\n=== 特徴量TOP10 インサイトランキング ===")
    top10_imp = feat_imp.tail(10).iloc[::-1]
    for idx, row in top10_imp.iterrows():
        print(f" - {row['feature']:<28} : スプリット数 {int(row['importance'])}")

if __name__ == "__main__":
    main()
