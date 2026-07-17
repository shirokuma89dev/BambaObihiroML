import argparse
import glob
import os
import re
import pandas as pd
import numpy as np

# ============================================================================
# モデルが使用する特徴量の正規リスト。
# engineer_features() が生成する列のうち、学習・推論の入力に使うものを列挙する。
# 気象の one-hot 列 (weather_*) はデータ依存で数が変わるため、
# 実行時に `FEATURE_COLS + [c for c in df.columns if c.startswith("weather_")]` で足す。
# ============================================================================
FEATURE_COLS = [
    # --- 積載重量(そり) と馬体・パワー ---
    "sled_weight_num", "sled_weight_change", "sled_weight_zscore",
    "horse_body_weight", "horse_weight_change", "power_ratio",
    "horse_body_weight_zscore", "power_ratio_zscore", "sled_weight_diff_max",
    # --- レース条件・クラス ---
    "track_moisture_num", "days_since_last_race", "class_level", "class_diff",
    # --- スピード・勢い ---
    "horse_avg_speed", "horse_max_speed", "speed_zscore", "momentum_score",
    # --- 実績エンコーディング(ベイズ平滑化) ---
    "horse_cum_win_rate", "horse_cum_top3_rate", "pair_top3_rate",
    "trainer_win_rate", "trainer_top3_rate", "jt_pair_top3_rate",
    # --- 近走成績・安定度 ---
    "horse_past_3_avg_rank", "horse_past_5_avg_rank", "horse_rank_std",
    "horse_past_3_avg_margin", "horse_best_time_sec", "horse_best_time_zscore",
    # --- 馬場適性(道悪巧者) ---
    "horse_dry_avg_rank", "horse_wet_avg_rank", "track_specialist_factor",
    # --- 騎手 ---
    "jockey_win_rate", "jockey_top3_rate",
    # --- 気象庁 外部データ ---
    "precip_total_mm", "temp_avg_c", "temp_max_c", "temp_min_c",
    "humidity_avg_pct", "wind_avg_mps", "sunlight_hours", "snowfall_cm", "snow_depth_cm",
    # --- 物理相互作用・調子・疲労・鞍上 ---
    "power_moisture_interaction", "sled_weight_moisture_interaction",
    "jockey_upgrade_factor", "recent_form_score", "fatigue_index",
    "sled_weight_relief", "jockey_moisture_specialist",
    # --- Eloレーティング / 補正タイム指数 ---
    "horse_elo_pre", "jockey_elo_pre", "horse_elo_zscore", "elo_gap_to_top",
    "horse_speed_figure", "horse_speed_figure_zscore",
    # --- 市場(人気順) ---
    "popularity_num", "pop_is_fav", "pop_inv", "pop_zscore",
]


def model_feature_cols(df):
    """FEATURE_COLS に、データに存在する気象one-hot列(weather_*)を加えて返す。"""
    return FEATURE_COLS + [c for c in df.columns if c.startswith("weather_")]


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

def parse_time_to_seconds(val):
    if pd.isna(val):
        return np.nan
    val_str = str(val).strip()
    match = re.search(r"(\d+):(\d+\.\d+|\d+)", val_str)
    if match:
        return float(match.group(1)) * 60.0 + float(match.group(2))
    match_sec = re.search(r"^\d+\.\d+$|^\d+$", val_str)
    if match_sec:
        return float(match_sec.group(0))
    return np.nan

def parse_margin_to_seconds(val):
    if pd.isna(val):
        return 0.0
    val_str = str(val).strip()
    val_str = val_str.translate(str.maketrans("０１２３４５６７８９．", "0123456789."))
    match = re.search(r"\d+\.\d+|\d+", val_str)
    return float(match.group(0)) if match else 0.0

def extract_class_level(race_name):
    if pd.isna(race_name):
        return 0
    name = str(race_name)
    if "オープン" in name or "重賞" in name or "ばんえい記念" in name:
        return 5
    elif "Ａ" in name or "A" in name:
        return 4
    elif "Ｂ" in name or "B" in name:
        return 3
    elif "Ｃ" in name or "C" in name:
        return 2
    elif "２歳" in name or "3歳" in name or "３歳" in name:
        return 1
    return 0

def bayesian_target_encoding(group_series, prior_mean, prior_weight=10):
    """
    ベイズ平滑化によるターゲットエンコーディング。
    試行回数が少ない場合は事前確率（prior_mean）に引っ張り、過大/過小評価を防ぐ。
    """
    cum_sum = group_series.shift(1).expanding().sum().fillna(0)
    cum_count = group_series.shift(1).expanding().count().fillna(0)
    return (cum_sum + prior_mean * prior_weight) / (cum_count + prior_weight)

def compute_multiplayer_elo(df, key, k=32.0, base=1500.0):
    """
    多頭レース用Eloレーティング。各馬(騎手)を相手全員とのペア対戦とみなし、
    着順の勝敗で更新。返すのは「そのレース開始前」のレート(=リークなし)。
    """
    ratings = {}
    pre_vals = np.full(len(df), base, dtype=float)
    groups = df.groupby("race_id", sort=False).indices
    names_all = df[key].astype(str).values
    ranks_all = df["rank_num"].values
    for rid in df["race_id"].drop_duplicates():
        ix = groups[rid]
        names = names_all[ix]
        ranks = ranks_all[ix]
        cur = np.array([ratings.get(n, base) for n in names])
        pre_vals[ix] = cur  # 更新前のレートを記録
        n = len(names)
        if n < 2:
            continue
        valid = ~np.isnan(ranks)
        new = cur.copy()
        for a in range(n):
            if not valid[a]:
                continue
            exp = act = 0.0
            cnt = 0
            for b in range(n):
                if a == b or not valid[b]:
                    continue
                exp += 1.0 / (1.0 + 10 ** ((cur[b] - cur[a]) / 400.0))
                if ranks[a] < ranks[b]:
                    act += 1.0
                elif ranks[a] == ranks[b]:
                    act += 0.5
                cnt += 1
            if cnt > 0:
                new[a] = cur[a] + k * (act - exp) / cnt
        for i, nm in enumerate(names):
            if valid[i]:
                ratings[nm] = new[i]
    return pre_vals


def compute_speed_figure(df):
    """
    タイムを斤量・馬場水分・クラスで補正した残差(=条件を除いた実力)。
    残差の符号を反転(速い=高得点)し、馬ごとに過去平均(shiftで当該レース除外)。
    物理条件→タイムの回帰であり着順は使わないためターゲットリークではない。
    """
    fit = df[["time_sec", "sled_weight_num", "track_moisture_num", "class_level"]].dropna()
    figure = np.full(len(df), np.nan)
    if len(fit) > 100:
        A = np.column_stack([
            fit["sled_weight_num"].values,
            fit["track_moisture_num"].values,
            fit["class_level"].values,
            np.ones(len(fit)),
        ])
        beta, *_ = np.linalg.lstsq(A, fit["time_sec"].values, rcond=None)
        X = np.column_stack([
            df["sled_weight_num"].values,
            df["track_moisture_num"].values,
            df["class_level"].values,
            np.ones(len(df)),
        ])
        expected = X @ beta
        # 速い(実測<期待)ほど大きくなるよう符号反転
        df = df.assign(_resid=(expected - df["time_sec"].values))
    else:
        df = df.assign(_resid=np.nan)
    fig = df.groupby("horse_name")["_resid"].transform(
        lambda s: s.shift(1).expanding(min_periods=1).mean()
    )
    fig = fig.fillna(0.0)
    fig_z = df.assign(_fig=fig).groupby("race_id")["_fig"].transform(
        lambda x: (x - x.mean()) / (x.std() + 1e-6) if len(x) > 1 else 0
    )
    return fig.values, fig_z.values


def engineer_features(df):
    """連結済み生データ(過去結果 + 任意で当日出走表)から全特徴量を生成して返す。
    shift/expanding系はレース時系列順に計算されるため、末尾に置いた出走表行は
    「そのレース開始前」の値(Elo/累積勝率/タイム指数等)を正しく受け取る。"""
    print(f"--- 特徴量エンジニアリング: 全 {len(df)} 件 ---")

    # 1. 基本パース
    df["rank_num"] = df["rank"].apply(clean_numeric)
    df["is_win"] = (df["rank_num"] == 1).astype(int)
    df["is_top3"] = (df["rank_num"] <= 3).astype(int)
    
    df["sled_weight_num"] = df["sled_weight"].apply(clean_numeric)
    df["track_moisture_num"] = df["track_moisture"].apply(clean_numeric)
    df["odds_num"] = df["odds"].apply(clean_numeric)
    df["popularity_num"] = df["popularity"].apply(clean_numeric).fillna(20.0)  # 欠損=人気薄扱い
    # 市場(人気順)由来の特徴量。オッズは過去年で欠損するため人気順を市場信号に用いる。
    df["pop_is_fav"] = (df["popularity_num"] == 1).astype(int)
    df["pop_inv"] = 1.0 / df["popularity_num"]
    df["pop_zscore"] = df.groupby("race_id")["popularity_num"].transform(
        lambda x: (x - x.mean()) / (x.std() + 1e-6) if len(x) > 1 else 0
    )
    df["time_sec"] = df["time"].apply(parse_time_to_seconds)
    df["margin_sec"] = df["margin"].apply(parse_margin_to_seconds)
    df["class_level"] = df["race_name"].apply(extract_class_level)
    
    hw_tuples = df["horse_weight"].apply(extract_horse_weight)
    df["horse_body_weight"] = [t[0] for t in hw_tuples]
    df["horse_weight_change"] = [t[1] for t in hw_tuples]
    df["power_ratio"] = df["sled_weight_num"] / df["horse_body_weight"]
    
    # 馬場水分との物理的相互作用特徴量
    df["power_moisture_interaction"] = df["power_ratio"] * df["track_moisture_num"]
    df["sled_weight_moisture_interaction"] = df["sled_weight_num"] * df["track_moisture_num"]
    
    # スピード (m/s) ばんえいは200m
    df["speed_mps"] = 200.0 / (df["time_sec"] + 1e-6)
    
    weather_dummies = pd.get_dummies(df["weather"], prefix="weather", dtype=int)
    df = pd.concat([df, weather_dummies], axis=1)
    
    df["date"] = pd.to_datetime(df["date"])
    df = df.sort_values(by=["date", "race_no", "umaban"]).reset_index(drop=True)
    
    # 気象庁の過去詳細気象データをマージ
    weather_csv = "data/raw/weather/jma_obihiro_weather.csv"
    if os.path.exists(weather_csv):
        print("気象庁の過去詳細気象データをマージ中...")
        weather_df = pd.read_csv(weather_csv)
        weather_df["date"] = pd.to_datetime(weather_df["date"])
        df = df.merge(weather_df, on="date", how="left")
    else:
        print("Warning: 気象データが見つかりません。デフォルト値（NaN）を使用します。")
        for col in ["precip_total_mm", "temp_avg_c", "temp_max_c", "temp_min_c", "humidity_avg_pct", "wind_avg_mps", "sunlight_hours", "snowfall_cm", "snow_depth_cm"]:
            df[col] = np.nan
    
    # 2. レース内相対特徴量 (Z-score系)
    print("レース内相対ハンデ・偏差値を計算中...")
    df["horse_body_weight_zscore"] = df.groupby("race_id")["horse_body_weight"].transform(
        lambda x: (x - x.mean()) / (x.std() + 1e-6) if len(x) > 1 else 0
    )
    df["power_ratio_zscore"] = df.groupby("race_id")["power_ratio"].transform(
        lambda x: (x - x.mean()) / (x.std() + 1e-6) if len(x) > 1 else 0
    )
    df["sled_weight_zscore"] = df.groupby("race_id")["sled_weight_num"].transform(
        lambda x: (x - x.mean()) / (x.std() + 1e-6) if len(x) > 1 else 0
    )
    df["sled_weight_diff_max"] = df.groupby("race_id")["sled_weight_num"].transform(
        lambda x: x - x.max()
    )
    
    # 3. 超高度ターゲットエンコーディング (ベイズ平滑化)
    print("ベイズ平滑化を用いた勝負気配・実力エンコーディング中...")
    GLOBAL_WIN_RATE = 0.125
    GLOBAL_TOP3_RATE = 0.375
    
    # 馬
    df["horse_cum_win_rate"] = df.groupby("horse_name")["is_win"].transform(lambda s: bayesian_target_encoding(s, prior_mean=GLOBAL_WIN_RATE, prior_weight=5))
    df["horse_cum_top3_rate"] = df.groupby("horse_name")["is_top3"].transform(lambda s: bayesian_target_encoding(s, prior_mean=GLOBAL_TOP3_RATE, prior_weight=5))
    
    # 騎手
    df["jockey_win_rate"] = df.groupby("jockey_name")["is_win"].transform(lambda s: bayesian_target_encoding(s, prior_mean=GLOBAL_WIN_RATE, prior_weight=15))
    df["jockey_top3_rate"] = df.groupby("jockey_name")["is_top3"].transform(lambda s: bayesian_target_encoding(s, prior_mean=GLOBAL_TOP3_RATE, prior_weight=15))
    
    # 乗り替わり勝負度合い (現在の騎手勝率 - 前走の騎手勝率)
    df["prev_jockey_win_rate"] = df.groupby("horse_name")["jockey_win_rate"].shift(1).fillna(GLOBAL_WIN_RATE)
    df["jockey_upgrade_factor"] = df["jockey_win_rate"] - df["prev_jockey_win_rate"]
    df["jockey_moisture_specialist"] = df["jockey_win_rate"] * df["track_moisture_num"]
    
    # 調教師
    df["trainer_win_rate"] = df.groupby("trainer_name")["is_win"].transform(lambda s: bayesian_target_encoding(s, prior_mean=GLOBAL_WIN_RATE, prior_weight=15))
    df["trainer_top3_rate"] = df.groupby("trainer_name")["is_top3"].transform(lambda s: bayesian_target_encoding(s, prior_mean=GLOBAL_TOP3_RATE, prior_weight=15))
    
    # 騎手×馬 / 騎手×調教師
    df["jockey_horse_pair"] = df["jockey_name"].astype(str) + "_" + df["horse_name"].astype(str)
    df["pair_top3_rate"] = df.groupby("jockey_horse_pair")["is_top3"].transform(lambda s: bayesian_target_encoding(s, prior_mean=GLOBAL_TOP3_RATE, prior_weight=3))
    
    df["jockey_trainer_pair"] = df["jockey_name"].astype(str) + "_" + df["trainer_name"].astype(str)
    df["jt_pair_top3_rate"] = df.groupby("jockey_trainer_pair")["is_top3"].transform(lambda s: bayesian_target_encoding(s, prior_mean=GLOBAL_TOP3_RATE, prior_weight=10))
    
    # 4. スピード＆モメンタム (勢い)
    print("スピード指標・モメンタム・道悪巧者インデックスを生成中...")
    df["prev_sled_weight"] = df.groupby("horse_name")["sled_weight_num"].shift(1)
    df["sled_weight_change"] = (df["sled_weight_num"] - df["prev_sled_weight"]).fillna(0)
    df["horse_past_3_avg_sled"] = df.groupby("horse_name")["sled_weight_num"].transform(lambda s: s.shift(1).rolling(3, min_periods=1).mean())
    df["sled_weight_relief"] = (df["horse_past_3_avg_sled"] - df["sled_weight_num"]).fillna(0)
    
    df["prev_date"] = df.groupby("horse_name")["date"].shift(1)
    df["days_since_last_race"] = (df["date"] - df["prev_date"]).dt.days.fillna(14)
    
    # 過去スピード実績
    df["horse_avg_speed"] = df.groupby("horse_name")["speed_mps"].transform(lambda s: s.shift(1).expanding(min_periods=1).mean()).fillna(1.5)
    df["horse_max_speed"] = df.groupby("horse_name")["speed_mps"].transform(lambda s: s.shift(1).cummax()).fillna(1.5)
    df["speed_zscore"] = df.groupby("race_id")["horse_avg_speed"].transform(lambda x: (x - x.mean()) / (x.std() + 1e-6) if len(x) > 1 else 0)
    
    # モメンタム（前走と前々走の着順差: マイナスなら着順良化＝勢いあり）
    df["prev_rank"] = df.groupby("horse_name")["rank_num"].shift(1)
    df["prev_prev_rank"] = df.groupby("horse_name")["rank_num"].shift(2)
    df["prev_prev_prev_rank"] = df.groupby("horse_name")["rank_num"].shift(3)
    df["momentum_score"] = (df["prev_rank"] - df["prev_prev_rank"]).fillna(0)
    
    # 近走調子スコア (加重順位平均: 直近に近いほど比重重め)
    df["recent_form_score"] = (
        df["prev_rank"].fillna(5.0) * 0.5 +
        df["prev_prev_rank"].fillna(5.0) * 0.3 +
        df["prev_prev_prev_rank"].fillna(5.0) * 0.2
    ).fillna(5.0)
    
    # 疲労インデックス (体重変化量 / レース間隔日数)
    # 体重が減っているのにレース間隔が短いほど、疲労度が大きく（マイナスに）なる
    df["fatigue_index"] = (df["horse_weight_change"] / (df["days_since_last_race"] + 1.0)).fillna(0)
    
    df["horse_past_3_avg_rank"] = df.groupby("horse_name")["rank_num"].transform(lambda s: s.shift(1).rolling(3, min_periods=1).mean())
    df["horse_past_5_avg_rank"] = df.groupby("horse_name")["rank_num"].transform(lambda s: s.shift(1).rolling(5, min_periods=1).mean())
    df["horse_rank_std"] = df.groupby("horse_name")["rank_num"].transform(lambda s: s.shift(1).rolling(5, min_periods=2).std())
    df["horse_past_3_avg_margin"] = df.groupby("horse_name")["margin_sec"].transform(lambda s: s.shift(1).rolling(3, min_periods=1).mean())
    
    df["prev_class_level"] = df.groupby("horse_name")["class_level"].shift(1)
    df["class_diff"] = (df["class_level"] - df["prev_class_level"]).fillna(0)
    
    df["horse_best_time_sec"] = df.groupby("horse_name")["time_sec"].transform(lambda s: s.shift(1).cummin())
    df["horse_best_time_zscore"] = df.groupby("race_id")["horse_best_time_sec"].transform(lambda x: (x - x.mean()) / (x.std() + 1e-6) if len(x) > 1 else 0)
    
    # 道悪巧者インデックス
    df["is_dry_track"] = (df["track_moisture_num"] < 2.0).astype(int)
    dry_mask = df["is_dry_track"] == 1
    df["dry_rank_val"] = np.where(dry_mask, df["rank_num"], np.nan)
    df["horse_dry_avg_rank"] = df.groupby("horse_name")["dry_rank_val"].transform(lambda s: s.shift(1).ffill().rolling(3, min_periods=1).mean())
    
    wet_mask = df["is_dry_track"] == 0
    df["wet_rank_val"] = np.where(wet_mask, df["rank_num"], np.nan)
    df["horse_wet_avg_rank"] = df.groupby("horse_name")["wet_rank_val"].transform(lambda s: s.shift(1).ffill().rolling(3, min_periods=1).mean())
    
    df["track_specialist_factor"] = (df["horse_wet_avg_rank"] - df["horse_dry_avg_rank"]).fillna(0)

    # 5. Eloレーティング (対戦相手の強さを織り込む動的能力指標)
    print("馬・騎手のEloレーティングを時系列更新中...")
    df["horse_elo_pre"] = compute_multiplayer_elo(df, "horse_name", k=32)
    df["jockey_elo_pre"] = compute_multiplayer_elo(df, "jockey_name", k=16)
    df["horse_elo_zscore"] = df.groupby("race_id")["horse_elo_pre"].transform(
        lambda x: (x - x.mean()) / (x.std() + 1e-6) if len(x) > 1 else 0
    )
    df["elo_gap_to_top"] = df.groupby("race_id")["horse_elo_pre"].transform(lambda x: x - x.max())

    # 6. 斤量・馬場補正タイム指数 (物理条件を除いた「真の能力」の時計)
    print("斤量・馬場補正タイム指数を生成中...")
    df["horse_speed_figure"], df["horse_speed_figure_zscore"] = compute_speed_figure(df)

    # 最終的な特徴量群 (全 45 次元 + 気象庁データ + 馬場水分相互作用 + 調子・疲労・鞍上 + Elo + タイム指数)
    base_cols = [
        "race_id", "date", "race_no", "race_name", "umaban", "waku",
        "horse_name", "sex_age", "jockey_name", "trainer_name", "weather",
        "class_level", "class_diff",
        "sled_weight_num", "sled_weight_change", "sled_weight_zscore", 
        "horse_body_weight", "horse_weight_change", "power_ratio",
        "horse_body_weight_zscore", "power_ratio_zscore", "sled_weight_diff_max",
        "track_moisture_num", "popularity_num", "odds_num", "days_since_last_race",
        "horse_avg_speed", "horse_max_speed", "speed_zscore", "momentum_score",
        "horse_cum_win_rate", "horse_cum_top3_rate", "pair_top3_rate",
        "trainer_win_rate", "trainer_top3_rate", "jt_pair_top3_rate",
        "horse_past_3_avg_rank", "horse_past_5_avg_rank", "horse_rank_std",
        "horse_past_3_avg_margin", "horse_best_time_sec", "horse_best_time_zscore",
        "horse_dry_avg_rank", "horse_wet_avg_rank", "track_specialist_factor",
        "jockey_win_rate", "jockey_top3_rate",
        "precip_total_mm", "temp_avg_c", "temp_max_c", "temp_min_c",
        "humidity_avg_pct", "wind_avg_mps", "sunlight_hours", "snowfall_cm", "snow_depth_cm",
        "power_moisture_interaction", "sled_weight_moisture_interaction",
        "jockey_upgrade_factor", "recent_form_score", "fatigue_index",
        "sled_weight_relief", "jockey_moisture_specialist",
        "horse_elo_pre", "jockey_elo_pre", "horse_elo_zscore", "elo_gap_to_top",
        "horse_speed_figure", "horse_speed_figure_zscore",
        "pop_is_fav", "pop_inv", "pop_zscore",
    ]
    weather_cols = list(weather_dummies.columns)
    target_cols = ["is_win", "is_top3", "rank_num", "speed_mps"]
    
    features_df = df[base_cols + weather_cols + target_cols].copy()
    return features_df


def process_raw_data(data_dir="data/raw", out_dir="data/processed"):
    os.makedirs(out_dir, exist_ok=True)

    result_files = glob.glob(os.path.join(data_dir, "banei_race_results_*.csv"))
    if not result_files:
        print(f"エラー: {data_dir} にレース結果CSVが存在しません。")
        return None

    dfs = [pd.read_csv(f) for f in result_files]
    df = pd.concat(dfs, ignore_index=True)
    print(f"--- 生データ処理開始: 全 {len(df)} 件 ---")

    features_df = engineer_features(df)

    out_path = os.path.join(out_dir, "features_train.csv")
    features_df.to_csv(out_path, index=False, encoding="utf-8-sig")
    print(f"完了: 特徴量データセット出力 -> {out_path} (全 {len(features_df)} 件)")
    return out_path

def main():
    parser = argparse.ArgumentParser(description="特徴量再生成（爆発的改善版）")
    parser.add_argument("--data-dir", type=str, default="data/raw")
    parser.add_argument("--out-dir", type=str, default="data/processed")
    args = parser.parse_args()
    process_raw_data(data_dir=args.data_dir, out_dir=args.out_dir)

if __name__ == "__main__":
    main()
