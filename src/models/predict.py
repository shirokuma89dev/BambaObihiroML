"""
帯広競馬場 現地対話型リアルタイム推論モジュール

競馬場現地で入手した直前情報（馬場水分量、パドック馬体重増減、直前オッズ・人気）を
その場で手入力・再修正し、AI着順予測をリアルタイムに即座計算するツール。

使い方:
  python src/models/predict.py
"""
import os
import re
import sys
import glob
import argparse
import numpy as np
import pandas as pd
import joblib

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "features"))
from build_features import engineer_features  # noqa: E402

MODEL_PATH = "models/banei_ranker.pkl"
FEATURES_PATH = "models/banei_ranker_features.pkl"
MARKS = ["◎", "○", "▲", "△", "☆"]


def parse_card_odds(val):
    """出走表の '5.4(3)' -> (オッズ5.4, 人気3)。"""
    if pd.isna(val):
        return np.nan, np.nan
    m = re.search(r"([\d.]+)\s*\((\d+)\)", str(val))
    if m:
        return float(m.group(1)), float(m.group(2))
    m2 = re.search(r"[\d.]+", str(val))
    return (float(m2.group(0)), np.nan) if m2 else (np.nan, np.nan)


def card_to_results_schema(card):
    """出走表の列を、過去結果CSVと同じスキーマに変換。"""
    odds_pop = card["popularity"].apply(parse_card_odds)
    return pd.DataFrame({
        "race_id": card["race_id"], "date": card["date"], "race_no": card["race_no"],
        "race_name": card["race_name"], "weather": card["weather"],
        "track_moisture": card["track_moisture"], "rank": np.nan,
        "waku": card["waku"], "umaban": card["umaban"], "horse_name": card["horse_name"],
        "sex_age": np.nan, "sled_weight": card.get("sled_weight", np.nan),
        "jockey_name": card["jockey_name"], "trainer_name": np.nan,
        "horse_weight": card.get("horse_weight", np.nan), "time": np.nan, "margin": np.nan,
        "popularity": [p for _, p in odds_pop], "odds": [o for o, _ in odds_pop],
    })


def select_card_file():
    """対話式に出走表CSVファイルを選択する。"""
    cards = sorted(glob.glob("data/raw/banei_race_card_*.csv"))
    if not cards:
        print("エラー: data/raw/ に出走表データ(banei_race_card_*.csv)が存在しません。")
        sys.exit(1)

    print("\n------------------------------------------------------------")
    print(" 📋 出走表ファイルの選択")
    print("------------------------------------------------------------")
    for idx, path in enumerate(cards, 1):
        filename = os.path.basename(path)
        default_mark = " (最新 - デフォルト)" if idx == len(cards) else ""
        print(f"  [{idx}] {filename}{default_mark}")
    
    choice = input(f"\n番号を入力してください [1-{len(cards)}] (Enterで最新): ").strip()
    if not choice:
        return cards[-1]
    
    try:
        n = int(choice)
        if 1 <= n <= len(cards):
            return cards[n - 1]
    except ValueError:
        pass
    
    print("最新の出走表を選択します。")
    return cards[-1]


def build_detail_reason(row):
    """馬個別の定量特徴に基づく根拠タグの生成"""
    reasons = []
    elo = row.get("horse_elo_pre", np.nan)
    if pd.notna(elo) and elo > 1530:
        reasons.append(f"Elo高({int(elo)})")
    
    recent = row.get("recent_form_score", np.nan)
    if pd.notna(recent) and recent <= 3.0:
        reasons.append("近走絶好調")
    
    wr = row.get("horse_cum_win_rate", 0)
    if wr >= 0.25:
        reasons.append(f"高勝率({wr*100:.0f}%)")
    
    upgrade = row.get("jockey_upgrade_factor", 0)
    if upgrade >= 0.05:
        reasons.append(f"鞍上強化(+{upgrade*100:.1f}%)")
        
    pop = row.get("popularity_num", np.nan)
    if pop == 1:
        reasons.append("1番人気")
        
    return " / ".join(reasons) if reasons else "標準評価"


def compute_and_predict_race(card_df, hist_df, feats, model, target_race_no):
    """指定レースの現在カードデータから特徴量を全ビルドして即座に再推論する。"""
    card_ids = set(card_df["race_id"])
    card_min_date = pd.to_datetime(card_df["date"]).dt.strftime("%Y-%m-%d").min()
    
    # 完全時系列リーク防止: 対象出走表の日付より「過去」のレース結果のみを参照
    hist_dates = pd.to_datetime(hist_df["date"]).dt.strftime("%Y-%m-%d")
    hist_filtered = hist_df[(~hist_df["race_id"].isin(card_ids)) & (hist_dates < card_min_date)].copy()
    
    combined = pd.concat([hist_filtered, card_to_results_schema(card_df)], ignore_index=True)
    fdf = engineer_features(combined)
    fdf["race_id"] = fdf["race_id"].astype(str)
    
    for c in feats:
        if c not in fdf.columns:
            fdf[c] = 0.0
    fdf[feats] = fdf[feats].fillna(0)

    target_race_df = fdf[(fdf["race_id"].isin(card_ids)) & (fdf["race_no"] == target_race_no)].copy()
    if target_race_df.empty:
        return None
        
    target_race_df["score"] = model.predict(target_race_df[feats])
    return target_race_df


def render_race_view(race_df):
    """単一レースのAI全頭分析画面の描画"""
    r = race_df.sort_values("score", ascending=False).reset_index(drop=True)
    e = np.exp(r["score"] - r["score"].max())
    pwin = (e / e.sum()).values
    
    race_name = str(r.iloc[0]["race_name"])
    moisture = r.iloc[0].get("track_moisture_num", np.nan)
    weather = r.iloc[0].get("weather", "")
    race_no = int(r.iloc[0]["race_no"])
    
    print(f"\n=========================================================================")
    print(f" 🏇 現地リアルタイム推論画面 : 第 {race_no} レース 『{race_name}』")
    print(f" 馬場水分: {moisture:.1f}% | 天候: {weather} | 出走: {len(r)}頭")
    print(f"=========================================================================")
    print(f"{'印':<2} {'馬番':>4} {'馬名':<14} {'騎手':<8} {'斤量':>4} {'馬体重':>9} {'人気':>4} {'予測勝率':>8} {'分析'}")
    print("-" * 78)
    
    for i, row in r.iterrows():
        mk = MARKS[i] if i < len(MARKS) else "  "
        uma = int(row["umaban"])
        name = str(row["horse_name"])
        jockey = str(row["jockey_name"]).replace("（ばんえい）", "")
        weight = int(row["sled_weight_num"]) if pd.notna(row.get("sled_weight_num")) else "-"
        
        h_weight = int(row["horse_body_weight"]) if pd.notna(row.get("horse_body_weight")) else "-"
        h_change = int(row.get("horse_weight_change", 0)) if pd.notna(row.get("horse_weight_change")) else 0
        h_weight_str = f"{h_weight}({h_change:+d})" if h_weight != "-" else "-"
        
        pop = int(row["popularity_num"]) if pd.notna(row.get("popularity_num")) else "-"
        p_pct = pwin[i] * 100
        reason = build_detail_reason(row)
        
        print(f"{mk:<2} {uma:>4} {name:<14} {jockey:<8} {weight:>4} {h_weight_str:>9} {pop:>4} {p_pct:>7.1f}% [{reason}]")
        
    combo = "-".join(str(int(x)) for x in r.head(3)["umaban"].values)
    trio_conf = pwin[0] * pwin[1] * pwin[2] if len(pwin) >= 3 else pwin[0]
    print("-" * 78)
    print(f" 🎯 AI推奨 3連単1点フォーメーション: {combo} (確信度: {trio_conf*100:.2f}%)")
    print("=========================================================================\n")


def edit_race_interactive(card_df, target_race_no):
    """現地直前情報の対話的書き換え処理"""
    r_idx = card_df[card_df["race_no"] == target_race_no].index
    if len(r_idx) == 0:
        return card_df
        
    print("\n【現地直前情報の手入力メニュー】")
    print("  [m] 馬場水分量 (%) を変更")
    print("  [w] 馬体重・増減 (例: 馬番 2 ➔ 1050 -10) を変更")
    print("  [o] オッズ・人気順 (例: 馬番 2 ➔ 3.2 1) を変更")
    print("  [b] レース画面に戻る")
    
    sub_cmd = input("\n現地入力項目を選択してください [m/w/o/b]: ").strip().lower()
    
    if sub_cmd == "m":
        val = input("最新の馬場水分量 (%) を入力してください (例: 1.8): ").strip()
        try:
            m_val = float(val)
            card_df.loc[r_idx, "track_moisture"] = m_val
            print(f"➔ 第 {target_race_no} レースの馬場水分量を {m_val}% に更新しました。")
        except ValueError:
            print("無効な数値入力です。")
            
    elif sub_cmd == "w":
        uma_in = input("対象の馬番を入力してください: ").strip()
        try:
            uma = int(uma_in)
            h_row = card_df[(card_df["race_no"] == target_race_no) & (card_df["umaban"] == uma)]
            if h_row.empty:
                print(f"馬番 {uma} は存在しません。")
                return card_df
                
            w_val = input("馬体重 (kg) を入力してください (例: 1040): ").strip()
            c_val = input("前走比増減 (kg) を入力してください (例: -10 や +5): ").strip()
            
            w_num = int(w_val)
            c_num = int(c_val)
            hw_str = f"{w_num}({c_num:+d})"
            
            c_idx = card_df[(card_df["race_no"] == target_race_no) & (card_df["umaban"] == uma)].index
            card_df.loc[c_idx, "horse_weight"] = hw_str
            print(f"➔ 馬番 {uma} の馬体重を {hw_str} に更新しました。")
        except ValueError:
            print("無効な入力形式です。")
            
    elif sub_cmd == "o":
        uma_in = input("対象の馬番を入力してください: ").strip()
        try:
            uma = int(uma_in)
            c_idx = card_df[(card_df["race_no"] == target_race_no) & (card_df["umaban"] == uma)].index
            if len(c_idx) == 0:
                print(f"馬番 {uma} は存在しません。")
                return card_df
                
            odds_in = input("直前オッズを入力してください (例: 3.2): ").strip()
            pop_in = input("人気順を入力してください (例: 1): ").strip()
            
            o_val = float(odds_in)
            p_val = int(pop_in)
            pop_str = f"{o_val:.1f}({p_val})"
            
            card_df.loc[c_idx, "popularity"] = pop_str
            card_df.loc[c_idx, "odds"] = o_val
            print(f"➔ 馬番 {uma} のオッズ/人気を {pop_str} に更新しました。")
        except ValueError:
            print("無効な入力形式です。")
            
    return card_df


def main():
    ap = argparse.ArgumentParser(description="帯広競馬場 現地対話型リアルタイム推論")
    ap.add_argument("--card", default="", help="出走表CSVパス")
    args = ap.parse_args()

    card_path = args.card if args.card else select_card_file()
    print(f"\nデータ読み込み中: {card_path} ...")

    if not os.path.exists(MODEL_PATH) or not os.path.exists(FEATURES_PATH):
        print(f"エラー: 学習済みモデルが見つかりません。先に python src/models/train.py を実行してください。")
        sys.exit(1)

    model = joblib.load(MODEL_PATH)
    feats = joblib.load(FEATURES_PATH)

    card_df = pd.read_csv(card_path)
    card_df["race_id"] = card_df["race_id"].astype(str)

    hist_files = glob.glob("data/raw/banei_race_results_*.csv")
    if not hist_files:
        print("エラー: data/raw/ に過去データ(banei_race_results_*.csv)が見つかりません。")
        sys.exit(1)

    hist_df = pd.concat([pd.read_csv(f) for f in hist_files], ignore_index=True)
    hist_df["race_id"] = hist_df["race_id"].astype(str)

    all_race_nos = sorted(card_df["race_no"].unique())
    current_race_no = all_race_nos[0]

    while True:
        # 現在のレースを再計算して表示
        target_df = compute_and_predict_race(card_df, hist_df, feats, model, current_race_no)
        if target_df is not None:
            render_race_view(target_df)
        else:
            print(f"第 {current_race_no} レースのデータ計算に失敗しました。")

        print("------------------------------------------------------------")
        print(f" [現地操作コマンド]")
        print(f"  [1-12] : レース切り替え (現在: {current_race_no}R)")
        print(f"  [e]    : パドック・馬場水分・直前オッズの『手入力・修正』")
        print(f"  [n]    : 次のレースへ進む ({current_race_no + 1 if current_race_no < max(all_race_nos) else current_race_no}R)")
        print(f"  [q]    : 終了")
        print("------------------------------------------------------------")
        
        cmd = input("コマンドを入力してください > ").strip().lower()

        if cmd == "q":
            print("\n現地予想セッションを終了します。")
            break
        elif cmd == "e":
            card_df = edit_race_interactive(card_df, current_race_no)
        elif cmd == "n":
            if current_race_no < max(all_race_nos):
                current_race_no += 1
            else:
                print("最終レースです。")
        elif cmd.isdigit():
            r_no = int(cmd)
            if r_no in all_race_nos:
                current_race_no = r_no
            else:
                print(f"エラー: 第 {r_no} レースのデータが存在しません。")
        else:
            print("無効な入力です。")


if __name__ == "__main__":
    main()
