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
    # odds と popularity のカラムが独立して存在し、値が入っている場合
    if "odds" in card.columns and "popularity" in card.columns and not card["odds"].isna().all():
        odds = card["odds"]
        popularity = card["popularity"]
    else:
        # 従来の popularity カラムに "19.9(5)" などと合体している場合のフォールバック
        odds_pop = card["popularity"].apply(parse_card_odds)
        odds = [o for o, _ in odds_pop]
        popularity = [p for _, p in odds_pop]

    return pd.DataFrame({
        "race_id": card["race_id"], "date": card["date"], "race_no": card["race_no"],
        "race_name": card["race_name"], "weather": card["weather"],
        "track_moisture": card["track_moisture"], "rank": np.nan,
        "waku": card["waku"], "umaban": card["umaban"], "horse_name": card["horse_name"],
        "sex_age": np.nan, "sled_weight": card.get("sled_weight", np.nan),
        "jockey_name": card["jockey_name"], "trainer_name": np.nan,
        "horse_weight": card.get("horse_weight", np.nan), "time": np.nan, "margin": np.nan,
        "popularity": popularity, "odds": odds,
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
    target_race_df = fdf[(fdf["race_id"].isin(card_ids)) & (fdf["race_no"] == target_race_no)].copy()
    if target_race_df.empty:
        return None
        
    # 推論時のみ一時的に NaN を 0 埋めして予測を実行（表示用のオリジナルデータは欠損を保持）
    X_inference = target_race_df[feats].fillna(0)
    target_race_df["score"] = model.predict(X_inference)
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


def compute_pl_probabilities(scores):
    """Plackett-Luceモデルに基づく1着、2着、3着確率の計算"""
    N = len(scores)
    exp_s = np.exp(scores - np.max(scores))
    
    # 1着確率
    p1 = exp_s / np.sum(exp_s)
    
    # 2着確率
    p2 = np.zeros(N)
    for j in range(N):
        pj_1st = p1[j]
        denom2 = np.sum(exp_s) - exp_s[j]
        if denom2 <= 0:
            continue
        for i in range(N):
            if i == j:
                continue
            p2[i] += pj_1st * (exp_s[i] / denom2)
            
    # 3着確率
    p3 = np.zeros(N)
    for j in range(N):
        pj_1st = p1[j]
        denom2 = np.sum(exp_s) - exp_s[j]
        if denom2 <= 0:
            continue
        for k in range(N):
            if k == j:
                continue
            pk_2nd = exp_s[k] / denom2
            denom3 = denom2 - exp_s[k]
            if denom3 <= 0:
                continue
            for i in range(N):
                if i == j or i == k:
                    continue
                p3[i] += pj_1st * pk_2nd * (exp_s[i] / denom3)
                
    return p1, p2, p3


def compute_pair_probabilities(p1, exp_s):
    """全ペアに対する馬連およびワイド確率の計算（Plackett-Luce準拠）"""
    N = len(p1)
    p_quinella = np.zeros((N, N))
    p_wide = np.zeros((N, N))
    sum_exp = np.sum(exp_s)
    
    for i in range(N):
        for j in range(i + 1, N):
            # 馬連 (i, j が 1着-2着 または 2着-1着)
            p_ij = p1[i] * (exp_s[j] / (sum_exp - exp_s[i]))
            p_ji = p1[j] * (exp_s[i] / (sum_exp - exp_s[j]))
            prob_q = p_ij + p_ji
            p_quinella[i, j] = p_quinella[j, i] = prob_q
            
            # ワイド (i, j がともに3着以内に入る確率)
            prob_w = 0.0
            for k in range(N):
                if k == i or k == j:
                    continue
                denom2_i = sum_exp - exp_s[i]
                denom2_j = sum_exp - exp_s[j]
                denom2_k = sum_exp - exp_s[k]
                
                # (i, j, k)
                p_ijk = p1[i] * (exp_s[j] / denom2_i) * (exp_s[k] / (denom2_i - exp_s[j]))
                # (j, i, k)
                p_jik = p1[j] * (exp_s[i] / denom2_j) * (exp_s[k] / (denom2_j - exp_s[i]))
                # (i, k, j)
                p_ikj = p1[i] * (exp_s[k] / denom2_i) * (exp_s[j] / (denom2_i - exp_s[k]))
                # (j, k, i)
                p_jki = p1[j] * (exp_s[k] / denom2_j) * (exp_s[i] / (denom2_j - exp_s[k]))
                # (k, i, j)
                p_kij = p1[k] * (exp_s[i] / denom2_k) * (exp_s[j] / (denom2_k - exp_s[i]))
                # (k, j, i)
                p_kji = p1[k] * (exp_s[j] / denom2_k) * (exp_s[i] / (denom2_k - exp_s[j]))
                
                prob_w += (p_ijk + p_jik + p_ikj + p_jki + p_kij + p_kji)
                
            p_wide[i, j] = p_wide[j, i] = prob_w
            
    return p_quinella, p_wide


def render_kelly_calculator(race_df):
    """単勝オッズとAI予測確率を用いたケリー基準およびダッシングの資金配分シミュレータ"""
    # 予測確率の再計算
    r = race_df.copy()
    e = np.exp(r["score"] - r["score"].max())
    p1, p2, p3 = compute_pl_probabilities(r["score"].values)
    r["prob"] = p1
    r["prob_place"] = p1 + p2 + p3

    # オッズが入っている馬だけを対象にする
    valid_r = r[r["odds_num"].notna() & (r["odds_num"] > 0)].copy()
    if len(valid_r) == 0:
        print("\n[!] オッズ情報が未入力です。[e] コマンドでオッズを入力してから実行してください。")
        return

    print("\n=========================================================================")
    print(" 📊 期待値＆資金配分計算シミュレータ (複数式別対応)")
    print("=========================================================================")
    
    # 予算の入力
    budget_in = input(" 今回のレースの総予算（円）を入力してください [デフォルト: 10000]: ").strip()
    budget = 10000.0
    if budget_in:
        try:
            budget = float(budget_in)
        except ValueError:
            print("無効な数値のため、デフォルトの 10,000 円で計算します。")

    probs = valid_r["prob"].values
    odds = valid_r["odds_num"].values
    names = valid_r["horse_name"].values
    umabans = valid_r["umaban"].values

    # 1. 各馬の単体期待値 (EV) の計算
    evs = probs * odds
    valid_r["ev"] = evs

    # 2. 複数排他ケリー基準 (Kelly Criterion) の計算
    n = len(valid_r)
    items = []
    for idx in range(n):
        items.append((idx, umabans[idx], names[idx], probs[idx], odds[idx], evs[idx]))
    items.sort(key=lambda x: x[5], reverse=True) # 期待値の降順

    S = []
    C = 1.0
    for idx, umaban, name, p, o, ev in items:
        temp_S = S + [(idx, umaban, name, p, o)]
        sum_p = sum(x[3] for x in temp_S)
        sum_inv_o = sum(1.0 / x[4] for x in temp_S)
        
        if sum_inv_o >= 1.0:
            break
            
        temp_C = (1.0 - sum_p) / (1.0 - sum_inv_o)
        if ev > temp_C:
            S = temp_S
            C = temp_C
        else:
            break

    kelly_fracs = np.zeros(n)
    for idx, umaban, name, p, o in S:
        kelly_fracs[idx] = p - C / o

    # 3. ダッシング (Dutching - 均等払戻) の計算
    dutch_items = [x for x in items if x[5] > 1.0]
    dutch_fracs = np.zeros(n)
    if dutch_items:
        sum_inv_o_dutch = sum(1.0 / x[4] for x in dutch_items)
        if sum_inv_o_dutch < 1.0:
            for idx, umaban, name, p, o, ev in dutch_items:
                dutch_fracs[idx] = 1.0 / (o * sum_inv_o_dutch)

    # 表示用にデータを結合
    valid_r["kelly_pct"] = kelly_fracs * 100
    valid_r["kelly_yen"] = np.round(kelly_fracs * budget / 100) * 100
    valid_r["q_kelly_pct"] = kelly_fracs * 0.25 * 100
    valid_r["q_kelly_yen"] = np.round(kelly_fracs * 0.25 * budget / 100) * 100
    valid_r["dutch_pct"] = dutch_fracs * 100
    valid_r["dutch_yen"] = np.round(dutch_fracs * budget / 100) * 100

    # 4. 複勝 (Place), 馬連 (Quinella), ワイド (Wide) の期待値計算
    inv_odds = 1.0 / odds
    pm1 = inv_odds / np.sum(inv_odds)
    
    # Plackett-Luceによる2着・3着確率
    exp_s_ai = np.exp(valid_r["score"].values - np.max(valid_r["score"].values))
    exp_s_m = pm1 # 市場のスコア重みは pm1 そのもの
    
    p1_ai, p2_ai, p3_ai = p1, p2, p3
    p1_m, p2_m, p3_m = compute_pl_probabilities(np.log(pm1 + 1e-12))
    
    p_place_ai = p1_ai + p2_ai + p3_ai
    p_place_m = p1_m + p2_m + p3_m
    
    # 複勝の期待値と想定オッズ (テラ銭 25% = 0.75)
    place_odds = 0.75 / (p_place_m + 1e-12)
    place_evs = p_place_ai * place_odds
    
    valid_r["place_odds_est"] = place_odds
    valid_r["place_ev"] = place_evs

    # 馬連 & ワイド
    pq_ai, pw_ai = compute_pair_probabilities(p1_ai, exp_s_ai)
    pq_m, pw_m = compute_pair_probabilities(p1_m, exp_s_m)
    
    # 馬連期待値と想定オッズ
    quinella_odds = np.zeros((n, n))
    quinella_evs = np.zeros((n, n))
    # ワイド期待値と想定オッズ
    wide_odds = np.zeros((n, n))
    wide_evs = np.zeros((n, n))
    
    for i in range(n):
        for j in range(i+1, n):
            quinella_odds[i, j] = quinella_odds[j, i] = 0.75 / (pq_m[i, j] + 1e-12)
            quinella_evs[i, j] = quinella_evs[j, i] = pq_ai[i, j] * quinella_odds[i, j]
            
            wide_odds[i, j] = wide_odds[j, i] = 0.75 / (pw_m[i, j] + 1e-12)
            wide_evs[i, j] = wide_evs[j, i] = pw_ai[i, j] * wide_odds[i, j]

    # --- 画面描画 ---
    print(f"\n 総予算: {int(budget):,} 円")
    print("-" * 92)
    print(f"{'馬番':>2} {'馬名':<14} {'単勝確率':>8} {'単勝オッズ':>6} {'単勝EV':>6} | {'25%ケリー(推奨)':^16} | {'均等払戻(ダッシング)':^18}")
    print(f"{'':>2} {'':<14} {'':>8} {'':>6} {'':>7} | {'割合':>6} {'購入額':>8} | {'割合':>6} {'購入額':>8} {'想定払戻':>8}")
    print("-" * 92)

    for i, row in valid_r.sort_values("prob", ascending=False).iterrows():
        uma = int(row["umaban"])
        name = str(row["horse_name"])
        prob_str = f"{row['prob']*100:.1f}%"
        odds_str = f"{row['odds_num']:.1f}"
        ev_str = f"{row['ev']:.2f}"
        
        ev_mark = "★" if row["ev"] > 1.0 else "  "
        qk_pct_str = f"{row['q_kelly_pct']:.1f}%" if row['q_kelly_pct'] > 0 else "-"
        qk_yen_str = f"{int(row['q_kelly_yen']):,}円" if row['q_kelly_yen'] > 0 else "-"
        dutch_pct_str = f"{row['dutch_pct']:.1f}%" if row['dutch_pct'] > 0 else "-"
        dutch_yen_str = f"{int(row['dutch_yen']):,}円" if row['dutch_yen'] > 0 else "-"
        dutch_pay = f"{int(np.round(row['dutch_yen'] * row['odds_num'])):,}円" if row['dutch_yen'] > 0 else "-"

        print(f"{uma:>2} {name:<14} {prob_str:>8} {odds_str:>6} {ev_str:>6}{ev_mark} | {qk_pct_str:>5} {qk_yen_str:>7} | {dutch_pct_str:>5} {dutch_yen_str:>7} {dutch_pay:>8}")

    print("-" * 92)
    total_qk = int(valid_r["q_kelly_yen"].sum())
    total_dutch = int(valid_r["dutch_yen"].sum())
    print(f" [合計購入金額]   25%ケリー: {total_qk:,}円 (予算の {total_qk/budget*100:.1f}%) | 均等払戻: {total_dutch:,}円")
    
    pos_ev_count = (valid_r["ev"] > 1.0).sum()

    # 🛒 【マークカード購入指示シート】の表示
    print("-" * 92)
    print(" 🛒 【マークカード記入用・期待値分散ポートフォリオ購入指示】")
    print("-" * 92)
    
    # ポール予算配分
    # 単勝 30%, 複勝 30%, 馬連 20%, ワイド 20%
    win_pool = int(budget * 0.30)
    place_pool = int(budget * 0.30)
    quinella_pool = int(budget * 0.20)
    wide_pool = int(budget * 0.20)
    
    actual_spent = 0
    saved_cash = 0

    def allocate_pool_budget(pool_budget, items):
        if not items:
            return {}
        total_w = sum(x[1] for x in items)
        if total_w <= 0:
            return {}
        
        alloc = {}
        raw_bets = []
        for item_id, w in items:
            raw_val = (w / total_w) * pool_budget
            raw_bets.append((item_id, raw_val))
            
        rounded_bets = []
        for item_id, val in raw_bets:
            r_val = int(np.round(val / 100) * 100)
            rounded_bets.append((item_id, r_val))
            
        diff = int(pool_budget) - sum(x[1] for x in rounded_bets)
        if diff != 0 and len(rounded_bets) > 0:
            max_idx = np.argmax([x[1] for x in items])
            temp = list(rounded_bets[max_idx])
            temp[1] = max(0, temp[1] + diff)
            rounded_bets[max_idx] = tuple(temp)
            
        return {x[0]: x[1] for x in rounded_bets if x[1] > 0}

    # 1. 三連単 (固定1枚保険)
    top_3_sorted = r.sort_values("score", ascending=False).head(3)
    if len(top_3_sorted) >= 3:
        comb_str = " -> ".join(str(int(x)) for x in top_3_sorted["umaban"].values)
        p1, p2, p3 = top_3_sorted["prob"].values
        conf = p1 * p2 * p3
        print(f" 🎯 【三連単・本命1点保険】 (AI本命推奨・確信度: {conf*100:.2f}%)")
        print(f"   👉 【三連単】 [ {comb_str} ] を 100円（1枚）買います。")
        actual_spent += 100
        print()

    # 2. 複勝 (Place) - 保険・ローリスク (予算30%)
    place_candidates = []
    for _, row in valid_r.iterrows():
        ev = row["place_ev"]
        if ev > 1.0:
            place_candidates.append((int(row["umaban"]), ev - 1.0, str(row["horse_name"]), ev, row["place_odds_est"]))
            
    if place_candidates:
        place_alloc = allocate_pool_budget(place_pool, [(x[0], x[1]) for x in place_candidates])
        if place_alloc:
            print(f" 🎯 【複勝・保険ローリスク】 - 予算 {place_pool:,}円 (EV比例配分)")
            for uma, _, name, ev, est_o in place_candidates:
                yen = place_alloc.get(uma, 0)
                if yen > 0:
                    sheets = yen // 100
                    print(f"   👉 【複勝】 [ {uma}番 ] ({name}) を {yen:,}円（{sheets}枚）買います。 (期待値: {ev:.2f}, 推定オッズ: {est_o:.1f}倍)")
                    actual_spent += yen
            print()
        else:
            print(f" 🎯 【複勝・保険ローリスク】")
            print(f"   👉 期待値がプラスの馬が存在しないため、予算 {place_pool:,}円 は購入を見送ります（貯蓄）。")
            saved_cash += place_pool
            print()
    else:
        print(f" 🎯 【複勝・保険ローリスク】")
        print(f"   👉 期待値がプラスの馬が存在しないため、予算 {place_pool:,}円 は購入を見送ります（貯蓄）。")
        saved_cash += place_pool
        print()

    # 3. 単勝 (Win) - 中リスク・高期待値 (予算30%)
    kelly_candidates = []
    for _, row in valid_r.iterrows():
        if row["q_kelly_yen"] > 0:
            kelly_candidates.append((int(row["umaban"]), row["q_kelly_pct"], str(row["horse_name"]), row["ev"], row["odds_num"]))
            
    if kelly_candidates:
        win_alloc = allocate_pool_budget(win_pool, [(x[0], x[1]) for x in kelly_candidates])
        if win_alloc:
            print(f" 🎯 【単勝・勝負中リスク】 - 予算 {win_pool:,}円 (25%ケリー最適配分)")
            for uma, _, name, ev, o in kelly_candidates:
                yen = win_alloc.get(uma, 0)
                if yen > 0:
                    sheets = yen // 100
                    print(f"   👉 【単勝】 [ {uma}番 ] ({name}) を {yen:,}円（{sheets}枚）買います。 (期待値: {ev:.2f}, オッズ: {o:.1f}倍)")
                    actual_spent += yen
            print()
        else:
            print(f" 🎯 【単勝・勝負中リスク】")
            print(f"   👉 期待値がプラスの馬が存在しないため、予算 {win_pool:,}円 は購入を見送ります（貯蓄）。")
            saved_cash += win_pool
            print()
    else:
        print(f" 🎯 【単勝・勝負中リスク】")
        print(f"   👉 期待値がプラスの馬が存在しないため、予算 {win_pool:,}円 は購入を見送ります（貯蓄）。")
        saved_cash += win_pool
        print()

    # 4. 馬連 (Quinella) - 中リスク・中リターン (予算20%)
    quinella_candidates = []
    for i in range(n):
        for j in range(i+1, n):
            ev = quinella_evs[i, j]
            if ev > 1.0:
                quinella_candidates.append((f"{int(umabans[i])}-{int(umabans[j])}", ev - 1.0, ev, quinella_odds[i, j]))
    quinella_candidates.sort(key=lambda x: x[2], reverse=True)
    
    # 上位3ペアに絞る
    quinella_candidates = quinella_candidates[:3]
    if quinella_candidates:
        q_alloc = allocate_pool_budget(quinella_pool, [(x[0], x[1]) for x in quinella_candidates])
        if q_alloc:
            print(f" 🎯 【馬連・中リスク中リターン】 - 予算 {quinella_pool:,}円 (EV比例配分)")
            for pair_key, _, ev, est_o in quinella_candidates:
                yen = q_alloc.get(pair_key, 0)
                if yen > 0:
                    sheets = yen // 100
                    print(f"   👉 【馬連】 [ {pair_key} ] を {yen:,}円（{sheets}枚）買います。 (期待値: {ev:.2f}, 推定オッズ: {est_o:.1f}倍)")
                    actual_spent += yen
            print()
        else:
            print(f" 🎯 【馬連・中リスク中リターン】")
            print(f"   👉 期待値がプラスのペアが存在しないため、予算 {quinella_pool:,}円 は購入を見送ります（貯蓄）。")
            saved_cash += quinella_pool
            print()
    else:
        print(f" 🎯 【馬連・中リスク中リターン】")
        print(f"   👉 期待値がプラスのペアが存在しないため、予算 {quinella_pool:,}円 は購入を見送ります（貯蓄）。")
        saved_cash += quinella_pool
        print()

    # 5. ワイド (Wide) - 低中リスク・中リターン (予算20%)
    wide_candidates = []
    for i in range(n):
        for j in range(i+1, n):
            ev = wide_evs[i, j]
            if ev > 1.0:
                wide_candidates.append((f"{int(umabans[i])}-{int(umabans[j])}", ev - 1.0, ev, wide_odds[i, j]))
    wide_candidates.sort(key=lambda x: x[2], reverse=True)
    
    # 上位3ペアに絞る
    wide_candidates = wide_candidates[:3]
    if wide_candidates:
        w_alloc = allocate_pool_budget(wide_pool, [(x[0], x[1]) for x in wide_candidates])
        if w_alloc:
            print(f" 🎯 【ワイド・抑えローリスク】 - 予算 {wide_pool:,}円 (EV比例配分)")
            for pair_key, _, ev, est_o in wide_candidates:
                yen = w_alloc.get(pair_key, 0)
                if yen > 0:
                    sheets = yen // 100
                    print(f"   👉 【ワイド】 [ {pair_key} ] を {yen:,}円（{sheets}枚）買います。 (期待値: {ev:.2f}, 推定オッズ: {est_o:.1f}倍)")
                    actual_spent += yen
            print()
        else:
            print(f" 🎯 【ワイド・抑えローリスク】")
            print(f"   👉 期待値がプラスのペアが存在しないため、予算 {wide_pool:,}円 は購入を見送ります（貯蓄）。")
            saved_cash += wide_pool
            print()
    else:
        print(f" 🎯 【ワイド・抑えローリスク】")
        print(f"   👉 期待値がプラスのペアが存在しないため、予算 {wide_pool:,}円 は購入を見送ります（貯蓄）。")
        saved_cash += wide_pool
        print()

    print("-" * 92)
    print(f" 【ポートフォリオ購入集計】")
    print(f"   👉 今回の総投資額  : {actual_spent:,}円 ({actual_spent // 100}枚)")
    print(f"   👉 購入見送り貯蓄  : {saved_cash:,}円")
    if actual_spent == 100:
        print("   ⚠️ 期待値がプラスの買い目が全式別で存在しません。三連単本命1点のみで遊ぶか、ケン（見送り）を強く推奨します。")
    print("=========================================================================\n")


def render_current_status(card_df, target_race_no):
    """現在の登録情報・入力状態を馬番順に一覧表示"""
    r = card_df[card_df["race_no"] == target_race_no].copy()
    if r.empty:
        print(f"第 {target_race_no} レースのデータが存在しません。")
        return
        
    r["umaban_int"] = r["umaban"].astype(int)
    r = r.sort_values("umaban_int").reset_index(drop=True)
    
    race_name = str(r.iloc[0]["race_name"])
    moisture = r.iloc[0].get("track_moisture")
    weather = r.iloc[0].get("weather", "")
    
    moisture_str = f"{moisture:.1f}%" if pd.notna(moisture) else "未設定"
    
    print(f"\n============================================================")
    print(f" 📋 現在の登録情報確認 (馬番順) : 第 {target_race_no} レース 『{race_name}』")
    print(f" 馬場水分: {moisture_str} | 天候: {weather}")
    print(f"============================================================")
    print(f"{'馬番':>4} {'馬名':<14} {'斤量':>4} {'馬体重(増減)':>12} {'オッズ':>7} {'人気':>4} {'ステータス'}")
    print("-" * 68)
    
    for _, row in r.iterrows():
        uma = int(row["umaban"])
        name = str(row["horse_name"])
        
        # Sled weight
        sled = row.get("sled_weight")
        sled_str = str(int(float(sled))) if pd.notna(sled) and str(sled).strip() else "-"
        
        # Horse weight
        hw = row.get("horse_weight")
        hw_str = str(hw) if pd.notna(hw) and str(hw).strip() else "-"
        
        # Odds & Popularity
        odds = row.get("odds")
        odds_str = f"{float(odds):.1f}" if pd.notna(odds) and str(odds).strip() else "-"
        
        pop = row.get("popularity")
        pop_str = str(int(float(pop))) if pd.notna(pop) and str(pop).strip() else "-"
        
        # Status check
        status_parts = []
        if hw_str == "-":
            status_parts.append("馬体重未入力")
        if odds_str == "-":
            status_parts.append("オッズ未入力")
            
        status = " / ".join(status_parts) if status_parts else "入力完了"
        if status == "入力完了":
            status = "✓ OK"
            
        print(f"{uma:>4} {name:<14} {sled_str:>4} {hw_str:>12} {odds_str:>7} {pop_str:>4}  {status}")
    print("============================================================\n")


def edit_race_interactive(card_df, target_race_no):
    """現地情報の手入力修正用サブメニュー"""
    r_idx = card_df[card_df["race_no"] == target_race_no].index
    if len(r_idx) == 0:
        return card_df
        
    while True:
        print("\n【現地直前情報の手入力メニュー】")
        print("  [m] 馬場水分量 (%) を変更")
        print("  [w] 馬体重・増減 (複数頭を連続で入力/変更可能)")
        print("  [o] オッズ・人気順 (複数頭を連続で入力/変更可能)")
        print("  [b] 編集を確定してレース予測を表示")
        
        sub_cmd = input("\n現地入力項目を選択してください [m/w/o/b]: ").strip().lower()
        if sub_cmd == "b":
            print("➔ 編集を確定しました。再推論を実行します。")
            break
            
        if sub_cmd == "m":
            current_moisture = card_df.loc[r_idx[0], "track_moisture"]
            moisture_desc = f"{current_moisture:.1f}%" if pd.notna(current_moisture) else "未設定"
            val = input(f"最新の馬場水分量 (%) を入力してください (現在の値: {moisture_desc}) [Enterでスキップ]: ").strip()
            if val:
                try:
                    m_val = float(val)
                    card_df.loc[r_idx, "track_moisture"] = m_val
                    print(f"➔ 第 {target_race_no} レースの馬場水分量を {m_val}% に更新しました。")
                except ValueError:
                    print("無効な数値入力です。")
                
        elif sub_cmd == "w":
            print("\n--- 馬体重の連続編集モード (終了するには馬番で Enter を押してください) ---")
            while True:
                uma_in = input("\n馬番を入力してください: ").strip()
                if not uma_in:
                    break
                try:
                    uma = int(uma_in)
                    c_idx = card_df[(card_df["race_no"] == target_race_no) & (card_df["umaban"] == uma)].index
                    if len(c_idx) == 0:
                        print(f"馬番 {uma} は存在しません。")
                        continue
                    
                    # 現在の値を取得してパース
                    current_w = np.nan
                    current_c = np.nan
                    hw_str = card_df.loc[c_idx[0], "horse_weight"]
                    if pd.notna(hw_str) and hw_str:
                        m_hw = re.search(r"(\d{3,4})\s*\(([+-]?\d+)\)", str(hw_str))
                        if m_hw:
                            current_w = int(m_hw.group(1))
                            current_c = int(m_hw.group(2))
                            
                    w_desc = f"{current_w} kg" if pd.notna(current_w) else "未設定"
                    c_desc = f"{current_c:+.0f} kg" if pd.notna(current_c) else "未設定"
                    
                    w_val = input(f"  馬番 {uma} の馬体重 (現在の値: {w_desc}) [Enterでスキップ]: ").strip()
                    c_val = input(f"  馬番 {uma} の前走比増減 (現在の値: {c_desc}) [Enterでスキップ]: ").strip()
                    
                    new_w = int(w_val) if w_val else current_w
                    new_c = int(c_val) if c_val else current_c
                    
                    if pd.notna(new_w) and pd.notna(new_c):
                        formatted_hw = f"{int(new_w)}({int(new_c):+d})"
                        card_df.loc[c_idx, "horse_weight"] = formatted_hw
                        print(f"  ➔ 馬体重を {formatted_hw} に更新しました。")
                except ValueError:
                    print("無効な入力値です。もう一度入力してください。")
                    
        elif sub_cmd == "o":
            print("\n--- オッズ・人気の連続編集モード (終了するには馬番で Enter を押してください) ---")
            while True:
                uma_in = input("\n馬番を入力してください: ").strip()
                if not uma_in:
                    break
                try:
                    uma = int(uma_in)
                    c_idx = card_df[(card_df["race_no"] == target_race_no) & (card_df["umaban"] == uma)].index
                    if len(c_idx) == 0:
                        print(f"馬番 {uma} は存在しません。")
                        continue
                        
                    current_odds = card_df.loc[c_idx[0], "odds"]
                    current_pop = card_df.loc[c_idx[0], "popularity"]
                    
                    odds_desc = f"{current_odds:.1f}" if pd.notna(current_odds) else "未設定"
                    pop_desc = f"{current_pop}" if pd.notna(current_pop) else "未設定"
                    
                    odds_in = input(f"  馬番 {uma} の直前オッズ (現在の値: {odds_desc}) [Enterでスキップ]: ").strip()
                    pop_in = input(f"  馬番 {uma} の人気順 (現在の値: {pop_desc}) [Enterでスキップ]: ").strip()
                    
                    new_odds = float(odds_in) if odds_in else current_odds
                    new_pop = int(pop_in) if pop_in else current_pop
                    
                    if pd.notna(new_odds):
                        card_df.loc[c_idx, "odds"] = float(new_odds)
                    if pd.notna(new_pop):
                        card_df.loc[c_idx, "popularity"] = str(new_pop)
                        
                    if pd.notna(new_odds) or pd.notna(new_pop):
                        print(f"  ➔ オッズ: {new_odds} / 人気: {new_pop} に更新しました。")
                except ValueError:
                    print("無効な入力値です。もう一度入力してください。")
                    
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
    card_df["popularity"] = card_df["popularity"].astype(object)
    card_df["horse_weight"] = card_df["horse_weight"].astype(object)

    hist_files = glob.glob("data/raw/banei_race_results_*.csv")
    if not hist_files:
        print("エラー: data/raw/ に過去データ(banei_race_results_*.csv)が見つかりません。")
        sys.exit(1)

    hist_df = pd.concat([pd.read_csv(f) for f in hist_files], ignore_index=True)
    hist_df["race_id"] = hist_df["race_id"].astype(str)

    all_race_nos = sorted(card_df["race_no"].unique())
    print(f"\n登録されているレース: {min(all_race_nos)}R 〜 {max(all_race_nos)}R")
    while True:
        r_in = input(f"最初に予測・表示するレース番号を入力してください ({min(all_race_nos)} - {max(all_race_nos)}, 既定値: {min(all_race_nos)}) > ").strip()
        if not r_in:
            current_race_no = min(all_race_nos)
            break
        try:
            r_val = int(r_in)
            if r_val in all_race_nos:
                current_race_no = r_val
                break
            else:
                print(f"エラー: 第 {r_val} レースは存在しません。")
        except ValueError:
            print("有効なレース番号を入力してください。")

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
        print(f"  [v]    : 現在の登録情報・入力状態を一覧表示 (馬番順)")
        print(f"  [e]    : パドック・馬場水分・直前オッズの『手入力・修正』")
        print(f"  [k]    : 期待値から最適な購入割合を計算（単勝ケリー/ダッシング）")
        print(f"  [n]    : 次のレースへ進む ({current_race_no + 1 if current_race_no < max(all_race_nos) else current_race_no}R)")
        print(f"  [q]    : 終了")
        print("------------------------------------------------------------")
        
        cmd = input("コマンドを入力してください > ").strip().lower()

        if cmd == "q":
            print("\n現地予想セッションを終了します。")
            break
        elif cmd == "e":
            card_df = edit_race_interactive(card_df, current_race_no)
        elif cmd == "v":
            render_current_status(card_df, current_race_no)
            input("Enterキーを押すと戻ります...")
        elif cmd == "k":
            if target_df is not None:
                render_kelly_calculator(target_df)
            else:
                print("予測データが存在しません。先に予測を実行してください。")
            input("Enterキーを押すと戻ります...")
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
