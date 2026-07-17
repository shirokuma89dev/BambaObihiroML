import pandas as pd
import numpy as np
import joblib

from train_position_models_v2 import FEATURE_COLS_BASE, MARKET_COLS, add_market_features

"""
真の勝利条件 = 回収率(ROI)。実配当(banei_race_payouts_2026.csv)で
「実際に儲かるか」を検証する。100%超で黒字。
戦略: モデル確率 × 実オッズ が +EV の馬券のみ選抜(バリューベット)。
"""

STAKE = 100.0  # 1点あたりの賭け金(円)


def paynum(s):
    return float(str(s).replace("円", "").replace(",", "").strip())


def load_payouts():
    p = pd.read_csv("data/raw/banei_race_payouts_2026.csv")
    p["pay"] = p["payout"].apply(paynum)
    lut = {}  # (race_id, ticket_type) -> (combination_str, payout)
    for _, row in p.iterrows():
        lut[(row["race_id"], row["ticket_type"])] = (str(row["combination"]), row["pay"])
    return lut


def get_scores():
    df = pd.read_csv("data/processed/features_train.csv")
    df["date"] = pd.to_datetime(df["date"])
    df = add_market_features(df)
    weather_cols = [c for c in df.columns if c.startswith("weather_")]
    feats = FEATURE_COLS_BASE + MARKET_COLS + weather_cols
    df[feats] = df[feats].fillna(0)
    test = df[df["date"] >= "2026-01-01"].copy()
    m1 = joblib.load("models/pos_v2_p_rank1.pkl")
    m2 = joblib.load("models/pos_v2_p_rank2.pkl")
    test["p1"] = m1.predict_proba(test[feats])[:, 1]
    test["p2"] = m2.predict_proba(test[feats])[:, 1]
    return test


def run_strategy(test, lut, name, bet_fn):
    """bet_fn(race_df) -> list of (ticket_type, combo_str, model_prob or None)。EVフィルタはbet_fn内。"""
    staked = returned = hits = bets = 0
    n_race = 0
    for rid, r in test.groupby("race_id"):
        if len(r) < 3:
            continue
        act = r.dropna(subset=["rank_num"])
        a1 = act[act["rank_num"] == 1]["umaban"].values
        a2 = act[act["rank_num"] == 2]["umaban"].values
        a3 = act[act["rank_num"] == 3]["umaban"].values
        if len(a1) == 0 or len(a2) == 0 or len(a3) == 0:
            continue
        n_race += 1
        a1, a2, a3 = a1[0], a2[0], a3[0]
        for ttype, combo, _ in bet_fn(r):
            staked += STAKE
            bets += 1
            key = (rid, ttype)
            if key not in lut:
                continue
            win_combo, pay = lut[key]
            # 実配当は勝ち組合せに対して1つ。自分の買い目が一致すれば的中
            if combo == win_combo:
                returned += pay * (STAKE / 100.0)
                hits += 1
    roi = returned / staked * 100 if staked else 0
    hit_rate = hits / bets * 100 if bets else 0
    print(f"\n[{name}]")
    print(f"  賭け点数: {bets} 点 / 投資: {staked:,.0f}円 / 払戻: {returned:,.0f}円")
    print(f"  的中率: {hit_rate:.2f}%  |  回収率(ROI): {'★' if roi>=100 else ' '} {roi:.1f}% {'★黒字★' if roi>=100 else '(赤字)'}")
    return roi


def main():
    print("=== 回収率(ROI)バックテスト: 2026年 実配当 ===")
    lut = load_payouts()
    test = get_scores()

    def top2(r):
        rr = r.sort_values("p1", ascending=False)
        p1u = rr.iloc[0]["umaban"]
        rest = rr[rr["umaban"] != p1u].sort_values("p2", ascending=False)
        p2u = rest.iloc[0]["umaban"] if len(rest) else rr.iloc[1]["umaban"]
        return int(p1u), int(p2u)

    # 戦略1: 単勝 全レース本命(モデルtop1)ベタ買い
    def s_win_all(r):
        p1u, _ = top2(r)
        return [("単勝", str(p1u), None)]

    # 戦略2: 単勝 +EVのみ (モデル確率 × 実オッズ > 1.2)
    def s_win_ev(r):
        rr = r.sort_values("p1", ascending=False)
        top = rr.iloc[0]
        odds = top["odds_num"]
        if pd.notna(odds) and top["p1"] * odds > 1.2:
            return [("単勝", str(int(top["umaban"])), top["p1"])]
        return []

    # 戦略3: 馬連単 (exacta) モデルtop1->top2 1点
    def s_exacta(r):
        p1u, p2u = top2(r)
        return [("馬連単", f"{p1u}-{p2u}", None)]

    # 戦略4: 馬連複 (quinella) モデルtop2 1点
    def s_quinella(r):
        p1u, p2u = top2(r)
        lo, hi = sorted([p1u, p2u])
        return [("馬連複", f"{lo}-{hi}", None)]

    # 戦略5: 馬連単 +EVのみ (モデルの馬単確率 p1*p2_cond の代理 × 実オッズ)
    def s_exacta_ev(r):
        rr = r.sort_values("p1", ascending=False)
        p1u = rr.iloc[0]["umaban"]
        rest = rr[rr["umaban"] != p1u].sort_values("p2", ascending=False)
        if len(rest) == 0:
            return []
        p2row = rest.iloc[0]
        joint = rr.iloc[0]["p1"] * p2row["p2"]  # 独立近似の馬単確率
        key = (r["race_id"].iloc[0], "馬連単")
        if key in lut:
            _, pay = lut[key]
            ev = joint * (pay / 100.0)  # 実配当ベースの期待値
            if ev > 1.3:
                return [("馬連単", f"{int(p1u)}-{int(p2row['umaban'])}", joint)]
        return []

    run_strategy(test, lut, "戦略1: 単勝ベタ買い(本命)", s_win_all)
    run_strategy(test, lut, "戦略2: 単勝 +EV厳選(p×odds>1.2)", s_win_ev)
    run_strategy(test, lut, "戦略3: 馬連単 1点(top1->top2)", s_exacta)
    run_strategy(test, lut, "戦略4: 馬連複 1点(top2)", s_quinella)
    run_strategy(test, lut, "戦略5: 馬連単 +EV厳選", s_exacta_ev)
    print("\n※ ROI 100%超 = 黒字。競馬の控除率で市場ベタ買いは通常75〜80%程度に沈む。")


if __name__ == "__main__":
    main()
