import os
import time
import requests
import pandas as pd
from bs4 import BeautifulSoup

"""
三連単・三連複の実配当を取得する(既存payoutsが取り逃していた)。
既存2026レース(race_id)を元に対象日・R番号を復元して払戻テーブルを再取得。
rowspan継続行と漢字「三連単/三連複」表記に対応。
出力: data/raw/banei_trifecta_payouts_2026.csv (gitignore対象)
"""

HEADERS = {"User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 Chrome/124.0.0.0 Safari/537.36"}
TYPES = ["単勝", "複勝", "枠連複", "枠連単", "馬連複", "馬連単", "ワイド", "三連複", "三連単"]


def parse_payouts(soup, race_id):
    rows = []
    cur_type = None
    for table in soup.find_all("table")[1:]:  # table0=成績
        for tr in table.find_all("tr"):
            cells = [c.get_text(strip=True) for c in tr.find_all(["th", "td"])]
            if len(cells) == 4:
                cur_type, combo, pay = cells[0], cells[1], cells[2]
            elif len(cells) == 3:  # rowspan継続行 (先頭の券種セルが省略)
                combo, pay = cells[0], cells[1]
            else:
                continue
            if cur_type in TYPES:
                rows.append({"race_id": race_id, "ticket_type": cur_type,
                             "combination": combo, "payout": pay})
    return rows


def main():
    res = pd.read_csv("data/raw/banei_race_results_2026.csv")
    race_ids = res["race_id"].astype(str).unique()
    print(f"対象レース: {len(race_ids)}")

    s = requests.Session(); s.headers.update(HEADERS)
    all_rows = []
    for i, rid in enumerate(race_ids):
        # race_id = YYYYMMDD + '33' + RR
        ymd, rno = rid[:8], int(rid[10:12])
        date = f"{ymd[:4]}/{ymd[4:6]}/{ymd[6:8]}"
        url = f"https://www.keiba.go.jp/KeibaWeb/TodayRaceInfo/RaceMarkTable?k_raceDate={requests.utils.quote(date)}&k_raceNo={rno}&k_babaCode=3"
        try:
            r = s.get(url, timeout=12); r.encoding = "utf-8"
            soup = BeautifulSoup(r.text, "html.parser")
            all_rows.extend(parse_payouts(soup, rid))
        except Exception as e:
            print(f"  skip {rid}: {type(e).__name__}")
        if (i + 1) % 50 == 0:
            n3 = sum(1 for x in all_rows if x["ticket_type"] == "三連単")
            print(f"  {i+1}/{len(race_ids)} 完了 (三連単 {n3}件)")
        time.sleep(0.1)
    s.close()

    df = pd.DataFrame(all_rows)
    out = "data/raw/banei_trifecta_payouts_2026.csv"
    df.to_csv(out, index=False, encoding="utf-8-sig")
    print(f"完了 -> {out} (全 {len(df)} 行, 三連単 {len(df[df.ticket_type=='三連単'])}件)")


if __name__ == "__main__":
    main()
