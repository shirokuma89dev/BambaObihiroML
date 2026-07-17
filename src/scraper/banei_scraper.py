import argparse
import os
import re
import time
from datetime import datetime, timedelta

import pandas as pd
import requests
from bs4 import BeautifulSoup

HEADERS = {
    "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36"
}


def create_session():
    session = requests.Session()
    session.headers.update(HEADERS)
    return session


def fetch_day_races(session, date_str):
    url = f"https://www.keiba.go.jp/KeibaWeb/TodayRaceInfo/RaceList?k_babaCode=3&k_raceDate={requests.utils.quote(date_str)}"
    try:
        res = session.get(url, timeout=10)
        if res.status_code != 200:
            return []
        res.encoding = "utf-8"
        soup = BeautifulSoup(res.text, "html.parser")

        race_metas = []
        for table in soup.find_all("table"):
            for tr in table.find_all("tr"):
                row = [t.get_text(strip=True) for t in tr.find_all(["th", "td"])]
                if len(row) >= 8 and row[0].endswith("R"):
                    try:
                        r_no = int(row[0].replace("R", ""))
                        race_metas.append(
                            {
                                "date": date_str.replace("/", "-"),
                                "race_no": r_no,
                                "race_name": row[4],
                                "distance": row[5],
                                "weather": row[6],
                                "track_moisture": float(row[7])
                                if row[7].replace(".", "", 1).isdigit()
                                else None,
                            }
                        )
                    except Exception:
                        continue
        return race_metas
    except Exception as e:
        print(f"Error fetching race list for {date_str}: {e}")
        return []


def fetch_race_result(session, date_str, race_no, meta_info):
    url = f"https://www.keiba.go.jp/KeibaWeb/TodayRaceInfo/RaceMarkTable?k_raceDate={requests.utils.quote(date_str)}&k_raceNo={race_no}&k_babaCode=3"
    try:
        res = session.get(url, timeout=10)
        if res.status_code != 200:
            return None, None
        res.encoding = "utf-8"
        soup = BeautifulSoup(res.text, "html.parser")

        tables = soup.find_all("table")
        if not tables:
            return None, None

        race_id = f"{date_str.replace('/', '')}33{race_no:02d}"

        # 1. 成績テーブル
        result_table = tables[0]
        rows = []
        for tr in result_table.find_all("tr")[1:]:
            tds = [td.get_text(strip=True) for td in tr.find_all(["th", "td"])]
            if len(tds) >= 8:
                row_dict = {
                    "race_id": race_id,
                    "date": meta_info["date"],
                    "race_no": race_no,
                    "race_name": meta_info["race_name"],
                    "weather": meta_info["weather"],
                    "track_moisture": meta_info["track_moisture"],
                    "rank": tds[0] if len(tds) > 0 else "",
                    "waku": tds[1] if len(tds) > 1 else "",
                    "umaban": tds[2] if len(tds) > 2 else "",
                    "horse_name": tds[3] if len(tds) > 3 else "",
                    "sex_age": tds[5] if len(tds) > 5 else "",
                    "sled_weight": tds[6] if len(tds) > 6 else "",
                    "jockey_name": tds[7] if len(tds) > 7 else "",
                    "trainer_name": tds[8] if len(tds) > 8 else "",
                    "horse_weight": tds[9] if len(tds) > 9 else "",
                    "time": tds[10] if len(tds) > 10 else "",
                    "margin": tds[11] if len(tds) > 11 else "",
                    "popularity": tds[13] if len(tds) > 13 else "",
                    "odds": tds[14] if len(tds) > 14 else "",
                }
                rows.append(row_dict)

        # 2. 払戻金テーブル
        payout_rows = []
        for table in tables[1:]:
            for tr in table.find_all("tr"):
                tds = [td.get_text(strip=True) for td in tr.find_all(["th", "td"])]
                if len(tds) >= 3 and any(
                    k in tds[0]
                    for k in [
                        "単勝",
                        "複勝",
                        "枠連",
                        "馬連",
                        "馬単",
                        "ワイド",
                        "３連複",
                        "３連単",
                    ]
                ):
                    payout_rows.append(
                        {
                            "race_id": race_id,
                            "ticket_type": tds[0],
                            "combination": tds[1] if len(tds) > 1 else "",
                            "payout": tds[2] if len(tds) > 2 else "",
                            "popularity": tds[3] if len(tds) > 3 else "",
                        }
                    )

        df_results = pd.DataFrame(rows)
        df_payouts = pd.DataFrame(payout_rows)
        return df_results, df_payouts
    except Exception as e:
        print(f"Error fetching race {race_no} on {date_str}: {e}")
        return None, None


def generate_dates(start_date_str, end_date_str):
    start = datetime.strptime(start_date_str, "%Y-%m-%d")
    end = datetime.strptime(end_date_str, "%Y-%m-%d")
    now = datetime.now()
    if end > now:
        end = now
    curr = start
    while curr <= end:
        if curr.weekday() in [0, 4, 5, 6]:
            yield curr.strftime("%Y/%m/%d")
        curr += timedelta(days=1)


def scrape_banei_races(year=2026, max_races=2000, output_dir="data/raw"):
    os.makedirs(output_dir, exist_ok=True)
    session = create_session()
    all_results = []
    all_payouts = []
    meta_list = []

    print(f"--- 帯広競馬場 ({year}年) 全データ収集開始 ---")
    count = 0

    start_date = f"{year}-01-01"
    end_date = f"{year}-12-31"

    for date_str in generate_dates(start_date, end_date):
        day_races = fetch_day_races(session, date_str)
        if not day_races:
            continue

        for race_meta in day_races:
            race_no = race_meta["race_no"]
            df_res, df_pay = fetch_race_result(session, date_str, race_no, race_meta)

            if df_res is not None and not df_res.empty:
                all_results.append(df_res)
                if df_pay is not None and not df_pay.empty:
                    all_payouts.append(df_pay)

                meta_list.append(race_meta)
                count += 1
                race_id = f"{date_str.replace('/', '')}33{race_no:02d}"
                if count % 50 == 0:
                    print(
                        f"  [{count} レース取得完了] 直近 Race ID: {race_id} ({race_meta['race_name']})"
                    )

                if count >= max_races:
                    break
            time.sleep(0.12)

        if count >= max_races:
            break

    session.close()

    if all_results:
        final_df = pd.concat(all_results, ignore_index=True)
        meta_df = pd.DataFrame(meta_list)

        results_file = os.path.join(output_dir, f"banei_race_results_{year}.csv")
        meta_file = os.path.join(output_dir, f"banei_race_meta_{year}.csv")

        final_df.to_csv(results_file, index=False, encoding="utf-8-sig")
        meta_df.to_csv(meta_file, index=False, encoding="utf-8-sig")

        if all_payouts:
            payout_df = pd.concat(all_payouts, ignore_index=True)
            payout_file = os.path.join(output_dir, f"banei_race_payouts_{year}.csv")
            payout_df.to_csv(payout_file, index=False, encoding="utf-8-sig")

        print(
            f"完了: {year}年 データ保存完了 -> {results_file} (全 {len(final_df)} レコード)"
        )
        return results_file
    else:
        print(f"警告: {year}年のデータが取得できませんでした")
        return None


def main():
    current_year = datetime.now().year
    parser = argparse.ArgumentParser(
        description="帯広（ばんえい）競馬データ決定型スクレイパー"
    )
    parser.add_argument(
        "--start-year", type=int, default=2023, help="収集開始年 (デフォルト: 2023)"
    )
    parser.add_argument(
        "--end-year",
        type=int,
        default=current_year,
        help=f"収集終了年 (デフォルト: {current_year})",
    )
    parser.add_argument(
        "--max-races", type=int, default=2000, help="各年の最大レース取得数"
    )
    parser.add_argument(
        "--out-dir", type=str, default="data/raw", help="保存先ディレクトリ"
    )

    args = parser.parse_args()

    for y in range(args.start_year, args.end_year + 1):
        scrape_banei_races(year=y, max_races=args.max_races, output_dir=args.out_dir)


if __name__ == "__main__":
    main()
