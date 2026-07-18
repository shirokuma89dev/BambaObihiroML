import argparse
import os
import re
import time
from datetime import datetime
import requests
from bs4 import BeautifulSoup
import pandas as pd

HEADERS = {
    "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36"
}

def create_session():
    session = requests.Session()
    session.headers.update(HEADERS)
    return session

def fetch_today_race_list(session, date_str):
    """
    指定日 (YYYY/MM/DD) の帯広全レース一覧および馬場水分量を取得
    """
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
                        race_metas.append({
                            "date": date_str.replace("/", "-"),
                            "race_no": r_no,
                            "start_time": row[1],
                            "race_name": row[4],
                            "distance": row[5],
                            "weather": row[6],
                            "track_moisture": float(row[7]) if row[7].replace(".", "", 1).isdigit() else None
                        })
                    except Exception:
                        continue
        return race_metas
    except Exception as e:
        print(f"Error fetching today race list ({date_str}): {e}")
        return []

def fetch_race_card(session, date_str, race_no, meta_info):
    """
    指定レースの出走表（DebaTable）から出走馬・重量・騎手・オッズ・当日の増減馬体重を取得
    """
    url = f"https://www.keiba.go.jp/KeibaWeb/TodayRaceInfo/DebaTable?k_raceDate={requests.utils.quote(date_str)}&k_raceNo={race_no}&k_babaCode=3"
    try:
        res = session.get(url, timeout=10)
        if res.status_code != 200:
            return None
        res.encoding = "utf-8"
        soup = BeautifulSoup(res.text, "html.parser")
        
        tables = soup.find_all("table")
        if not tables:
            return None
            
        race_id = f"{date_str.replace('/', '')}33{race_no:02d}"
        rows = []
        
        # DebaTable から出走情報抽出
        current_horse = None
        last_waku = ""
        for table in tables:
            for tr in table.find_all("tr"):
                tds = [td.get_text(strip=True) for td in tr.find_all(["th", "td"])]
                if len(tds) >= 3 and tds[0].isdigit():
                    # 前の馬のデータを保存
                    if current_horse is not None:
                        rows.append(current_horse)
                    
                    if len(tds) >= 4 and tds[1].isdigit():
                        # Case A: waku と umaban が両方存在
                        waku = tds[0]
                        umaban = tds[1]
                        horse_name = tds[2]
                        jockey_info = tds[3]
                        odds_pop_text = tds[4] if len(tds) > 4 else ""
                        last_waku = waku
                    else:
                        # Case B: waku が省略され、tds[0] が umaban
                        waku = last_waku
                        umaban = tds[0]
                        horse_name = tds[1]
                        jockey_info = tds[2] if len(tds) > 2 else ""
                        odds_pop_text = tds[3] if len(tds) > 3 else ""
                    
                    # オッズと人気順をパース
                    odds_text = ""
                    popularity_text = ""
                    if odds_pop_text:
                        m_odds = re.search(r"([\d.]+)", odds_pop_text)
                        if m_odds:
                            odds_text = m_odds.group(1)
                        m_pop = re.search(r"\((\d+)人気\)", odds_pop_text)
                        if m_pop:
                            popularity_text = m_pop.group(1)
                            
                    current_horse = {
                        "race_id": race_id,
                        "date": meta_info["date"],
                        "race_no": race_no,
                        "start_time": meta_info["start_time"],
                        "race_name": meta_info["race_name"],
                        "weather": meta_info["weather"],
                        "track_moisture": meta_info["track_moisture"],
                        "waku": waku,
                        "umaban": umaban,
                        "horse_name": horse_name,
                        "sled_weight": "",
                        "jockey_name": jockey_info,
                        "horse_weight": "",
                        "odds": odds_text,
                        "popularity": popularity_text
                    }
                elif current_horse is not None:
                    # サブ行からソリ重量と馬体重を抽出
                    # ソリ重量 (例: "☆ 460　0-0-0-0") は index 3 に配置（減量マーク対応で先頭マッチを外す）
                    if len(tds) >= 4:
                        m_sled = re.search(r"(\d{3,4})(?:\s|　)*\d+-\d+-\d+-\d+", tds[3])
                        if m_sled:
                            current_horse["sled_weight"] = m_sled.group(1)
                    # パドック馬体重 (例: "1061(+26)" または増減なし "852") は index 2 に配置
                    if len(tds) >= 3:
                        m_hw = re.search(r"\d{3,4}(?:\([+-]?\d+\))?", tds[2])
                        if m_hw:
                            current_horse["horse_weight"] = m_hw.group(0)
                            
        # 最後の馬を保存
        if current_horse is not None:
            rows.append(current_horse)
                    
        return pd.DataFrame(rows)
    except Exception as e:
        print(f"Error fetching race card for race {race_no} on {date_str}: {e}")
        return None

def scrape_race_day_card(date_str=None, output_dir="data/raw"):
    """
    当日の推論用出走データを取得
    date_str: "YYYY/MM/DD" (省略時は本日の日付)
    """
    if not date_str:
        date_str = datetime.now().strftime("%Y/%m/%d")
        
    os.makedirs(output_dir, exist_ok=True)
    session = create_session()
    
    print(f"--- 帯広競馬場 ({date_str}) 出走表・推論用データ収集開始 ---")
    race_metas = fetch_today_race_list(session, date_str)
    
    if not race_metas:
        print(f"[{date_str}] 本日の帯広競馬のレース開催情報はありませんでした。")
        session.close()
        return None
        
    all_cards = []
    for meta in race_metas:
        race_no = meta["race_no"]
        df_card = fetch_race_card(session, date_str, race_no, meta)
        if df_card is not None and not df_card.empty:
            all_cards.append(df_card)
            print(f"  [Race {race_no}R] 出走表取得成功: {meta['race_name']} (出走 {len(df_card)} 頭)")
        time.sleep(0.2)
        
    session.close()
    
    if all_cards:
        final_card_df = pd.concat(all_cards, ignore_index=True)
        filename = f"banei_race_card_{date_str.replace('/', '')}.csv"
        out_path = os.path.join(output_dir, filename)
        final_card_df.to_csv(out_path, index=False, encoding="utf-8-sig")
        print(f"完了: 推論用出走データ保存 -> {out_path} (全 {len(final_card_df)} レコード)")
        return out_path
    else:
        print("警告: 出走表データを取得できませんでした")
        return None

def main():
    parser = argparse.ArgumentParser(description="帯広（ばんえい）当日出走表・推論用データスクレイパー")
    parser.add_argument("--date", type=str, default="", help="対象日付 (YYYY/MM/DD 形式, 例: 2024/02/11)。指定しない場合は本日。")
    parser.add_argument("--out-dir", type=str, default="data/raw", help="出力先ディレクトリ")
    
    args = parser.parse_args()
    scrape_race_day_card(date_str=args.date, output_dir=args.out_dir)

if __name__ == "__main__":
    main()
