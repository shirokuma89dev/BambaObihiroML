import os
import re
import sys
import time
import glob
import pandas as pd
import requests
from bs4 import BeautifulSoup

HEADERS = {
    "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36"
}
CODE_MAP_PATH = "data/processed/horse_codes.csv"
PEDIGREE_MAP_PATH = "data/processed/pedigree_map.csv"


def get_unique_horses():
    """過去の全レースデータからユニークな馬名を取得"""
    files = glob.glob("data/raw/banei_race_results_*.csv")
    if not files:
        print("エラー: data/raw/ に過去のレース結果(results)が見つかりません。")
        sys.exit(1)
        
    df = pd.concat([pd.read_csv(f) for f in files], ignore_index=True)
    unique_horses = set(df["horse_name"].dropna().unique())
    
    # レースの逆順検索用に、レース単位の(date, race_no)リストも作成
    # 日付を YYYY/MM/DD 形式にしてソート
    df["date_slash"] = df["date"].str.replace("-", "/")
    races = df[["date_slash", "race_no"]].drop_duplicates().sort_values(
        by=["date_slash", "race_no"], ascending=[False, False]
    ).values.tolist()
    
    return unique_horses, races


def build_horse_code_map(unique_horses, races):
    """過去レース結果ページを遡り、馬名と登録コードのマップを構築"""
    os.makedirs("data/processed", exist_ok=True)
    
    # 既存のマップがあればロード
    code_map = {}
    if os.path.exists(CODE_MAP_PATH):
        cdf = pd.read_csv(CODE_MAP_PATH)
        code_map = dict(zip(cdf["horse_name"], cdf["code"]))
        print(f"既存の馬コードマップをロード: {len(code_map)}頭分")
        
    missing_horses = unique_horses - set(code_map.keys())
    if not missing_horses:
        print("すべての馬のコードがすでに取得済みです。")
        return code_map
        
    print(f"未取得の馬: {len(missing_horses)} / {len(unique_horses)} 頭")
    print("過去レース結果を遡って登録コードを探索します...")
    
    session = requests.Session()
    session.headers.update(HEADERS)
    
    found_count = 0
    for idx, (date_str, race_no) in enumerate(races):
        if not missing_horses:
            print("すべての馬のコードを特定しました！")
            break
            
        # サーバー負荷防止
        if idx > 0 and idx % 20 == 0:
            print(f"進捗: {idx}レース解析完了 ... 残りの未特定馬: {len(missing_horses)}頭")
            # 中間セーブ
            temp_df = pd.DataFrame(list(code_map.items()), columns=["horse_name", "code"])
            temp_df.to_csv(CODE_MAP_PATH, index=False)
            time.sleep(1.0)
            
        url = f"https://www.keiba.go.jp/KeibaWeb/TodayRaceInfo/RaceMarkTable?k_raceDate={requests.utils.quote(date_str)}&k_raceNo={race_no}&k_babaCode=3"
        try:
            res = session.get(url, timeout=10)
            if res.status_code != 200:
                continue
            res.encoding = "utf-8"
            soup = BeautifulSoup(res.text, "html.parser")
            
            # 馬名リンクからコードを抽出
            links = soup.find_all("a")
            for a in links:
                href = a.get("href", "")
                name = a.get_text(strip=True)
                
                if name in missing_horses and "HorseMarkInfo" in href:
                    # href の中から k_lineageLoginCode=XXXXXXXXXX を抽出
                    m = re.search(r"k_lineageLoginCode=(\d+)", href)
                    if m:
                        code = m.group(1)
                        code_map[name] = code
                        missing_horses.remove(name)
                        found_count += 1
                        print(f"発見: {name} -> {code} (残り: {len(missing_horses)}頭)")
                        
        except Exception as e:
            print(f"レース {date_str} {race_no}R の取得でエラー: {e}")
            time.sleep(2.0)
            
    # 最終セーブ
    final_df = pd.DataFrame(list(code_map.items()), columns=["horse_name", "code"])
    final_df.to_csv(CODE_MAP_PATH, index=False)
    print(f"馬コード探索完了。探索済み: {len(code_map)}頭")
    return code_map


def clean_pedigree_name(name):
    """血統名から（日輓）や（半血）などの属性を取り除く"""
    if not isinstance(name, str):
        return ""
    # 全角括弧、半角括弧内の文字を消去
    name = re.sub(r"（[^）]+）", "", name)
    name = re.sub(r"\([^)]+\)", "", name)
    return name.strip()


def scrape_pedigrees(code_map):
    """各馬の登録コードを使って血統表をスクレイピング"""
    pedigree_map = {}
    if os.path.exists(PEDIGREE_MAP_PATH):
        pdf = pd.read_csv(PEDIGREE_MAP_PATH)
        # カラムが存在することを確認
        if "horse_name" in pdf.columns:
            for _, row in pdf.iterrows():
                pedigree_map[row["horse_name"]] = {
                    "sire": row.get("sire", ""),
                    "dam": row.get("dam", ""),
                    "dam_sire": row.get("dam_sire", "")
                }
            print(f"既存の血統マップをロード: {len(pedigree_map)}頭分")

    session = requests.Session()
    session.headers.update(HEADERS)
    
    missing_pedigrees = set(code_map.keys()) - set(pedigree_map.keys())
    if not missing_pedigrees:
        print("すべての馬の血統が取得済みです。")
        return
        
    print(f"未取得の血統: {len(missing_pedigrees)} / {len(code_map)} 頭")
    print("馬情報ページから血統表をスクレイピングします...")
    
    count = 0
    for name in sorted(missing_pedigrees):
        code = code_map[name]
        
        # 1秒間のウェイトを入れてサーバー負荷を避ける
        time.sleep(1.0)
        count += 1
        if count % 20 == 0:
            print(f"血統取得進捗: {count}頭完了 ...")
            # 中間セーブ
            save_pedigree_map(pedigree_map)
            
        url = f"https://www.keiba.go.jp/KeibaWeb/DataRoom/RaceHorseInfo?k_lineageLoginCode={code}&k_activeCode=1"
        try:
            res = session.get(url, timeout=10)
            if res.status_code != 200:
                print(f"警告: {name} の取得失敗 (Status: {res.status_code})")
                continue
                
            res.encoding = "utf-8"
            soup = BeautifulSoup(res.text, "html.parser")
            
            tables = soup.find_all("table")
            if len(tables) < 2:
                print(f"警告: {name} のページに血統表テーブルが見つかりません。")
                continue
                
            # Table 1 が血統表
            blood_table = tables[1]
            sire = ""
            dam = ""
            dam_sire = ""
            
            for tr in blood_table.find_all("tr"):
                tds = [td.get_text(strip=True) for td in tr.find_all(["th", "td"])]
                if len(tds) >= 2:
                    if tds[0] == "父":
                        sire = clean_pedigree_name(tds[1])
                    elif tds[0] == "母":
                        dam = clean_pedigree_name(tds[1])
                    elif tds[0] == "母父" or (len(tds) >= 4 and tds[2] == "母父"):
                        # レイアウトによってインデックスが異なる場合をケア
                        dam_sire = clean_pedigree_name(tds[3] if len(tds) >= 4 else tds[1])
            
            # 再度チェックし、見つからなかった場合のフォールバック
            # 母父は tr の中に複数あるため走査
            for tr in blood_table.find_all("tr"):
                texts = [td.get_text(strip=True) for td in tr.find_all(["th", "td"])]
                for i, txt in enumerate(texts):
                    if txt == "母父" and i + 1 < len(texts):
                        dam_sire = clean_pedigree_name(texts[i+1])
                        
            pedigree_map[name] = {
                "sire": sire,
                "dam": dam,
                "dam_sire": dam_sire
            }
            print(f"取得 [{count}]: {name} (父: {sire} / 母: {dam} / 母父: {dam_sire})")
            
        except Exception as e:
            print(f"馬 {name} (コード: {code}) の血統取得でエラー: {e}")
            
    # 最終セーブ
    save_pedigree_map(pedigree_map)
    print("血統データのスクレイピングが完了しました。")


def save_pedigree_map(pedigree_map):
    """血統データをCSVに保存"""
    rows = []
    for name, ped in pedigree_map.items():
        rows.append({
            "horse_name": name,
            "sire": ped["sire"],
            "dam": ped["dam"],
            "dam_sire": ped["dam_sire"]
        })
    df = pd.DataFrame(rows)
    df.to_csv(PEDIGREE_MAP_PATH, index=False)
    print(f"血統マップをセーブしました: {len(df)}頭分")


def main():
    print("=== 血統情報（親馬）の取得処理開始 ===")
    unique_horses, races = get_unique_horses()
    print(f"総ユニーク馬頭数: {len(unique_horses)}頭 | 総過去レース数: {len(races)}レース")
    
    code_map = build_horse_code_map(unique_horses, races)
    scrape_pedigrees(code_map)
    print("=== すべての処理が正常終了しました ===")


if __name__ == "__main__":
    main()
