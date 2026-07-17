import requests
import re
from bs4 import BeautifulSoup
url = "https://www.data.jma.go.jp/obd/stats/etrn/view/daily_s1.php?prec_no=71&block_no=47417&year=2021&month=1&day=&view="
headers = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64)"}
r = requests.get(url, headers=headers)
r.encoding = r.apparent_encoding
soup = BeautifulSoup(r.text, "html.parser")
tables = soup.find_all("table")
for i, t in enumerate(tables):
    print(f"Table {i}: id={t.get('id')}, class={t.get('class')}")
