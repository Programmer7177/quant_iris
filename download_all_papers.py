"""
DOWNLOAD ALL 30 CURATED ARXIV QUANT FINANCE PAPERS INTO jurnal/
"""
import os
import json
import urllib.request
import time

JURNAL_DIR = r"C:\Mirza Personal\crypto quant\jurnal"
os.makedirs(JURNAL_DIR, exist_ok=True)

f1 = r'C:\Users\mirza\arxiv_crypto_forecasting_papers.json'
f2 = r'C:\Users\mirza\arxiv_quant_finance_papers.json'

with open(f1) as fp1, open(f2) as fp2:
    d1 = json.load(fp1)
    d2 = json.load(fp2)

papers = {}
for x in d1:
    aid = (x.get('arxiv_id') or x.get('id')).split('v')[0].strip()
    papers[aid] = x.get('title')

for k, v in d2.items():
    if isinstance(v, list):
        for x in v:
            aid = (x.get('arxiv_id') or x.get('id')).split('v')[0].strip()
            papers[aid] = x.get('title')

print(f"Total target papers to sync: {len(papers)}")

success = 0
for aid, title in papers.items():
    # clean filename
    safe_title = "".join(c if c.isalnum() else "_" for c in title)[:45]
    fn = f"{aid}_{safe_title}.pdf"
    out_path = os.path.join(JURNAL_DIR, fn)
    
    # check if any file starts with aid
    exists = any(f.startswith(aid) for f in os.listdir(JURNAL_DIR))
    if exists:
        print(f"[OK EXISTS] {aid} | {title[:50]}...")
        success += 1
        continue
        
    print(f"Downloading [{aid}] {title[:50]}...")
    url = f"https://export.arxiv.org/pdf/{aid}.pdf"
    try:
        req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64)"})
        with urllib.request.urlopen(req, timeout=30) as resp, open(out_path, "wb") as f:
            f.write(resp.read())
        print(f"  -> Downloaded: {fn}")
        success += 1
        time.sleep(1.5) # respect arxiv rate limit
    except Exception as e:
        print(f"  -> FAILED {aid}: {e}")

print(f"\nSelesai! Total file tersimpan: {len(os.listdir(JURNAL_DIR))}")
