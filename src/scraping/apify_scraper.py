import json
import os
import time
from pathlib import Path

import pandas as pd
from dotenv import load_dotenv
from apify_client import ApifyClient

load_dotenv()

APIFY_TOKEN = os.getenv("APIFY_API_TOKEN")
ACTOR_ID = "PJDFRxJdpPknN9nYn"

if not APIFY_TOKEN:
    raise ValueError("APIFY_API_TOKEN not found in .env")

STOCKS = [
    {"ticker": "9988.HK", "name": "Alibaba", "keyword": "阿里巴巴"},
    {"ticker": "3690.HK", "name": "Meituan", "keyword": "美團"},
    {"ticker": "1810.HK", "name": "Xiaomi", "keyword": "小米"},
    {"ticker": "0941.HK", "name": "China Mobile", "keyword": "中國移動"},
]

def main():
    client = ApifyClient(APIFY_TOKEN)
    all_posts = []

    for stock in STOCKS:
        print("\nCollecting:", stock["name"])

        run_input = {
            "aiLabels": False,
            "aiTopic": stock["name"] + " (" + stock["ticker"] + ")",
            "catIds": ["1"],
            "keywords": [stock["keyword"]],
            "maxItems": 5,
            "mode": "search",
            "onlyNew": False,
            "order": "desc_reply_time",
            "threadIds": [],
            "threadUrls": [],
            "bulkInput": "",
            "proxyConfiguration": {"useApifyProxy": True},
        }

        run = client.actor(ACTOR_ID).call(run_input=run_input)

        if not run:
            print("WARNING: No run returned.")
            continue

        print("Run ID:", run.id)
        print("Status:", run.status)

        if run.status != "SUCCEEDED":
            print("WARNING: Actor did not succeed for", stock["name"])
            continue

        dataset_id = run.default_dataset_id
        if not dataset_id:
            print("WARNING: No dataset ID.")
            continue

        items = client.dataset(dataset_id).list_items().items

        for item in items:
            item["ticker"] = stock["ticker"]
            item["stock_name"] = stock["name"]
            item["exchange"] = "Hong Kong"
            item["source"] = "LIHKG"
            all_posts.append(item)

        print("Records collected:", len(items))
        time.sleep(2)

    output_file = Path("data/raw/lihkg_hk_stocks.json")
    output_file.parent.mkdir(parents=True, exist_ok=True)

    with open(output_file, "w", encoding="utf-8") as file:
        json.dump(all_posts, file, ensure_ascii=False, indent=2)

    if all_posts:
        csv_file = Path("data/raw/lihkg_hk_stocks.csv")
        pd.DataFrame(all_posts).to_csv(
            csv_file, index=False, encoding="utf-8-sig"
        )
        print("CSV saved:", csv_file)

    print("\nFINAL SUMMARY")
    print("=" * 35)
    print("Total records:", len(all_posts))

    if all_posts:
        print(pd.DataFrame(all_posts).groupby("stock_name").size())

if __name__ == "__main__":
    main()
