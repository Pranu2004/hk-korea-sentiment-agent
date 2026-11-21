import os
from dotenv import load_dotenv

load_dotenv()

apify_token = os.getenv("APIFY_API_TOKEN")
exa_key = os.getenv("EXA_API_KEY")

print("Apify API key:", "Found" if apify_token else "Missing")
print("Exa API key:", "Found" if exa_key else "Missing")
