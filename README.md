# Hong Kong & South Korea Market Sentiment Analysis Agent

A Python-based market monitoring and sentiment analysis project for selected Hong Kong and South Korean equities. It collects financial news and public social posts, performs multilingual sentiment classification, aligns posts with exchange trading sessions, and presents market information through an interactive Streamlit dashboard.

## Project Overview

The project monitors 10 selected stocks:

### Hong Kong
| Ticker | Company |
|---|---|
| 0700.HK | Tencent |
| 9988.HK | Alibaba |
| 3690.HK | Meituan |
| 1810.HK | Xiaomi |
| 0941.HK | China Mobile |

### South Korea
| Ticker | Company |
|---|---|
| 005930.KS | Samsung Electronics |
| 000660.KS | SK Hynix |
| 005380.KS | Hyundai Motor |
| 373220.KS | LG Energy Solution |
| 035420.KS | NAVER |

## Features

- Financial news collection using Exa.
- Public social-post collection using Apify actors.
- Chinese sentiment analysis using SnowNLP.
- Korean financial sentiment analysis using KR-FinBERT-SC.
- English sentiment analysis using VADER.
- UTC timestamp normalization and exchange-session alignment.
- Daily sentiment aggregation by stock.
- Historical stock-price collection using Yahoo Finance.
- Technical features including moving averages, volatility, support and resistance.
- Ridge Regression benchmark with chronological out-of-sample evaluation.
- Interactive Streamlit dashboard with Plotly charts.
- Separate display of social posts that are awaiting a subsequent trading-price bar.

## Architecture

```text
Exa Financial News       Apify Public Social Sources
        |                         |
        v                         v
 Financial News CSV          Raw Social JSON
        |                         |
        v                         v
 VADER Sentiment       SnowNLP / KR-FinBERT-SC
        |                         |
        +------------+------------+
                     |
                     v
           Timestamp Normalization
                     |
                     v
          Exchange Trading Alignment
                     |
                     v
          Daily Sentiment Aggregation
                     |
                     v
             Yahoo Finance Prices
                     |
           +---------+---------+
           |                   |
           v                   v
    Feature Engineering   Current Monitor
           |                   |
           v                   v
     Ridge Benchmark     Streamlit Dashboard
```

## Technology Stack

- Python
- Pandas and NumPy
- Scikit-learn
- Yahoo Finance (`yfinance`)
- Apify Python Client
- Exa Python SDK
- VADER Sentiment
- SnowNLP
- PyTorch and Hugging Face Transformers
- KR-FinBERT-SC
- Streamlit and Plotly

## Project Structure

```text
hengseng-sentiment-agent/
|
|-- dashboard.py
|-- main.py
|-- requirements.txt
|-- README.md
|-- .env.example
|-- .gitignore
|
|-- data/
|   |-- raw/
|   |-- processed/
|
|-- reports/
|
|-- src/
|   |-- scraping/
|   |   |-- apify_scraper.py
|   |   |-- news_scraper.py
|   |   |-- process_lihkg.py
|   |   |-- process_naver_data.py
|   |   |-- combine_lihkg_data.py
|   |
|   |-- analysis/
|       |-- price_data.py
|       |-- sentiment.py
|       |-- lihkg_sentiment.py
|       |-- naver_sentiment.py
|       |-- telegram_sentiment.py
|       |-- feature_engineering.py
|       |-- sentiment_features.py
|       |-- sentiment_alignment.py
|       |-- align_naver_sentiment.py
|       |-- align_telegram_sentiment.py
|       |-- combine_sentiment_prices.py
|       |-- combine_all_sentiment.py
|       |-- build_current_monitor.py
|       |-- model.py
|       |-- prediction.py
|       |-- report.py
|
|-- tests/
```

## Setup

### 1. Clone the repository

```bash
git clone YOUR_GITHUB_REPOSITORY_URL
cd hengseng-sentiment-agent
```

### 2. Create and activate a virtual environment

Windows PowerShell:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

### 3. Install dependencies

Install the CPU version of PyTorch first:

```powershell
python -m pip install torch==2.15.0 --index-url https://download.pytorch.org/whl/cpu
```

Then install the remaining dependencies:

```powershell
python -m pip install -r requirements.txt
```

### 4. Configure API credentials

Create a `.env` file in the project root using `.env.example`:

```text
APIFY_API_TOKEN=your_apify_token
EXA_API_KEY=your_exa_api_key
```

Never commit `.env` or publish API tokens. Use your own credentials and permitted API access.

## Running the Project

Activate the virtual environment before running scripts.

### Collect stock prices

```powershell
python -m src.analysis.price_data
```

### Collect financial news

```powershell
python -m src.scraping.news_scraper
```

### Process news sentiment

```powershell
python -m src.analysis.sentiment
```

### Build daily news sentiment features

```powershell
python -m src.analysis.sentiment_features
```

### Build price features

```powershell
python -m src.analysis.feature_engineering
```

### Process social sentiment

Run the applicable processing scripts after collecting and exporting the corresponding Apify datasets:

```powershell
python -m src.scraping.process_lihkg
python -m src.scraping.process_naver_data
python -m src.analysis.lihkg_sentiment
python -m src.analysis.naver_sentiment
python -m src.analysis.telegram_sentiment
```

### Align social posts with trading sessions

```powershell
python -m src.analysis.combine_sentiment_prices
python -m src.analysis.align_naver_sentiment
python -m src.analysis.align_telegram_sentiment
```

### Combine sentiment sources and build the current monitor

```powershell
python -m src.analysis.combine_all_sentiment
python -m src.analysis.build_current_monitor
```

### Run the price-only benchmark

```powershell
python -m src.analysis.model
```

### Launch the dashboard

```powershell
streamlit run dashboard.py
```

Open the local URL displayed by Streamlit, usually:

`http://localhost:8501`

Some processing scripts depend on files generated by earlier stages. Run the collection and processing steps in the appropriate order.

## Data and Model Results

The current experiment collected:

- 10 selected equities across two markets.
- 4,895 historical stock-price records.
- 97 financial news articles.
- 63 news articles aligned to available trading sessions.
- 12 recent LIHKG posts, of which 9 were aligned.
- 100 Naver discussion posts, of which 9 were aligned to an available price bar.
- 1 stock-specific Telegram post, awaiting the next trading-price bar.

Social-source coverage was incomplete. Some sources returned limited or no relevant posts for selected stocks. The collected Naver posts covered approximately 34 hours rather than the full requested 10-day period.

### Benchmark evaluation

A Ridge Regression model was evaluated using a chronological train/test split.

| Metric | Result |
|---|---:|
| Training rows | 3,745 |
| Test rows | 940 |
| Test split date | 2026-07-07 |
| Model price MAE | 10,603.51 |
| Baseline price MAE | 10,437.46 |
| Model price RMSE | 33,449.30 |
| Baseline price RMSE | 32,818.49 |
| Model directional accuracy | 49.68% |

The benchmark uses historical price-derived features. It does not yet use sentiment as a trained predictive feature. Its overall error was slightly worse than the previous-close baseline, so it should not be interpreted as a validated profitable trading signal.

## Important Limitations

- Social data availability varies by platform, stock and collection period.
- Some collected posts are pending alignment because the next trading session has not occurred yet.
- Sentiment models can misunderstand financial context, sarcasm and market-specific language.
- The current model is a price-feature benchmark, not a validated sentiment-driven forecasting model.
- Support and resistance levels are historical technical references, not guaranteed price levels.
- This project is for research and demonstration, not financial advice.

## Security

- Keep `.env` private.
- Do not commit API keys, access tokens or private credentials.
- Review `.gitignore` before pushing the repository.
- Share credentials with evaluators only through an approved secure channel, if explicitly required.

## Author

Developed as an AI Agents Developer Internship assignment.

## Data Collection Actor IDs

- LIHKG Scraper: `PJDFRxJdpPknN9nYn`
- Naver Stock Discussion Scraper: `hgservices/naver-stock-discussion-scraper`
- Telegram Channel Scraper: `datalume/telegram-channel-scraper`
- Exa: News search API (API key configured locally in `.env`; not an Apify actor).

Only successful, usable source runs are included. Xueqiu was tested but returned no usable records and was excluded.

API credentials are intentionally excluded from this public repository.
