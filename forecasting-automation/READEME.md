# Building an Automated Ad Spend Forecasting System

_Pull, store, and predict your ad spend with code_

Right now, you’re probably manually pulling spend data from Google Ads or Meta Ads, copy-pasting it into spreadsheets, and trying to eyeball where your ad budget is going.

What we want instead:

- Automatically pull the last 24 months of ad spend from Google Ads and Meta Ads APIs.

- Store that into a Google Sheet, updated regularly.

- Use a forecasting model (like Prophet, ARIMA, or ETS) to predict the next 12 months of spend.

- Schedule it to run on its own, hands-free.

---

## The Stack: Tools of the Trade

We’ll build this with a hybrid of modern dev tools and cloud services:

### Data Collection

- **Google Ads API**
- **Meta Ads API**
- **Shopify / Amazon Seller APIs** (for revenue / aMER)

### Data Storage & Sync

- **Google BigQuery** (main database)
- **Google Sheets API** (for visibility)
- **Cloud Scheduler + Cloud Functions** (automated jobs)

### Forecasting

- [**Prophet**](https://otexts.com/fpp3/prophet.html) (seasonality-focused time series)
- [**ARIMA / ETS**](https://otexts.com/fpp3/arima.html) (classical forecasting)

### Insights

- Optional: GPT-style models to explain anomalies or simulate future scenarios

---

## Step-by-Step: How We Build It

### 1. **Automate the Data Ingestion**

Set up a daily pipeline that fetches the last 30 days of data from all connected ad platforms.

```python
# Cloud Function - Example Pseudocode
def fetch_google_ads_data():
    client = authenticate_google_ads()
    campaigns = client.get_campaigns(date_range='LAST_30_DAYS')
    return normalize_data(campaigns)

def store_to_bigquery(data):
    bigquery_client.insert_rows('ad_spend_table', data)

fetch_google_ads_data() → store_to_bigquery()
```

```python
from facebook_business.api import FacebookAdsApi
from facebook_business.adobjects.adaccount import AdAccount

FacebookAdsApi.init(app_id, app_secret, access_token)
account = AdAccount('act_<ad_account_id>')
insights = account.get_insights(fields=['spend'], params={'date_preset': 'last_30_days'})
```

```python
from sp_api.api import Orders
orders = Orders().get_orders(CreatedAfter='2025-06-01')
total_revenue = sum([float(order['OrderTotal']['Amount']) for order in orders.payload['Orders']])
```

```python
import gspread
gc = gspread.service_account(filename='leo.json')
sh = gc.open("Forecast Model").worksheet("June Spend")
sh.update('A2', [[date, revenue, meta_spend, google_spend, amazon_rev]])
```

This runs every day via **Cloud Scheduler**, keeps BigQuery fresh, and updates Google Sheets for visibility.

---

### 2. **Forecast the Next 12 Months (Automatically)**

Now the fun part. Using Prophet or ARIMA, we train on the last 24 months of data and forecast ahead.

```python
from prophet import Prophet

df = bigquery_to_df('ad_spend_table')
model = Prophet()
model.fit(df[['ds', 'y']])
future = model.make_future_dataframe(periods=365)
forecast = model.predict(future)
```

---

### 3. **Make It Adaptive: Smart Automation**

We don’t just want predictions — we want **feedback loops** and intelligent behavior.

#### Retrain Models Automatically

Run weekly retraining jobs using updated BigQuery data.

#### Detect Anomalies

If actual spend or aMER deviates too far from forecast (say ±15%), trigger an alert:

```python
error = abs(actual - predicted) / predicted
if error > 0.15:
    send_slack("⚠️ Spend deviation on Meta Ads!")
```

#### Budget Suggestions

Use the model to simulate:

- “What if we cut Meta by 20% and increased Google by 10%?”
- “Which campaign has the best aMER historically in June?”

We could even use a GPT model to write human-readable recommendations:

> "_Meta spend is 18% above forecast, but conversions are flat. Consider reallocating to Google._"

---

## Scalable Architecture

Here’s what the system looks like end-to-end:

```text
         Ad APIs         Revenue APIs
        (Google, Meta)    (Shopify, Amazon)
              │                 │
              ▼                 ▼
       ┌────────────┐   ┌────────────┐
       │  ETL Jobs  │   │ Revenue ETL│
       └────────────┘   └────────────┘
              ▼
         ┌──────────────┐
         │   BigQuery   │
         └──────────────┘
              ▼
    ┌───────────────────────┐
    │ Forecasting Engine    │
    │ (Prophet + ML + GPT)  │
    └───────────────────────┘
              ▼
        ┌────────────┐
        │ Google Sheet│ (for humans)
        └────────────┘
              │
         Slack / Email
        (alerts + summaries)
```

Run everything in:

- **Cloud Functions** (fast, scalable, serverless)
- Optionally **Docker + Cloud Run** if you need more control
- **Airflow via Cloud Composer** if orchestration gets complex

---

## Monitoring & Maintenance

We’ll want:

- Logging in Cloud Logging or Datadog
- Fallback logic if APIs fail
- Versioning with DVC or MLflow
- Data quality checks (duplicates, nulls, etc.)
- Monthly performance summaries via Slack/Email

---

## TL;DR

We're evolving from:

> _"Copy-paste ad spend, guess next month’s budget..."_

to:

> **"Smart pipeline that forecasts, detects surprises, and helps guide strategy."**

---
