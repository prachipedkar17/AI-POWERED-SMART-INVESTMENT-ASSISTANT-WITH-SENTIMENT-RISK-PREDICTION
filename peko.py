import os, re, logging, traceback
from datetime import datetime, timezone, timedelta
from functools import lru_cache
import json

from flask import Flask, jsonify, request, Response
import numpy as np, pandas as pd, yfinance as yf, requests
from textblob import TextBlob

from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping
 
# ---------------- Configuration ----------------
MODEL_PATH = "lstm_model.h5"
LOOKBACK = 30
DEFAULT_HORIZON = 14
NEWSAPI_KEY = os.getenv("NEWSAPI_KEY")
ALPHA_VANTAGE_KEY = os.getenv("ALPHA_VANTAGE_KEY")
MAX_NEWS = 8
TICKER_RE = re.compile(r"^[A-Za-z0-9.\-]+$")
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger("smart-invest")

POPULAR_STOCKS = [
    {"symbol": "AAPL", "name": "Apple Inc.", "sector": "Technology"},
    {"symbol": "MSFT", "name": "Microsoft Corporation", "sector": "Technology"},
    {"symbol": "GOOGL", "name": "Alphabet Inc.", "sector": "Technology"},
    {"symbol": "AMZN", "name": "Amazon.com Inc.", "sector": "Consumer Cyclical"},
    {"symbol": "TSLA", "name": "Tesla Inc.", "sector": "Automotive"},
    {"symbol": "META", "name": "Meta Platforms Inc.", "sector": "Communication Services"},
    {"symbol": "NVDA", "name": "NVIDIA Corporation", "sector": "Technology"},
    {"symbol": "NFLX", "name": "Netflix Inc.", "sector": "Communication Services"},
    {"symbol": "JPM", "name": "JPMorgan Chase & Co.", "sector": "Financial Services"},
    {"symbol": "V", "name": "Visa Inc.", "sector": "Financial Services"},
    {"symbol": "JNJ", "name": "Johnson & Johnson", "sector": "Healthcare"},
    {"symbol": "WMT", "name": "Walmart Inc.", "sector": "Consumer Defensive"},
    {"symbol": "PG", "name": "Procter & Gamble", "sector": "Consumer Defensive"},
    {"symbol": "MA", "name": "Mastercard Inc.", "sector": "Financial Services"},
    {"symbol": "DIS", "name": "Walt Disney Co.", "sector": "Communication Services"},
    {"symbol": "BAC", "name": "Bank of America", "sector": "Financial Services"},
    {"symbol": "XOM", "name": "Exxon Mobil Corp.", "sector": "Energy"},
    {"symbol": "HD", "name": "Home Depot Inc.", "sector": "Consumer Cyclical"},
    {"symbol": "PYPL", "name": "PayPal Holdings", "sector": "Financial Services"},
    {"symbol": "CSCO", "name": "Cisco Systems", "sector": "Technology"},
    {"symbol": "ADBE", "name": "Adobe Inc.", "sector": "Technology"},
    {"symbol": "CRM", "name": "Salesforce Inc.", "sector": "Technology"},
    {"symbol": "NKE", "name": "Nike Inc.", "sector": "Consumer Cyclical"},
    {"symbol": "PFE", "name": "Pfizer Inc.", "sector": "Healthcare"},
    {"symbol": "ABT", "name": "Abbott Laboratories", "sector": "Healthcare"},
    {"symbol": "TMO", "name": "Thermo Fisher Scientific", "sector": "Healthcare"},
    {"symbol": "AVGO", "name": "Broadcom Inc.", "sector": "Technology"},
    {"symbol": "COST", "name": "Costco Wholesale", "sector": "Consumer Defensive"},
    {"symbol": "ACN", "name": "Accenture plc", "sector": "Technology"},
    {"symbol": "T", "name": "AT&T Inc.", "sector": "Communication Services"},
    {"symbol": "DHR", "name": "Danaher Corporation", "sector": "Healthcare"},
    {"symbol": "VZ", "name": "Verizon Communications", "sector": "Communication Services"},
    {"symbol": "NEE", "name": "NextEra Energy", "sector": "Utilities"},
    {"symbol": "UNH", "name": "UnitedHealth Group", "sector": "Healthcare"},
    {"symbol": "LIN", "name": "Linde plc", "sector": "Basic Materials"},
    {"symbol": "RTX", "name": "Raytheon Technologies", "sector": "Industrials"},
    {"symbol": "HON", "name": "Honeywell International", "sector": "Industrials"},
    {"symbol": "SBUX", "name": "Starbucks Corporation", "sector": "Consumer Cyclical"},
    {"symbol": "LOW", "name": "Lowe's Companies", "sector": "Consumer Cyclical"},
    {"symbol": "BMY", "name": "Bristol-Myers Squibb", "sector": "Healthcare"},
    {"symbol": "INTC", "name": "Intel Corporation", "sector": "Technology"},
    {"symbol": "AXP", "name": "American Express", "sector": "Financial Services"},
    {"symbol": "AMD", "name": "Advanced Micro Devices", "sector": "Technology"},
    {"symbol": "IBM", "name": "International Business Machines", "sector": "Technology"},
    {"symbol": "CAT", "name": "Caterpillar Inc.", "sector": "Industrials"},
    {"symbol": "GS", "name": "Goldman Sachs Group", "sector": "Financial Services"},
    {"symbol": "UNP", "name": "Union Pacific Corporation", "sector": "Industrials"},
    {"symbol": "SPGI", "name": "S&P Global Inc.", "sector": "Financial Services"},
    {"symbol": "PLD", "name": "Prologis Inc.", "sector": "Real Estate"},
    {"symbol": "DE", "name": "Deere & Company", "sector": "Industrials"},
    {"symbol": "NOW", "name": "ServiceNow Inc.", "sector": "Technology"},
    {"symbol": "HDFCBANK.NS", "name": "HDFC Bank", "sector": "Financial Services"},
    {"symbol": "RELIANCE.NS", "name": "Reliance Industries", "sector": "Energy"},
    {"symbol": "TCS.NS", "name": "TCS", "sector": "Technology"},
    {"symbol": "INFY.NS", "name": "Infosys", "sector": "Technology"},
    {"symbol": "ICICIBANK.NS", "name": "ICICI Bank", "sector": "Financial Services"},
    {"symbol": "SBIN.NS", "name": "State Bank of India", "sector": "Financial Services"},
    {"symbol": "ITC.NS", "name": "ITC", "sector": "Consumer Defensive"},
    {"symbol": "WIPRO.NS", "name": "Wipro", "sector": "Technology"},
    {"symbol": "ADANIENT.NS", "name": "Adani Enterprises", "sector": "Conglomerate"},
    {"symbol": "BAJFINANCE.NS", "name": "Bajaj Finance", "sector": "Financial Services"},
    {"symbol": "HINDUNILVR.NS", "name": "Hindustan Unilever", "sector": "Consumer Defensive"},
    {"symbol": "BHARTIARTL.NS", "name": "Bharti Airtel", "sector": "Communication Services"},
    {"symbol": "KOTAKBANK.NS", "name": "Kotak Mahindra Bank", "sector": "Financial Services"},
    {"symbol": "LT.NS", "name": "Larsen & Toubro", "sector": "Industrials"},
    {"symbol": "AXISBANK.NS", "name": "Axis Bank", "sector": "Financial Services"},
    {"symbol": "ASIANPAINT.NS", "name": "Asian Paints", "sector": "Basic Materials"},
    {"symbol": "MARUTI.NS", "name": "Maruti Suzuki", "sector": "Automotive"},
    {"symbol": "TITAN.NS", "name": "Titan Company", "sector": "Consumer Cyclical"},
    {"symbol": "NTPC.NS", "name": "NTPC Limited", "sector": "Utilities"},
    {"symbol": "ONGC.NS", "name": "Oil and Natural Gas Corporation", "sector": "Energy"},
    {"symbol": "POWERGRID.NS", "name": "Power Grid Corporation", "sector": "Utilities"},
    {"symbol": "SUNPHARMA.NS", "name": "Sun Pharmaceutical", "sector": "Healthcare"},
    {"symbol": "INDUSINDBK.NS", "name": "IndusInd Bank", "sector": "Financial Services"},
    {"symbol": "COALINDIA.NS", "name": "Coal India", "sector": "Energy"},
    {"symbol": "ULTRACEMCO.NS", "name": "UltraTech Cement", "sector": "Basic Materials"},
    {"symbol": "HCLTECH.NS", "name": "HCL Technologies", "sector": "Technology"},
    {"symbol": "M&M.NS", "name": "Mahindra & Mahindra", "sector": "Automotive"},
    {"symbol": "TECHM.NS", "name": "Tech Mahindra", "sector": "Technology"},
    {"symbol": "BRITANNIA.NS", "name": "Britannia Industries", "sector": "Consumer Defensive"},
    {"symbol": "NESTLEIND.NS", "name": "Nestle India", "sector": "Consumer Defensive"},
    {"symbol": "GRASIM.NS", "name": "Grasim Industries", "sector": "Basic Materials"},
    {"symbol": "JSWSTEEL.NS", "name": "JSW Steel", "sector": "Basic Materials"},
    {"symbol": "BAJAJFINSV.NS", "name": "Bajaj Finserv", "sector": "Financial Services"},
    {"symbol": "TATAMOTORS.NS", "name": "Tata Motors", "sector": "Automotive"},
    {"symbol": "HINDALCO.NS", "name": "Hindalco Industries", "sector": "Basic Materials"},
    {"symbol": "DRREDDY.NS", "name": "Dr. Reddy's Laboratories", "sector": "Healthcare"},
    {"symbol": "DIVISLAB.NS", "name": "Divi's Laboratories", "sector": "Healthcare"},
    {"symbol": "CIPLA.NS", "name": "Cipla", "sector": "Healthcare"},
    {"symbol": "BPCL.NS", "name": "Bharat Petroleum", "sector": "Energy"},
    {"symbol": "HEROMOTOCO.NS", "name": "Hero MotoCorp", "sector": "Automotive"},
    {"symbol": "EICHERMOT.NS", "name": "Eicher Motors", "sector": "Automotive"},
    {"symbol": "ADANIPORTS.NS", "name": "Adani Ports", "sector": "Industrials"},
    {"symbol": "DLF.NS", "name": "DLF Limited", "sector": "Real Estate"},
    {"symbol": "SBILIFE.NS", "name": "SBI Life Insurance", "sector": "Financial Services"},
    {"symbol": "HINDZINC.NS", "name": "Hindustan Zinc", "sector": "Basic Materials"},
    {"symbol": "BERGEPAINT.NS", "name": "Berger Paints", "sector": "Basic Materials"},
    {"symbol": "DABUR.NS", "name": "Dabur India", "sector": "Consumer Defensive"},
    {"symbol": "PIDILITIND.NS", "name": "Pidilite Industries", "sector": "Basic Materials"},
    {"symbol": "HAVELLS.NS", "name": "Havells India", "sector": "Industrials"},
    {"symbol": "GODREJCP.NS", "name": "Godrej Consumer Products", "sector": "Consumer Defensive"},
    {"symbol": "BOSCHLTD.NS", "name": "Bosch Limited", "sector": "Automotive"},
    {"symbol": "BIOCON.NS", "name": "Biocon Limited", "sector": "Healthcare"},
    {"symbol": "RELIANCE.NS", "name": "Reliance Industries", "sector": "Energy"},
    {"symbol": "HDFCBANK.NS", "name": "HDFC Bank", "sector": "Financial Services"},
    {"symbol": "BHARTIARTL.NS", "name": "Bharti Airtel", "sector": "Telecommunications"},
    {"symbol": "SBIN.NS", "name": "State Bank of India", "sector": "Financial Services"},
    {"symbol": "ICICIBANK.NS", "name": "ICICI Bank", "sector": "Financial Services"},
    {"symbol": "TCS.NS", "name": "Tata Consultancy Services", "sector": "Information Technology"},
    {"symbol": "LT.NS", "name": "Larsen & Toubro", "sector": "Industrials"},
    {"symbol": "ITC.NS", "name": "ITC Limited", "sector": "Consumer Defensive"},
    {"symbol": "HINDUNILVR.NS", "name": "Hindustan Unilever", "sector": "Consumer Defensive"},
    {"symbol": "INFY.NS", "name": "Infosys", "sector": "Information Technology"},
    {"symbol": "AXISBANK.NS", "name": "Axis Bank", "sector": "Financial Services"},
    {"symbol": "TATAMOTORS.NS", "name": "Tata Motors", "sector": "Automotive"},
    {"symbol": "MARUTI.NS", "name": "Maruti Suzuki India", "sector": "Automotive"},
    {"symbol": "ONGC.NS", "name": "Oil & Natural Gas Corp", "sector": "Energy"},
    {"symbol": "JSWSTEEL.NS", "name": "JSW Steel", "sector": "Materials"},
    {"symbol": "NESTLEIND.NS", "name": "Nestlé India", "sector": "Consumer Defensive"},
    {"symbol": "ULTRACEMCO.NS", "name": "UltraTech Cement", "sector": "Materials"},
    {"symbol": "BAJFINANCE.NS", "name": "Bajaj Finance", "sector": "Financial Services"},
    {"symbol": "IBULHSGFIN.NS", "name": "Indiabulls Housing Finance", "sector": "Financial Services"},
    {"symbol": "HCLTECH.NS", "name": "HCL Technologies", "sector": "Information Technology"},
    {"symbol": "TECHM.NS", "name": "Tech Mahindra", "sector": "Information Technology"},
    {"symbol": "SUNPHARMA.NS", "name": "Sun Pharma", "sector": "Healthcare"},
    {"symbol": "TITAN.NS", "name": "Titan Company", "sector": "Consumer Cyclical"},
    {"symbol": "POWERGRID.NS", "name": "Power Grid Corporation of India", "sector": "Utilities"},
    {"symbol": "BPCL.NS", "name": "Bharat Petroleum", "sector": "Energy"},
    {"symbol": "GRASIM.NS", "name": "Grasim Industries", "sector": "Materials"},
    {"symbol": "INDUSINDBK.NS", "name": "IndusInd Bank", "sector": "Financial Services"},
    {"symbol": "TATASTEEL.NS", "name": "Tata Steel", "sector": "Materials"},
    {"symbol": "COALINDIA.NS", "name": "Coal India", "sector": "Energy"},
    {"symbol": "VEDL.NS", "name": "Vedanta", "sector": "Materials"},
    {"symbol": "ADANIPORTS.NS", "name": "Adani Ports & SEZ", "sector": "Industrials"},
    {"symbol": "BPCL.NS", "name": "Bharat Petroleum", "sector": "Energy"},
    {"symbol": "BRITANNIA.NS", "name": "Britannia Industries", "sector": "Consumer Defensive"},
    {"symbol": "CIPLA.NS", "name": "Cipla Limited", "sector": "Healthcare"},
    {"symbol": "DRREDDY.NS", "name": "Dr. Reddy's Laboratories", "sector": "Healthcare"},
    {"symbol": "EICHERMOT.NS", "name": "Eicher Motors", "sector": "Automotive"},
    {"symbol": "GAIL.NS", "name": "GAIL (India)", "sector": "Energy"},
    {"symbol": "HDFC.NS", "name": "HDFC Ltd", "sector": "Financial Services"},
    {"symbol": "HDFCLIFE.NS", "name": "HDFC Life", "sector": "Financial Services"},
    {"symbol": "IOC.NS", "name": "Indian Oil Corporation", "sector": "Energy"},
    {"symbol": "JSWENERGY.NS", "name": "JSW Energy", "sector": "Utilities"},
    {"symbol": "M&M.NS", "name": "Mahindra & Mahindra", "sector": "Automotive"},
    {"symbol": "NMDC.NS", "name": "NMDC Limited", "sector": "Materials"},
    {"symbol": "ONGC.NS", "name": "Oil & Natural Gas Corp", "sector": "Energy"},
    {"symbol": "PETRONET.NS", "name": "Petronet LNG", "sector": "Energy"},
    {"symbol": "PFC.NS", "name": "Power Finance Corporation", "sector": "Financial Services"},
    {"symbol": "RECLTD.NS", "name": "REC Ltd", "sector": "Financial Services"},
    {"symbol": "SBILIFE.NS", "name": "SBI Life Insurance", "sector": "Financial Services"},
    {"symbol": "SHREECEM.NS", "name": "Shree Cement", "sector": "Materials"},
    {"symbol": "SIEMENS.NS", "name": "Siemens India", "sector": "Industrials"},
    {"symbol": "SRF.NS", "name": "SRF Ltd", "sector": "Materials"},
    {"symbol": "SUNTV.NS", "name": "Sun TV Network", "sector": "Communication Services"},
    {"symbol": "TATACONSUM.NS", "name": "Tata Consumer Products", "sector": "Consumer Defensive"},
    {"symbol": "TATAPOWER.NS", "name": "Tata Power", "sector": "Utilities"},
    {"symbol": "UBL.NS", "name": "United Breweries", "sector": "Consumer Defensive"},
    {"symbol": "UPL.NS", "name": "UPL Ltd", "sector": "Materials"},
    {"symbol": "VEDL.NS", "name": "Vedanta", "sector": "Materials"},
    {"symbol": "ZEEL.NS", "name": "Zee Entertainment Enterprises", "sector": "Communication Services"},
    {"symbol": "BAJAJFINSV.NS", "name": "Bajaj Finserv", "sector": "Financial Services"},
    {"symbol": "BANKBARODA.NS", "name": "Bank of Baroda", "sector": "Financial Services"},
    {"symbol": "CANBK.NS", "name": "Canara Bank", "sector": "Financial Services"},
    {"symbol": "CHOLAFIN.NS", "name": "Cholamandalam Investment & Finance", "sector": "Financial Services"},
    {"symbol": "CUMMINSIND.NS", "name": "Cummins India", "sector": "Industrials"},
    {"symbol": "INDUSTOWER.NS", "name": "Indus Towers", "sector": "Telecommunications"},
    {"symbol": "TATACHEM.NS", "name": "Tata Chemicals", "sector": "Materials"},
    {"symbol": "TVSMOTOR.NS", "name": "TVS Motor Company", "sector": "Automotive"},
    {"symbol": "ADANIGREEN.NS", "name": "Adani Green Energy", "sector": "Utilities"},
    {"symbol": "BPCL.NS", "name": "Bharat Petroleum Corporation", "sector": "Energy"},
    {"symbol": "GODREJCP.NS", "name": "Godrej Consumer Products", "sector": "Consumer Defensive"},
    {"symbol": "BOSCHLTD.NS", "name": "Bosch Limited", "sector": "Automotive"},
    {"symbol": "BIOCON.NS", "name": "Biocon Limited", "sector": "Healthcare"},
    {"symbol": "LTIM.NS", "name": "LTI Mindtree", "sector": "Information Technology"},
    {"symbol": "MUTHOOTFIN.NS", "name": "Muthoot Finance", "sector": "Financial Services"},
    {"symbol": "PIDILITIND.NS", "name": "Pidilite Industries", "sector": "Materials"},
    {"symbol": "TRENT.NS", "name": "Trent Limited", "sector": "Consumer Cyclical"},
    {"symbol": "AUROPHARMA.NS", "name": "Aurobindo Pharma", "sector": "Healthcare"},
    {"symbol": "BHEL.NS", "name": "Bharat Heavy Electricals", "sector": "Industrials"},
    {"symbol": "ADANIPOWER.NS", "name": "Adani Power", "sector": "Utilities"},
    {"symbol": "GODREJPROP.NS", "name": "Godrej Properties", "sector": "Real Estate"},
    {"symbol": "ASHOKLEY.NS", "name": "Ashok Leyland", "sector": "Automotive"},
    {"symbol": "GLENMARK.NS", "name": "Glenmark Pharmaceuticals", "sector": "Healthcare"},
    {"symbol": "TORNTPHARM.NS", "name": "Torrent Pharmaceuticals", "sector": "Healthcare"},
    {"symbol": "PAGEIND.NS", "name": "Page Industries", "sector": "Consumer Cyclical"},
    {"symbol": "BHARATFORG.NS", "name": "Bharat Forge", "sector": "Industrials"},
    {"symbol": "HINDZINC.NS", "name": "Hindustan Zinc", "sector": "Materials"},
    {"symbol": "BANKINDIA.NS", "name": "Bank of India", "sector": "Financial Services"}
]
    



POS_WORDS = {"good","up","positive","gain","bull","beat","surge","rise","optimistic","upgrade","soared","strong","growth","profit","success","win","high","record","breakthrough","innovative","leader","exceed","outperform","rally","boom","thrive","flourish","prosper","expand","increase","soar","jump","climb","advance","improve","recover","rebound"}
NEG_WORDS = {"bad","down","negative","loss","bear","miss","sell","drop","plummet","pessimistic","downgrade","weak","decline","fall","crash","slump","trouble","worry","risk","danger","crisis","fail","bankrupt","cut","reduce","layoff","fire","fraud","scandal","investigation","lawsuit","debt","default","bankruptcy","recession","downturn","volatile","uncertain","uncertainty","fear","panic","selloff","plunge","tumble","slide","dip","downturn","slowdown","contraction","shrink","deteriorate","worsen"}

# ---------------- Sentiment Analysis ----------------
def price_based_sentiment(prices_array) -> float:
    """
    Derive a sentiment score purely from price data so every stock
    gets a unique, accurate reading regardless of news availability.
    Score range: -1.0 (very negative) to +1.0 (very positive)
    """
    arr = np.array(prices_array, dtype="float32")
    if len(arr) < 5:
        return 0.0

    current = float(arr[-1])

    # 1. Short-term momentum: last 5 days vs 20 days ago
    ref5  = float(arr[-6])  if len(arr) > 6  else current
    ref20 = float(arr[-21]) if len(arr) > 21 else float(arr[0])
    mom5  = (current - ref5)  / (ref5  + 1e-9)
    mom20 = (current - ref20) / (ref20 + 1e-9)

    # 2. RSI (14-day): oversold <30 = bullish, overbought >70 = bearish
    delta = pd.Series(arr).diff()
    gain  = delta.clip(lower=0).rolling(14).mean()
    loss  = (-delta.clip(upper=0)).rolling(14).mean()
    rs    = gain / loss.replace(0, np.nan)
    rsi_val = float(100 - (100 / (1 + rs.iloc[-1]))) if not pd.isna(rs.iloc[-1]) else 50.0
    rsi_sentiment = -(rsi_val - 50) / 50  # +1 at RSI=0, -1 at RSI=100

    # 3. MA crossover: MA10 vs MA30
    ma10 = float(pd.Series(arr).rolling(10).mean().iloc[-1]) if len(arr) >= 10 else current
    ma30 = float(pd.Series(arr).rolling(30).mean().iloc[-1]) if len(arr) >= 30 else current
    ma_sent = np.tanh((ma10 - ma30) / (ma30 + 1e-9) * 10)  # -1 to +1

    # 4. Volatility: high volatility → more uncertainty → slight negative bias
    if len(arr) >= 20:
        returns = pd.Series(arr[-20:]).pct_change().dropna()
        vol = float(returns.std())
        vol_sent = -min(vol * 10, 1.0)  # high vol → slightly negative
    else:
        vol_sent = 0.0

    # 5. Recent trend: linear regression slope over last 20 days
    window = arr[-20:] if len(arr) >= 20 else arr
    x = np.arange(len(window), dtype="float32")
    slope = float(np.polyfit(x, window, 1)[0])
    trend_sent = np.tanh(slope / (float(np.mean(window)) + 1e-9) * 50)

    # Weighted composite
    score = (
        mom5       * 0.20 +
        mom20      * 0.15 +
        rsi_sentiment * 0.20 +
        ma_sent    * 0.20 +
        vol_sent   * 0.10 +
        trend_sent * 0.15
    )
    return round(float(np.clip(score, -1.0, 1.0)), 3)


def advanced_sentiment_score(texts, prices_array=None) -> float:
    """
    Combine news-text sentiment with price-data sentiment.
    If no real news is available (only fallback/generic text), weight
    is shifted fully to price-based sentiment so each stock is unique.
    """
    # --- Price-based sentiment (always stock-specific) ---
    price_sent = 0.0
    if prices_array is not None and len(prices_array) > 5:
        price_sent = price_based_sentiment(prices_array)

    # --- Text sentiment ---
    if isinstance(texts, str):
        texts = [texts]

    text_scores = []
    real_news_count = 0  # count headlines that look like real news vs generic fallback

    GENERIC_PHRASES = {
        "market data", "system", "company profile",
        "investors weigh macroeconomic", "being analyzed"
    }

    for text in texts:
        if not text or len(text.strip()) < 5:
            continue

        text_lower = text.lower()
        # Detect if this is a real news headline or our generic fallback
        is_generic = any(p in text_lower for p in GENERIC_PHRASES)
        if not is_generic:
            real_news_count += 1

        blob = TextBlob(text)
        polarity = blob.sentiment.polarity

        words_in_text = set(re.findall(r'\b\w+\b', text_lower))
        pos_count = len(words_in_text & POS_WORDS)
        neg_count = len(words_in_text & NEG_WORDS)
        total_kw = pos_count + neg_count
        keyword_score = (pos_count - neg_count) / total_kw if total_kw > 0 else 0.0

        combined = (polarity * 0.6) + (keyword_score * 0.4)
        text_scores.append(combined)

    text_sent = round(sum(text_scores) / len(text_scores), 3) if text_scores else 0.0

    # Blend: if we have real news headlines, weight text more; otherwise use price data
    if real_news_count >= 3:
        # Good real news available — 50% text, 50% price
        final = (text_sent * 0.50) + (price_sent * 0.50)
    elif real_news_count >= 1:
        # Some real news — 30% text, 70% price
        final = (text_sent * 0.30) + (price_sent * 0.70)
    else:
        # No real news — 100% price-based so each stock is unique
        final = price_sent

    return round(float(np.clip(final, -1.0, 1.0)), 3)


def get_sentiment_label(score: float) -> str:
    if score >= 0.35:
        return "Very Positive"
    elif score >= 0.10:
        return "Positive"
    elif score >= -0.10:
        return "Neutral"
    elif score >= -0.35:
        return "Negative"
    else:
        return "Very Negative"


def fetch_news(ticker):
    """
    Fetch real, ticker-specific news using yfinance (no API key needed).
    Falls back to NewsAPI if key is set, then to a ticker-aware fallback.
    """
    news_items = []

    # --- Primary: yfinance built-in news (free, no key needed) ---
    try:
        yf_news = yf.Ticker(ticker).news or []
        for item in yf_news[:MAX_NEWS]:
            content = item.get("content", {})
            title = content.get("title", "") or item.get("title", "")
            url   = content.get("canonicalUrl", {}).get("url", "") or item.get("link", "")
            src   = content.get("provider", {}).get("displayName", "") or item.get("publisher", "")
            pub   = content.get("pubDate", "") or item.get("providerPublishTime", "")
            # Convert unix timestamp if needed
            if isinstance(pub, (int, float)):
                pub = datetime.fromtimestamp(pub, tz=timezone.utc).isoformat()
            if title:
                news_items.append({"title": title, "url": url, "source": src, "publishedAt": pub})
    except Exception as e:
        log.warning("yfinance news fetch failed for %s: %s", ticker, e)

    if news_items:
        return news_items[:MAX_NEWS]

    # --- Secondary: NewsAPI (if key configured) ---
    if NEWSAPI_KEY:
        try:
            r = requests.get(
                "https://newsapi.org/v2/everything",
                params={
                    "q": ticker,
                    "pageSize": MAX_NEWS,
                    "sortBy": "publishedAt",
                    "language": "en",
                    "apiKey": NEWSAPI_KEY
                },
                timeout=10
            )
            if r.ok:
                articles = r.json().get("articles", [])
                for a in articles:
                    title = a.get("title")
                    if not title or title == "[Removed]":
                        continue
                    news_items.append({
                        "title": title,
                        "url": a.get("url", ""),
                        "source": a.get("source", {}).get("name", ""),
                        "publishedAt": a.get("publishedAt", "")
                    })
                if news_items:
                    return news_items[:MAX_NEWS]
        except Exception as e:
            log.warning("NewsAPI fetch failed: %s", e)

    # --- Fallback: use yfinance stock info summary as sentiment text ---
    try:
        info = yf.Ticker(ticker).info
        summary = info.get("longBusinessSummary", "")
        recent_change = info.get("52WeekChange", 0) or 0
        now = datetime.now(timezone.utc).isoformat()
        fallback = []
        if summary:
            fallback.append({"title": summary[:200], "url": "", "source": "Company Profile", "publishedAt": now})
        # Add price-momentum-based synthetic headline
        if recent_change > 0.1:
            fallback.append({"title": f"{ticker} stock has risen {recent_change*100:.1f}% over the past year, showing strong performance", "url": "", "source": "Market Data", "publishedAt": now})
        elif recent_change < -0.1:
            fallback.append({"title": f"{ticker} stock has fallen {abs(recent_change)*100:.1f}% over the past year, facing headwinds", "url": "", "source": "Market Data", "publishedAt": now})
        else:
            fallback.append({"title": f"{ticker} stock has remained relatively flat over the past year", "url": "", "source": "Market Data", "publishedAt": now})
        if fallback:
            return fallback
    except Exception as e:
        log.warning("Fallback news generation failed for %s: %s", ticker, e)

    # Last resort static sample
    now = datetime.now(timezone.utc).isoformat()
    return [
        {"title": f"Market data for {ticker} is being analyzed", "url": "", "source": "System", "publishedAt": now},
        {"title": "Investors weigh macroeconomic factors affecting stock performance", "url": "", "source": "System", "publishedAt": now},
    ]


# ---------------- Stock Data Functions ----------------
def get_stock_info(ticker):
    """Get additional stock information"""
    try:
        stock = yf.Ticker(ticker)
        info = stock.info
        
        return {
            "sector": info.get("sector", "N/A"),
            "industry": info.get("industry", "N/A"),
            "marketCap": info.get("marketCap", 0),
            "peRatio": info.get("trailingPE", 0),
            "dividendYield": info.get("dividendYield", 0) * 100 if info.get("dividendYield") else 0,
            "52WeekHigh": info.get("fiftyTwoWeekHigh", 0),
            "52WeekLow": info.get("fiftyTwoWeekLow", 0),
            "volume": info.get("volume", 0),
            "avgVolume": info.get("averageVolume", 0)
        }
    except Exception as e:
        log.warning("Failed to get stock info for %s: %s", ticker, e)
        return {}

def get_currency(ticker: str) -> str:
    """Return 'INR' for Indian (.NS / .BO) stocks, 'USD' for everything else."""
    t = ticker.upper()
    if t.endswith(".NS") or t.endswith(".BO"):
        return "INR"
    return "USD"


class PriceModel:
    def __init__(self, path=MODEL_PATH, lookback=LOOKBACK):
        self.path = path
        self.lookback = lookback
        self.model = None
        
        if os.path.exists(path):
            try:
                self.model = load_model(path, compile=False)
                log.info("Loaded model from %s", path)
            except Exception as e: 
                log.warning("Model load failed: %s", e)

    def _prep(self, series):
        arr = np.array(series, dtype="float32")
        X = []
        y = []
        
        for i in range(len(arr) - self.lookback):
            X.append(arr[i:i+self.lookback])
            y.append(arr[i+self.lookback])
            
        return np.array(X).reshape(-1, self.lookback, 1), np.array(y)

    def train(self, series, epochs=20):
        X, y = self._prep(series)
        if len(X) < 20: 
            raise ValueError("Not enough data for training")
            
        m = Sequential([
            LSTM(100, return_sequences=True, input_shape=(self.lookback, 1)),
            Dropout(0.2),
            LSTM(80, return_sequences=False),
            Dropout(0.2),
            Dense(40, activation="relu"),
            Dense(1)
        ])
        
        m.compile(optimizer="adam", loss="mse", metrics=["mae"])
        m.fit(
            X, y, 
            epochs=epochs, 
            batch_size=16,
            callbacks=[EarlyStopping(monitor="loss", patience=5, restore_best_weights=True)],
            verbose=0
        )
        
        m.save(self.path)
        self.model = m
        log.info("Model trained and saved to %s", self.path)

    def forecast(self, series, horizon):
        arr = np.array(series, dtype="float32")
        
        # Normalize prices to 0-1 range so predictions are stock-specific
        price_min = float(arr.min())
        price_max = float(arr.max())
        price_range = price_max - price_min if price_max != price_min else 1.0
        normalized = (arr - price_min) / price_range

        if self.model is None or len(series) < self.lookback:
            self.train(normalized)

        seq = list(normalized[-self.lookback:])
        preds_norm = []

        for _ in range(horizon):
            x = np.array(seq[-self.lookback:]).reshape(1, self.lookback, 1)
            p = float(self.model.predict(x, verbose=0)[0, 0])
            p = max(0.0, min(1.0, p))  # clip to valid normalized range
            preds_norm.append(p)
            seq.append(p)

        # Denormalize back to real prices for this ticker
        preds = [round(float(p * price_range + price_min), 4) for p in preds_norm]
        return preds

# ---------------- Flask Application ----------------
app = Flask(__name__)
model = PriceModel()

@app.route("/api/health")
def health():
    return jsonify({
        "status": "ok", 
        "time": datetime.now(timezone.utc).isoformat(),
        "model_loaded": model.model is not None
    })

@app.route("/api/search")
def search():
    q = (request.args.get("q") or "").strip().lower()
    if not q: 
        return jsonify([])
    
    # Filter by symbol, name, or sector
    results = [
        s for s in POPULAR_STOCKS 
        if q in s["symbol"].lower() or 
           q in s["name"].lower() or 
           (s.get("sector") and q in s["sector"].lower())
    ]
    
    # Add custom ticker if it matches pattern
    if not results and TICKER_RE.match(q): 
        results = [{"symbol": q.upper(), "name": f"{q.upper()} (custom)", "sector": "Unknown"}]
    
    return jsonify(results[:30])

@app.route("/api/data")
def data():
    ticker = request.args.get("ticker", "AAPL")
    n = int(request.args.get("n", 180))  # Default to 180 days
    
    try:
        stock = yf.Ticker(ticker)
        df = stock.history(period=f"{n}d").reset_index()[["Date", "Open", "High", "Low", "Close", "Volume"]]
        df.columns = ["date", "open", "high", "low", "close", "volume"]
        df["date"] = df["date"].astype(str)
        
        # Add stock info
        info = get_stock_info(ticker)
        
        return jsonify({
            "prices": df.to_dict("records"),
            "info": info,
            "currency": get_currency(ticker)
        })
    except Exception as e: 
        log.error("Data fetch error: %s", e)
        return jsonify({"error": str(e)}), 500

@app.route("/api/predict", methods=["POST"])
def predict():
    data = request.get_json(silent=True) or {}
    ticker = data.get("ticker", "AAPL")
    horizon = int(data.get("horizon", DEFAULT_HORIZON))
    
    try:
        prices = yf.Ticker(ticker).history(period="1y")["Close"].values
        predictions = model.forecast(prices, horizon)
        
        return jsonify({
            "ticker": ticker,
            "horizon": horizon,
            "predictions": predictions,
            "last_price": float(prices[-1]) if len(prices) > 0 else 0
        })
    except Exception as e: 
        traceback.print_exc()
        return jsonify({"error": str(e)}), 500

@app.route("/api/news/<ticker>")
def news(ticker):
    try:
        news_items = fetch_news(ticker)
        titles = [item["title"] if isinstance(item, dict) else item for item in news_items]

        # Fetch recent prices to power price-based sentiment
        try:
            prices_arr = yf.Ticker(ticker).history(period="3mo")["Close"].values
        except Exception:
            prices_arr = None

        sentiment = advanced_sentiment_score(titles, prices_array=prices_arr)

        return jsonify({
            "ticker": ticker,
            "news": news_items,
            "sentiment_score": sentiment,
            "sentiment_label": get_sentiment_label(sentiment),
            "fetched_at": datetime.now(timezone.utc).isoformat()
        })
    except Exception as e:
        log.error("News fetch error: %s", e)
        return jsonify({"error": str(e)}), 500

@app.route("/api/recommend", methods=["POST"])
def recommend():
    data = request.get_json(silent=True) or {}
    ticker = data.get("ticker", "AAPL")
    horizon = int(data.get("horizon", DEFAULT_HORIZON))
    
    try:
        # Get price data - use 1y for technical indicators
        stock = yf.Ticker(ticker)
        hist = stock.history(period="1y")
        if len(hist) == 0:
            return jsonify({"error": "No price data available"}), 400

        prices = hist["Close"].values
        current_price = float(prices[-1])

        # Get predictions (normalized per-ticker)
        preds = model.forecast(prices, horizon)
        price_change = (preds[-1] - current_price) / current_price * 100

        # --- Technical Indicators ---
        # 1. RSI (14-day)
        delta = pd.Series(prices).diff()
        gain = delta.clip(lower=0).rolling(14).mean()
        loss = (-delta.clip(upper=0)).rolling(14).mean()
        rs = gain / loss.replace(0, np.nan)
        rsi = float(100 - (100 / (1 + rs.iloc[-1]))) if not np.isnan(rs.iloc[-1]) else 50.0

        # 2. Moving Average Crossover (20 vs 50 day)
        ma20 = float(pd.Series(prices).rolling(20).mean().iloc[-1])
        ma50 = float(pd.Series(prices).rolling(50).mean().iloc[-1]) if len(prices) >= 50 else ma20
        ma_signal = (ma20 - ma50) / ma50 * 100  # positive = bullish

        # 3. Price momentum (last 20 days %)
        momentum = (current_price - float(prices[-21])) / float(prices[-21]) * 100 if len(prices) > 21 else 0.0

        # 4. Volume trend (recent vs average)
        volumes = hist["Volume"].values
        vol_recent = float(volumes[-5:].mean()) if len(volumes) >= 5 else float(volumes.mean())
        vol_avg = float(volumes.mean()) if len(volumes) > 0 else 1.0
        vol_signal = (vol_recent - vol_avg) / vol_avg * 100  # positive = increasing interest

        # 5. News sentiment (ticker-specific, price-data blended)
        news_items = fetch_news(ticker)
        titles = [item["title"] if isinstance(item, dict) else item for item in news_items]
        sentiment = advanced_sentiment_score(titles, prices_array=prices)

        # --- Composite Score ---
        # RSI: oversold (<30) = bullish, overbought (>70) = bearish
        rsi_score = (50 - rsi) / 50 * 10  # range roughly -10 to +10
        # MA crossover score
        ma_score = max(-10, min(10, ma_signal))
        # Momentum score
        mom_score = max(-10, min(10, momentum * 0.5))
        # Volume score (amplifier)
        vol_score = max(-5, min(5, vol_signal * 0.05))
        # Sentiment score
        sent_score = sentiment * 20  # -20 to +20
        # LSTM predicted return score
        lstm_score = max(-15, min(15, price_change * 0.3))

        score = round(rsi_score + ma_score + mom_score + vol_score + sent_score + lstm_score, 2)

        if score > 12:
            action = "Strong Buy"
        elif score > 4:
            action = "Buy"
        elif score > -4:
            action = "Hold"
        elif score > -12:
            action = "Sell"
        else:
            action = "Strong Sell"

        stock_info = get_stock_info(ticker)

        return jsonify({
            "action": action,
            "score": score,
            "predicted_return_pct": round(price_change, 2),
            "sentiment_score": sentiment,
            "sentiment_label": get_sentiment_label(sentiment),
            "predictions": preds,
            "current_price": round(current_price, 2),
            "target_price": round(float(preds[-1]), 2),
            "rsi": round(rsi, 1),
            "ma20": round(ma20, 2),
            "ma50": round(ma50, 2),
            "momentum_pct": round(momentum, 2),
            "news_sample": news_items[:3],
            "stock_info": stock_info,
            "currency": get_currency(ticker)
        })
        
    except Exception as e: 
        traceback.print_exc()
        return jsonify({"error": str(e)}), 500

@app.route("/api/market-overview")
def market_overview():
    """Get market overview data"""
    try:
        # Get major indices
        indices = {
            "SPY": "S&P 500",
            "QQQ": "NASDAQ 100",
            "DIA": "Dow Jones",
            "IWM": "Russell 2000",
            "^NSEI": "Nifty 50",
            "^BSESN": "Sensex"
        }
        
        overview = {}
        for symbol, name in indices.items():
            try:
                data = yf.Ticker(symbol).history(period="5d")
                if len(data) > 0:
                    current = data["Close"].iloc[-1]
                    previous = data["Close"].iloc[-2] if len(data) > 1 else current
                    change = ((current - previous) / previous) * 100
                    
                    overview[symbol] = {
                        "name": name,
                        "price": round(current, 2),
                        "change": round(change, 2),
                        "isPositive": change >= 0,
                        "currency": "INR" if symbol in ("^NSEI", "^BSESN") else "USD"
                    }
            except Exception as e:
                log.warning("Failed to get data for %s: %s", symbol, e)
                continue
                
        return jsonify(overview)
    except Exception as e:
        log.error("Market overview error: %s", e)
        return jsonify({"error": str(e)}), 500

# ---------------- Frontend ----------------
INDEX_HTML = r"""
<!doctype html>
<html lang="en">
<head>
    <meta charset="utf-8">
    <meta name="viewport" content="width=device-width, initial-scale=1">
    <title>Smart Investment Assistant</title>
    <script src="https://cdn.tailwindcss.com"></script>
    <script src="https://cdn.jsdelivr.net/npm/chart.js"></script>
    <script src="https://cdn.jsdelivr.net/npm/luxon@3.3.0/build/global/luxon.min.js"></script>
    <script src="https://cdn.jsdelivr.net/npm/chartjs-adapter-luxon@1.3.1/dist/chartjs-adapter-luxon.min.js"></script>
    <link rel="stylesheet" href="https://cdnjs.cloudflare.com/ajax/libs/font-awesome/6.4.0/css/all.min.css">
</head>
<body class="bg-gray-50 min-h-screen">
    <header class="bg-gradient-to-r from-indigo-600 to-blue-500 text-white shadow-lg">
        <div class="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-4">
            <div class="flex justify-between items-center">
                <div>
                    <h1 class="text-2xl font-bold"><i class="fas fa-chart-line mr-2"></i>Smart Investment Assistant</h1>
                    <p class="text-indigo-100 text-sm">AI-powered stock analysis and predictions</p>
                </div>
                <div id="marketStatus" class="text-sm bg-white/10 backdrop-blur-sm rounded-lg px-3 py-1">
                    <span id="marketStatusText">🟢 Market Live</span>
                </div>
            </div>
        </div>
    </header>

    <main class="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-8 space-y-8">
        <!-- Search Section -->
        <section class="bg-white rounded-xl shadow-md p-6">
            <div class="flex flex-col md:flex-row gap-4 items-start md:items-center">
                <div class="flex-1">
                    <label class="block text-gray-700 font-medium mb-2">Search Stocks:</label>
                    <div class="relative">
                        <input id="searchBox" class="w-full border border-gray-300 rounded-lg px-4 py-3 pl-10 focus:ring-2 focus:ring-indigo-500 focus:border-transparent" placeholder="🔍 Type company name or ticker symbol">
                        <div class="absolute inset-y-0 left-0 pl-3 flex items-center pointer-events-none">
                            <i class="fas fa-search text-gray-400"></i>
                        </div>
                        <div class="absolute inset-y-0 right-0 pr-3 flex items-center space-x-2">
                            <button id="voiceBtn" class="flex items-center text-gray-400 hover:text-indigo-500" aria-pressed="false">
                                <i class="fas fa-microphone"></i>
                            </button>
                            <span id="voiceStatus" class="text-xs text-gray-500 hidden">Idle</span>
                        </div>
                    </div>
                    <ul id="searchResults" class="mt-2 bg-white border border-gray-200 rounded-lg shadow-lg max-h-60 overflow-y-auto hidden"></ul>
                </div>
                <div class="flex gap-2">
                    <button id="refreshBtn" class="px-6 py-2 bg-indigo-600 text-white rounded-lg hover:bg-indigo-700 transition flex items-center gap-2">
                        <i class="fas fa-sync-alt"></i> Refresh
                    </button>
                </div>
            </div>
        </section>

        <!-- Market Overview -->
        <section id="marketOverview" class="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-4">
            <!-- Market data will be loaded here -->
        </section>

        <!-- Currency Converter -->
        <section class="bg-white rounded-xl shadow-md p-6 mb-8">
            <h3 class="font-bold text-lg text-gray-800 mb-4">
                <i class="fas fa-exchange-alt mr-2"></i>USD/INR Converter
            </h3>
            <div class="flex flex-col md:flex-row gap-4">
                <div class="flex-1">
                    <label class="block text-sm font-medium text-gray-600 mb-2">USD Amount</label>
                    <div class="relative">
                        <span class="absolute inset-y-0 left-0 pl-3 flex items-center text-gray-500">$</span>
                        <input type="number" id="usdAmount" class="w-full pl-8 pr-4 py-2 border rounded-lg focus:ring-2 focus:ring-indigo-500" 
                               placeholder="Enter USD amount" value="1" min="0" step="0.01">
                    </div>
                </div>
                <div class="flex-1">
                    <label class="block text-sm font-medium text-gray-600 mb-2">INR Amount</label>
                    <div class="relative">
                        <span class="absolute inset-y-0 left-0 pl-3 flex items-center text-gray-500">₹</span>
                        <input type="number" id="inrAmount" class="w-full pl-8 pr-4 py-2 border rounded-lg focus:ring-2 focus:ring-indigo-500" 
                               placeholder="Enter INR amount" value="" min="0" step="0.01">
                    </div>
                </div>
                <div class="flex-none self-end">
                    <button id="refreshRate" class="px-4 py-2 bg-indigo-600 text-white rounded-lg hover:bg-indigo-700 transition">
                        <i class="fas fa-sync-alt mr-1"></i>Update Rate
                    </button>
                </div>
            </div>
            <p class="text-sm text-gray-500 mt-3" id="lastRateUpdate"></p>
        </section>

        <!-- Main Content Grid -->
        <div class="grid grid-cols-1 lg:grid-cols-3 gap-8">
            <!-- Left Column - Chart -->
            <div class="lg:col-span-2">
                <div class="bg-white rounded-xl shadow-md p-6">
                    <div class="flex justify-between items-center mb-4">
                        <h2 class="text-xl font-semibold text-gray-800" id="companyName">Price & Forecast</h2>
                        <div class="text-sm text-gray-500" id="stockInfo"></div>
                    </div>
                    <!-- Chart time filters -->
                    <div class="flex gap-1 mb-4 flex-wrap">
                        <button onclick="setChartFilter(7)"   id="filter-7"   class="chart-filter-btn px-3 py-1 text-xs rounded-full border border-gray-300 text-gray-600 hover:bg-indigo-600 hover:text-white transition">1W</button>
                        <button onclick="setChartFilter(30)"  id="filter-30"  class="chart-filter-btn px-3 py-1 text-xs rounded-full border border-gray-300 text-gray-600 hover:bg-indigo-600 hover:text-white transition">1M</button>
                        <button onclick="setChartFilter(90)"  id="filter-90"  class="chart-filter-btn px-3 py-1 text-xs rounded-full border border-gray-300 text-gray-600 hover:bg-indigo-600 hover:text-white transition">3M</button>
                        <button onclick="setChartFilter(180)" id="filter-180" class="chart-filter-btn px-3 py-1 text-xs rounded-full border border-gray-300 text-gray-600 hover:bg-indigo-600 hover:text-white transition">6M</button>
                        <button onclick="setChartFilter(365)" id="filter-365" class="chart-filter-btn px-3 py-1 text-xs rounded-full border bg-indigo-600 text-white transition">1Y</button>
                        <button onclick="setChartFilter(0)"   id="filter-0"   class="chart-filter-btn px-3 py-1 text-xs rounded-full border border-gray-300 text-gray-600 hover:bg-indigo-600 hover:text-white transition">All</button>
                    </div>
                    <div class="h-80">
                        <canvas id="priceChart"></canvas>
                    </div>
                    <div class="mt-4 flex justify-between items-center text-sm text-gray-500">
                        <span id="lastUpdated">Updated: -</span>
                        <div id="priceChange" class="font-medium"></div>
                    </div>
                </div>
            </div>

            <!-- Right Column - Recommendation & News -->
            <div class="space-y-6">
                <!-- Recommendation Card -->
                <div class="bg-gradient-to-br from-indigo-50 to-blue-100 rounded-xl shadow-md p-6">
                    <h3 class="font-bold text-lg text-gray-800 mb-4"><i class="fas fa-star mr-2"></i>Investment Recommendation</h3>
                    <div id="action" class="text-4xl font-extrabold text-center my-4">-</div>
                    <div id="recommendationDetails" class="space-y-3 text-sm">
                        <div class="flex justify-between">
                            <span class="text-gray-600">Current Price:</span>
                            <span id="currentPrice" class="font-medium text-indigo-700">-</span>
                        </div>
                        <div class="flex justify-between">
                            <span class="text-gray-600">Predicted Return:</span>
                            <span id="predReturn" class="font-medium">-</span>
                        </div>
                        <div class="flex justify-between">
                            <span class="text-gray-600">Sentiment:</span>
                            <span id="sentiment" class="font-medium">-</span>
                        </div>
                        <div class="flex justify-between">
                            <span class="text-gray-600">Confidence Score:</span>
                            <span id="confidence" class="font-medium">-</span>
                        </div>
                        <div class="flex justify-between">
                            <span class="text-gray-600">Target Price:</span>
                            <span id="targetPrice" class="font-medium">-</span>
                        </div>
                        <hr class="border-indigo-200 my-1">
                        <div class="flex justify-between">
                            <span class="text-gray-600">RSI (14):</span>
                            <span id="rsiValue" class="font-medium">-</span>
                        </div>
                        <div class="flex justify-between">
                            <span class="text-gray-600">MA20 / MA50:</span>
                            <span id="maValue" class="font-medium">-</span>
                        </div>
                        <div class="flex justify-between">
                            <span class="text-gray-600">Momentum (20d):</span>
                            <span id="momentumValue" class="font-medium">-</span>
                        </div>
                    </div>
                </div>

                <!-- News Card -->
                <div class="bg-white rounded-xl shadow-md p-6">
                    <div class="flex justify-between items-center mb-4">
                        <h3 class="font-bold text-lg text-gray-800"><i class="fas fa-newspaper mr-2"></i>Latest News</h3>
                        <span id="newsUpdatedAt" class="text-xs text-gray-400"></span>
                    </div>
                    <div id="newsContainer" class="space-y-3 max-h-80 overflow-y-auto">
                        <div class="text-center text-gray-500 py-4">Search for a stock to see related news</div>
                    </div>
                </div>
            </div>
        </div>

        <!-- Stock Details -->
        <section id="stockDetails" class="bg-white rounded-xl shadow-md p-6 hidden">
            <h3 class="font-bold text-lg text-gray-800 mb-4"><i class="fas fa-info-circle mr-2"></i>Stock Details</h3>
            <div class="grid grid-cols-2 md:grid-cols-4 gap-4" id="detailsGrid">
                <!-- Details will be populated here -->
            </div>
        </section>
    </main>

    <footer class="bg-gray-800 text-white text-center py-6 mt-12">
        <div class="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
            <p>© 2025 Smart Investment Assistant. Powered by AI and Machine Learning.</p>
            <p class="text-gray-400 text-sm mt-2">This is for educational purposes only. Not financial advice.</p>
        </div>
    </footer>

    <script>
        let selectedTicker = "AAPL";
        let priceChart = null;
        let currentData = null;
        let currentRecommendation = null;
        let activeChartDays = 365; // default 1Y
        let ctx;

        function setChartFilter(days) {
            activeChartDays = days;
            // Update button styles
            document.querySelectorAll('.chart-filter-btn').forEach(btn => {
                btn.className = 'chart-filter-btn px-3 py-1 text-xs rounded-full border border-gray-300 text-gray-600 hover:bg-indigo-600 hover:text-white transition';
            });
            const activeBtn = document.getElementById(`filter-${days}`);
            if (activeBtn) activeBtn.className = 'chart-filter-btn px-3 py-1 text-xs rounded-full border bg-indigo-600 text-white transition';
            // Re-render chart with filter
            if (currentData && currentRecommendation) {
                updateChart(currentData, currentRecommendation);
            }
        }

    document.addEventListener('DOMContentLoaded', function() {
    ctx = document.getElementById('priceChart').getContext('2d');
    loadMarketOverview();
    refreshData();
    setupEventListeners();
});


        function setupEventListeners() {
            // Search functionality
            document.getElementById('searchBox').addEventListener('input', debounce(handleSearch, 300));
            document.getElementById('refreshBtn').addEventListener('click', refreshData);
            document.getElementById('voiceBtn').addEventListener('click', startVoiceRecognition);
            
            // Currency converter events
            document.getElementById('usdAmount').addEventListener('input', handleUSDInput);
            document.getElementById('inrAmount').addEventListener('input', handleINRInput);
            document.getElementById('refreshRate').addEventListener('click', updateExchangeRate);
            
            // Initial exchange rate load
            updateExchangeRate();
        }

        // Global exchange rate cache
        let currentRate = 83.25; // fallback rate if API fails
        let currentCurrency = 'USD'; // tracks currency of currently selected stock

        /**
         * Convert a price to INR display string.
         * If currency is already INR (Indian .NS/.BO stocks), do NOT multiply.
         * If currency is USD (US stocks), multiply by currentRate.
         */
        function toINR(price, currency) {
            if (price === null || price === undefined || isNaN(price)) return 'N/A';
            const inrVal = (currency === 'INR') ? price : price * currentRate;
            return '₹' + inrVal.toLocaleString('en-IN', { maximumFractionDigits: 2 });
        }

        function toINRRaw(price, currency) {
            if (price === null || price === undefined || isNaN(price)) return null;
            return (currency === 'INR') ? price : price * currentRate;
        }
        
        async function updateExchangeRate() {
            try {
                const response = await fetch('https://api.exchangerate-api.com/v4/latest/USD');
                const data = await response.json();
                currentRate = data.rates.INR;
                document.getElementById('lastRateUpdate').textContent = 
                    `Current rate: 1 USD = ₹${currentRate.toFixed(2)} (Updated: ${new Date().toLocaleTimeString()})`;
                
                // Update the INR amount based on current USD value
                handleUSDInput({ target: document.getElementById('usdAmount') });
            } catch (error) {
                console.error('Exchange rate update failed:', error);
                document.getElementById('lastRateUpdate').textContent = 
                    `Using fallback rate: 1 USD = ₹${currentRate.toFixed(2)} (API unavailable)`;
            }
        }
        
        function handleUSDInput(e) {
            const usd = parseFloat(e.target.value) || 0;
            const inr = (usd * currentRate).toFixed(2);
            document.getElementById('inrAmount').value = inr;
        }
        
        function handleINRInput(e) {
            const inr = parseFloat(e.target.value) || 0;
            const usd = (inr / currentRate).toFixed(2);
            document.getElementById('usdAmount').value = usd;
        }

        function debounce(func, wait) {
            let timeout;
            return function executedFunction(...args) {
                const later = () => {
                    clearTimeout(timeout);
                    func(...args);
                };
                clearTimeout(timeout);
                timeout = setTimeout(later, wait);
            };
        }

        function speak(text) {
            if ('speechSynthesis' in window) {
                const utterance = new SpeechSynthesisUtterance(text);
                utterance.lang = 'en-US';
                utterance.volume = 1;
                utterance.rate = 1;
                utterance.pitch = 1;
                window.speechSynthesis.speak(utterance);
            } else {
                console.warn('Text-to-speech not supported in this browser.');
            }
        }

    // Global recognition instance to avoid multiple simultaneous listeners
    let recognitionInstance = null;
    let isListening = false;
    let manualStop = false; // true when user explicitly stops via button
    let restartAttempts = 0;
    const MAX_RESTARTS = 5;

        async function startVoiceRecognition() {
            const SpeechRecognition = window.SpeechRecognition || window.webkitSpeechRecognition;
            const voiceBtn = document.getElementById('voiceBtn');
            const voiceStatus = document.getElementById('voiceStatus');

            if (!SpeechRecognition) {
                speak('Speech recognition is not supported in this browser. Please use Chrome, Edge, or a compatible browser.');
                alert('Speech recognition not supported in this browser. Try Chrome or Edge.');
                return;
            }

            // If already listening, treat click as stop toggle
            if (isListening) {
                manualStop = true;
                try { recognitionInstance && recognitionInstance.stop(); } catch(e){ console.warn('Stop error', e); }
                return;
            }

            // Warm up mic and permissions
            if (navigator.mediaDevices && navigator.mediaDevices.getUserMedia) {
                try {
                    const stream = await navigator.mediaDevices.getUserMedia({ audio: true });
                    stream.getTracks().forEach(t => t.stop());
                } catch (e) {
                    console.warn('getUserMedia failed or denied:', e);
                }
            }

            // Show a small countdown so user can start speaking after pressing the button
            voiceBtn.classList.add('text-red-500');
            voiceBtn.setAttribute('aria-pressed', 'true');
            let countdown = 3;
            if (voiceStatus) { voiceStatus.classList.remove('hidden'); voiceStatus.textContent = `Start speaking in ${countdown}...`; }
            while (countdown > 0) {
                await new Promise(r => setTimeout(r, 700));
                countdown -= 1;
                if (voiceStatus) voiceStatus.textContent = countdown > 0 ? `Start speaking in ${countdown}...` : 'Listening...';
            }

            manualStop = false;
            restartAttempts = 0;

            // Create and start a new recognition instance
            const createRecognition = () => {
                const r = new SpeechRecognition();
                r.lang = 'en-US';
                r.interimResults = true;
                r.continuous = false; // single command mode - stop after processing
                r.maxAlternatives = 1;

                r.onstart = () => {
                    isListening = true;
                    restartAttempts = 0;
                    if (voiceStatus) { voiceStatus.textContent = 'Listening...'; }
                    voiceBtn.disabled = false;
                };

                r.onspeechstart = () => {
                    if (voiceStatus) voiceStatus.textContent = 'Receiving speech...';
                };

                r.onresult = (event) => {
                    try {
                        let interim = '';
                        let finalTranscript = '';
                        for (let i = event.resultIndex; i < event.results.length; i++) {
                            const res = event.results[i];
                            const text = res[0].transcript;
                            if (res.isFinal) {
                                finalTranscript += text;
                            } else {
                                interim += text;
                            }
                        }

                        if (interim && voiceStatus) {
                            voiceStatus.textContent = interim.trim();
                        }

                        if (finalTranscript) {
                            const transcript = finalTranscript.trim().toLowerCase();
                            if (voiceStatus) { voiceStatus.textContent = 'Processing command...'; }
                            
                            // Process command and stop listening
                            processVoiceCommand(transcript);
                            
                            // Auto-stop after processing
                            manualStop = true;
                            try { r.stop(); } catch(e){}
                            cleanupRecognition();
                            if (voiceStatus) { voiceStatus.textContent = 'Done'; }
                        }
                    } catch (e) {
                        console.error('Error processing recognition result:', e);
                    }
                };

                r.onerror = (event) => {
                    console.error('Speech recognition error:', event.error);
                    if (event.error === 'not-allowed' || event.error === 'security') {
                        speak('Microphone permission denied. Please allow microphone access and try again.');
                        alert('Microphone permission denied. Please allow access and reload the page.');
                        manualStop = true;
                        try { r.stop(); } catch(e){}
                        cleanupRecognition();
                        return;
                    }

                    // For no-speech, don't abort immediately; let continuous mode keep listening
                    if (event.error === 'no-speech') {
                        if (voiceStatus) { voiceStatus.textContent = 'No speech detected — still listening...'; }
                        // Let the recognizer continue; if it ends repeatedly, onend will restart up to limit
                        return;
                    }

                    // Other errors: stop and cleanup
                    speak('Speech recognition error: ' + event.error);
                    try { r.stop(); } catch(e){}
                    cleanupRecognition();
                };

                r.onend = () => {
                    isListening = false;
                    // If user didn't manually stop, attempt to restart a few times
                    if (!manualStop && restartAttempts < MAX_RESTARTS) {
                        restartAttempts += 1;
                        console.info('Recognition ended unexpectedly. Restart attempt', restartAttempts);
                        setTimeout(() => {
                            if (!manualStop) {
                                recognitionInstance = createRecognition();
                                try { recognitionInstance.start(); } catch (e) { console.warn('Restart start failed', e); }
                            }
                        }, 300);
                        return;
                    }

                    // Final cleanup
                    cleanupRecognition();
                };

                return r;
            };

            const cleanupRecognition = () => {
                try {
                    if (recognitionInstance) {
                        recognitionInstance.onstart = null;
                        recognitionInstance.onresult = null;
                        recognitionInstance.onerror = null;
                        recognitionInstance.onend = null;
                    }
                } catch (e) { }
                recognitionInstance = null;
                isListening = false;
                voiceBtn.classList.remove('text-red-500');
                voiceBtn.removeAttribute('aria-pressed');
                voiceBtn.disabled = false;
                if (voiceStatus) { voiceStatus.textContent = 'Idle'; }
            };

            try {
                recognitionInstance = createRecognition();
                recognitionInstance.start();
                speak('Listening for your command.');
            } catch (e) {
                console.warn('Failed to start recognition:', e);
                cleanupRecognition();
            }
        }



        async function processVoiceCommand(transcript) {
            console.log('Voice command:', transcript);

            // Refresh or update data
            if (transcript.includes('refresh') || transcript.includes('update')) {
                speak('Refreshing data.');
                refreshData();
                return;
            }

            // Market overview
            if (transcript.includes('market overview') || transcript.includes('market status')) {
                speak('Loading market overview.');
                loadMarketOverview();
                return;
            }

            // Search for stock (handles "search Apple", "show Apple", "predict Apple", etc.)
            const searchTerms = ['search', 'show', 'predict', 'find', 'lookup', 'stock'];
            if (searchTerms.some(term => transcript.includes(term))) {
                let query = transcript;
                searchTerms.forEach(term => {
                    query = query.replace(term, '').trim();
                });
                if (query) {
                    speak(`Searching for ${query}.`);
                    document.getElementById('searchBox').value = query;
                    await handleSearch({target: {value: query}});
                    // If search returns results, select the first one
                    const results = document.getElementById('searchResults').querySelectorAll('li');
                    if (results.length > 0) {
                        selectedTicker = results[0].dataset.symbol;
                        document.getElementById('companyName').textContent = results[0].dataset.name;
                        document.getElementById('searchBox').value = '';
                        document.getElementById('searchResults').classList.add('hidden');
                        speak(`Displaying data for ${results[0].dataset.name}.`);
                        resetRecommendationCard();
                        refreshData();
                    } else {
                        speak('No stocks found for ' + query);
                    }
                    return;
                }
            }

            // Default: treat as stock search
            if (transcript) {
                speak(`Searching for ${transcript}.`);
                document.getElementById('searchBox').value = transcript;
                await handleSearch({target: {value: transcript}});
                const results = document.getElementById('searchResults').querySelectorAll('li');
                if (results.length > 0) {
                    selectedTicker = results[0].dataset.symbol;
                    document.getElementById('companyName').textContent = results[0].dataset.name;
                    document.getElementById('searchBox').value = '';
                    document.getElementById('searchResults').classList.add('hidden');
                    speak(`Displaying data for ${results[0].dataset.name}.`);
                    resetRecommendationCard();
                    refreshData();
                } else {
                    speak('No stocks found for ' + transcript);
                }
            } else {
                speak('No command recognized. Please try again.');
            }
        }

        async function handleSearch(e) {
            const query = e.target.value.trim();
            const resultsContainer = document.getElementById('searchResults');
            
            if (!query) {
                resultsContainer.classList.add('hidden');
                return;
            }
            
            try {
                const response = await fetch(`/api/search?q=${encodeURIComponent(query)}`);
                const results = await response.json();
                
                if (results.length > 0) {
                    resultsContainer.innerHTML = results.map(stock => `
                        <li class="px-4 py-3 hover:bg-indigo-50 cursor-pointer border-b border-gray-100 last:border-b-0" 
                            data-symbol="${stock.symbol}" data-name="${stock.name}">
                            <div class="font-medium">${stock.name}</div>
                            <div class="text-sm text-gray-500">${stock.symbol} ${stock.sector ? '• ' + stock.sector : ''}</div>
                        </li>
                    `).join('');
                    
                    resultsContainer.classList.remove('hidden');
                    
                    // Add click event to results
                    resultsContainer.querySelectorAll('li').forEach(item => {
                        item.addEventListener('click', () => {
                            selectedTicker = item.dataset.symbol;
                            document.getElementById('companyName').textContent = item.dataset.name;
                            document.getElementById('searchBox').value = '';
                            resultsContainer.classList.add('hidden');
                            speak(`Displaying data for ${item.dataset.name}.`);
                            // Reset recommendation card immediately so stale data from previous stock is cleared
                            resetRecommendationCard();
                            refreshData();
                        });
                    });
                } else {
                    resultsContainer.innerHTML = '<li class="px-4 py-3 text-gray-500">No results found</li>';
                    resultsContainer.classList.remove('hidden');
                }
            } catch (error) {
                console.error('Search error:', error);
                speak('Error searching for stocks. Please try again.');
            }
        }

        async function loadMarketOverview() {
            try {
                const response = await fetch('/api/market-overview');
                const data = await response.json();
                
                const container = document.getElementById('marketOverview');
                container.innerHTML = '';
                
                for (const [symbol, info] of Object.entries(data)) {
                    // Skip entries with invalid/missing price
                    if (!info.price || isNaN(info.price) || !isFinite(info.price)) continue;
                    if (info.change === undefined || isNaN(info.change)) continue;
                    
                    const changeClass = info.change >= 0 ? 'text-green-600' : 'text-red-600';
                    const changeIcon = info.change >= 0 ? 'fa-arrow-up' : 'fa-arrow-down';
                    const mktCurrency = info.currency || 'USD';
                    const displayPrice = toINR(info.price, mktCurrency);
                    
                    container.innerHTML += `
                        <div class="bg-white rounded-lg shadow p-4 flex items-center justify-between">
                            <div>
                                <h4 class="font-medium text-gray-900">${info.name}</h4>
                                <p class="text-2xl font-bold">${displayPrice}</p>
                            </div>
                            <div class="text-right">
                                <p class="${changeClass} font-medium">
                                    <i class="fas ${changeIcon}"></i> ${Math.abs(info.change)}%
                                </p>
                                <p class="text-sm text-gray-500">${symbol}</p>
                            </div>
                        </div>
                    `;
                }
            } catch (error) {
                console.error('Market overview error:', error);
                speak('Error loading market overview.');
            }
        }

        function resetRecommendationCard() {
            document.getElementById('action').textContent = '-';
            document.getElementById('action').className = 'text-4xl font-extrabold text-center my-4 text-gray-400';
            document.getElementById('currentPrice').textContent = '...';
            document.getElementById('predReturn').textContent = '...';
            document.getElementById('predReturn').className = 'font-medium text-gray-400';
            document.getElementById('sentiment').textContent = '...';
            document.getElementById('sentiment').className = 'font-medium text-gray-400';
            document.getElementById('confidence').textContent = '...';
            document.getElementById('targetPrice').textContent = '...';
            if (document.getElementById('rsiValue')) document.getElementById('rsiValue').textContent = '...';
            if (document.getElementById('maValue')) document.getElementById('maValue').textContent = '...';
            if (document.getElementById('momentumValue')) document.getElementById('momentumValue').textContent = '...';
            // Reset news card
            document.getElementById('newsContainer').innerHTML =
                '<div class="text-center text-gray-400 py-6"><i class="fas fa-spinner fa-spin mr-2"></i>Fetching latest news...</div>';
            if (document.getElementById('newsUpdatedAt')) {
                document.getElementById('newsUpdatedAt').textContent = '';
            }
        }

        async function refreshData() {
            const refreshBtn = document.getElementById('refreshBtn');
            const horizon = 14; // default 14-day prediction horizon
            
            refreshBtn.disabled = true;
            refreshBtn.innerHTML = '<i class="fas fa-spinner fa-spin"></i> Loading...';
            
            // Show loading state in news card
            document.getElementById('newsContainer').innerHTML = 
                '<div class="text-center text-gray-400 py-6"><i class="fas fa-spinner fa-spin mr-2"></i>Fetching latest news...</div>';
            if (document.getElementById('newsUpdatedAt')) {
                document.getElementById('newsUpdatedAt').textContent = '';
            }
            
            // Show loading state in recommendation card
            document.getElementById('action').textContent = '...';
            document.getElementById('action').className = 'text-4xl font-extrabold text-center my-4 text-gray-400';
            
            try {
                const [priceData, recommendation, news] = await Promise.all([
                    fetch(`/api/data?ticker=${selectedTicker}&n=180`).then(r => r.json()),
                    fetch('/api/recommend', {
                        method: 'POST',
                        headers: {'Content-Type': 'application/json'},
                        body: JSON.stringify({ticker: selectedTicker, horizon: horizon})
                    }).then(r => r.json()),
                    fetch(`/api/news/${selectedTicker}`).then(r => r.json())
                ]);
                
                currentData = priceData;
                currentRecommendation = recommendation;
                updateChart(priceData, recommendation);
                updateRecommendation(recommendation);
                updateNews(news);
                updateStockDetails(priceData.info);
                
                speak(`Data refreshed for ${document.getElementById('companyName').textContent}.`);
                
            } catch (error) {
                console.error('Error refreshing data:', error);
                speak('Failed to load data. Please try again.');
                alert('Failed to load data. Please try again.');
            } finally {
                refreshBtn.disabled = false;
                refreshBtn.innerHTML = '<i class="fas fa-sync-alt"></i> Refresh';
                document.getElementById('lastUpdated').textContent = `Updated: ${new Date().toLocaleTimeString()}`;
            }
        }

        function updateChart(priceData, recommendation) {
            currentRecommendation = recommendation; // store for filter re-renders
            // Set currency from priceData so chart renders correctly before recommendation arrives
            if (priceData.currency) currentCurrency = priceData.currency;
            let prices = priceData.prices || [];
            const preds = recommendation.predictions || [];
            
            if (prices.length === 0) return;

            // Apply time filter
            if (activeChartDays > 0 && prices.length > activeChartDays) {
                prices = prices.slice(prices.length - activeChartDays);
            }
            
            const labels = prices.map(p => p.date);
            const priceValues = prices.map(p => p.close);
            
            // Generate prediction dates
            const lastDate = new Date(prices[prices.length - 1].date);
            const predDates = [];
            for (let i = 1; i <= preds.length; i++) {
                const nextDate = new Date(lastDate);
                nextDate.setDate(nextDate.getDate() + i);
                predDates.push(nextDate.toISOString().split('T')[0]);
            }
            
            const allLabels = [...labels, ...predDates];
            const historicalData = [...priceValues, ...Array(preds.length).fill(null)];
            const predictionData = [...Array(priceValues.length).fill(null), ...preds];
            
            if (priceChart) {
                priceChart.destroy();
            }
            
            priceChart = new Chart(ctx, {
                type: 'line',
                data: {
                    labels: allLabels,
                    datasets: [
                        {
                            label: 'Historical Price',
                            data: historicalData,
                            borderColor: 'rgb(79, 70, 229)',
                            backgroundColor: 'rgba(79, 70, 229, 0.1)',
                            fill: true,
                            tension: 0.3,
                            pointRadius: 0,
                            borderWidth: 2
                        },
                        {
                            label: 'Prediction',
                            data: predictionData,
                            borderColor: 'rgb(16, 185, 129)',
                            borderDash: [5, 5],
                            tension: 0.3,
                            pointRadius: 3,
                            borderWidth: 2
                        }
                    ]
                },
                options: {
                    responsive: true,
                    maintainAspectRatio: false,
                    plugins: {
                        legend: {
                            position: 'bottom',
                            labels: {
                                usePointStyle: true,
                                padding: 20
                            }
                        },
                        tooltip: {
                            mode: 'index',
                            intersect: false,
                            callbacks: {
                                label: function(context) {
                                    if (context.raw === null) return null;
                                    return `${context.dataset.label}: ${toINR(context.raw, currentCurrency)}`;
                                }
                            }
                        }
                    },
                    scales: {
                        x: {
                            grid: {
                                display: false
                            },
                            ticks: {
                                maxRotation: 0,
                                autoSkip: true,
                                maxTicksLimit: 8
                            }
                        },
                        y: {
                            grid: {
                                color: 'rgba(0, 0, 0, 0.05)'
                            },
                            ticks: {
                                callback: function(value) {
                                    const v = toINRRaw(value, currentCurrency);
                                    if (v === null) return '';
                                    return '₹' + v.toLocaleString('en-IN', {maximumFractionDigits: 0});
                                }
                            }
                        }
                    }
                }
            });
            
            // Update price change
            const firstPrice = priceValues[0];
            const lastPrice = priceValues[priceValues.length - 1];
            const change = ((lastPrice - firstPrice) / firstPrice) * 100;
            const changeElem = document.getElementById('priceChange');
            
            if (change >= 0) {
                changeElem.innerHTML = `<span class="text-green-600"><i class="fas fa-arrow-up"></i> ${change.toFixed(2)}%</span>`;
            } else {
                changeElem.innerHTML = `<span class="text-red-600"><i class="fas fa-arrow-down"></i> ${change.toFixed(2)}%</span>`;
            }
        }

        function updateRecommendation(data) {
            if (data.error) {
                document.getElementById('action').textContent = 'Error';
                document.getElementById('recommendationDetails').innerHTML = `<div class="text-red-500">${data.error}</div>`;
                speak('Error in recommendation: ' + data.error);
                return;
            }
            
            // Update action with color coding
            const action = document.getElementById('action');
            action.textContent = data.action;
            
            switch(data.action) {
                case 'Strong Buy':
                    action.className = 'text-4xl font-extrabold text-center my-4 text-green-700';
                    break;
                case 'Buy':
                    action.className = 'text-4xl font-extrabold text-center my-4 text-green-500';
                    break;
                case 'Hold':
                    action.className = 'text-4xl font-extrabold text-center my-4 text-yellow-500';
                    break;
                case 'Sell':
                    action.className = 'text-4xl font-extrabold text-center my-4 text-orange-500';
                    break;
                case 'Strong Sell':
                    action.className = 'text-4xl font-extrabold text-center my-4 text-red-700';
                    break;
                default:
                    action.className = 'text-4xl font-extrabold text-center my-4 text-gray-500';
            }
            
            // Update details
            document.getElementById('predReturn').textContent = 
                `${data.predicted_return_pct >= 0 ? '+' : ''}${data.predicted_return_pct}%`;
            document.getElementById('predReturn').className = 
                `font-medium ${data.predicted_return_pct >= 0 ? 'text-green-600' : 'text-red-600'}`;
                
            document.getElementById('sentiment').textContent = 
                `${data.sentiment_label} (${data.sentiment_score})`;
            document.getElementById('sentiment').className = 
                `font-medium ${data.sentiment_score >= 0 ? 'text-green-600' : 'text-red-600'}`;
                
            document.getElementById('confidence').textContent = `${data.score}`;
            
            // Store currency for chart and other displays
            currentCurrency = data.currency || 'USD';

            // Show current price in correct currency
            if (data.current_price) {
                document.getElementById('currentPrice').textContent = toINR(data.current_price, currentCurrency);
            }
            
            // Target price in correct currency
            document.getElementById('targetPrice').textContent = toINR(data.target_price, currentCurrency);

            // RSI
            if (document.getElementById('rsiValue')) {
                const rsi = data.rsi || '-';
                let rsiColor = 'text-gray-700';
                if (rsi !== '-') {
                    if (rsi < 30) rsiColor = 'text-green-600';
                    else if (rsi > 70) rsiColor = 'text-red-600';
                    else rsiColor = 'text-yellow-600';
                }
                document.getElementById('rsiValue').textContent = rsi !== '-' ? `${rsi} ${rsi < 30 ? '(Oversold)' : rsi > 70 ? '(Overbought)' : '(Neutral)'}` : '-';
                document.getElementById('rsiValue').className = `font-medium ${rsiColor}`;
            }

            // MA20 / MA50 — use correct currency
            if (document.getElementById('maValue') && data.ma20 && data.ma50) {
                const ma20inr = toINR(data.ma20, currentCurrency);
                const ma50inr = toINR(data.ma50, currentCurrency);
                const maBull = data.ma20 > data.ma50;
                document.getElementById('maValue').textContent = `${ma20inr} / ${ma50inr}`;
                document.getElementById('maValue').className = `font-medium text-xs ${maBull ? 'text-green-600' : 'text-red-600'}`;
            }

            // Momentum
            if (document.getElementById('momentumValue') && data.momentum_pct !== undefined) {
                const mom = data.momentum_pct;
                document.getElementById('momentumValue').textContent = `${mom >= 0 ? '+' : ''}${mom}%`;
                document.getElementById('momentumValue').className = `font-medium ${mom >= 0 ? 'text-green-600' : 'text-red-600'}`;
            }
        }

        function updateNews(data) {
            const container = document.getElementById('newsContainer');
            const updatedLabel = document.getElementById('newsUpdatedAt');
            
            if (data.error || !data.news || data.news.length === 0) {
                container.innerHTML = '<div class="text-center text-gray-500 py-4">No news available</div>';
                speak('No news available for this stock.');
                return;
            }

            // Show fetch timestamp
            if (data.fetched_at) {
                const fetchTime = new Date(data.fetched_at).toLocaleTimeString();
                updatedLabel.textContent = `🔄 Updated ${fetchTime}`;
            } else {
                updatedLabel.textContent = `🔄 Updated ${new Date().toLocaleTimeString()}`;
            }

            container.innerHTML = data.news.slice(0, 5).map(item => {
                // Support both dict (with title/source/url/publishedAt) and plain string
                const isDict = typeof item === 'object' && item !== null;
                const title = isDict ? item.title : item;
                const source = isDict && item.source ? item.source : '';
                const url = isDict && item.url ? item.url : '';
                const pubDate = isDict && item.publishedAt
                    ? new Date(item.publishedAt).toLocaleDateString('en-IN', {day:'numeric', month:'short', year:'numeric'})
                    : '';

                return `
                <div class="p-3 bg-gray-50 rounded-lg border border-gray-100 hover:border-indigo-200 transition">
                    ${url
                        ? `<a href="${url}" target="_blank" rel="noopener noreferrer" class="text-sm text-gray-800 font-medium hover:text-indigo-600 leading-snug block">${title}</a>`
                        : `<p class="text-sm text-gray-800 font-medium leading-snug">${title}</p>`
                    }
                    <div class="flex justify-between items-center mt-1">
                        ${source ? `<span class="text-xs text-indigo-500 font-medium">${source}</span>` : '<span></span>'}
                        ${pubDate ? `<span class="text-xs text-gray-400">${pubDate}</span>` : ''}
                    </div>
                </div>`;
            }).join('');
        }

        function updateStockDetails(info) {
            const detailsSection = document.getElementById('stockDetails');
            const detailsGrid = document.getElementById('detailsGrid');
            
            if (!info || Object.keys(info).length === 0) {
                detailsSection.classList.add('hidden');
                speak('No stock details available.');
                return;
            }
            
            detailsSection.classList.remove('hidden');
            
            detailsGrid.innerHTML = `
                <div class="bg-gray-50 p-4 rounded-lg">
                    <div class="text-gray-500 text-sm">Sector</div>
                    <div class="font-medium">${info.sector || 'N/A'}</div>
                </div>
                <div class="bg-gray-50 p-4 rounded-lg">
                    <div class="text-gray-500 text-sm">Market Cap</div>
                    <div class="font-medium">${formatMarketCap(info.marketCap)}</div>
                </div>
                <div class="bg-gray-50 p-4 rounded-lg">
                    <div class="text-gray-500 text-sm">P/E Ratio</div>
                    <div class="font-medium">${info.peRatio ? info.peRatio.toFixed(2) : 'N/A'}</div>
                </div>
                <div class="bg-gray-50 p-4 rounded-lg">
                    <div class="text-gray-500 text-sm">Dividend Yield</div>
                    <div class="font-medium">${info.dividendYield ? info.dividendYield.toFixed(2) + '%' : 'N/A'}</div>
                </div>
                <div class="bg-gray-50 p-4 rounded-lg">
                    <div class="text-gray-500 text-sm">52W High</div>
                    <div class="font-medium">${info['52WeekHigh'] ? toINR(info['52WeekHigh'], currentCurrency) : 'N/A'}</div>
                </div>
                <div class="bg-gray-50 p-4 rounded-lg">
                    <div class="text-gray-500 text-sm">52W Low</div>
                    <div class="font-medium">${info['52WeekLow'] ? toINR(info['52WeekLow'], currentCurrency) : 'N/A'}</div>
                </div>
                <div class="bg-gray-50 p-4 rounded-lg">
                    <div class="text-gray-500 text-sm">Volume</div>
                    <div class="font-medium">${info.volume ? formatNumber(info.volume) : 'N/A'}</div>
                </div>
                <div class="bg-gray-50 p-4 rounded-lg">
                    <div class="text-gray-500 text-sm">Avg Volume</div>
                    <div class="font-medium">${info.avgVolume ? formatNumber(info.avgVolume) : 'N/A'}</div>
                </div>
            `;
        }

        function formatMarketCap(value) {
            if (!value) return 'N/A';
            const inrValue = toINRRaw(value, currentCurrency);
            if (inrValue === null) return 'N/A';
            if (inrValue >= 1e12) return '₹' + (inrValue / 1e12).toFixed(2) + 'T';
            if (inrValue >= 1e9)  return '₹' + (inrValue / 1e9).toFixed(2)  + 'B';
            if (inrValue >= 1e7)  return '₹' + (inrValue / 1e7).toFixed(2)  + 'Cr';
            if (inrValue >= 1e5)  return '₹' + (inrValue / 1e5).toFixed(2)  + 'L';
            return '₹' + inrValue.toFixed(2);
        }

        function formatNumber(num) {
            if (!num) return 'N/A';
            return num.toLocaleString();
        }
    </script>
</body>
</html>
"""

@app.route("/")
def index():
    return Response(INDEX_HTML, mimetype="text/html")

if __name__ == "__main__":
    try:   
        from waitress import serve
        log.info("Running with Waitress on http://127.0.0.1:8000")
        serve(app, host="0.0.0.0", port=8000)
    except ImportError:
        log.info("Waitress not installed; using Flask dev server")
        app.run(host="0.0.0.0", port=8000, debug=True)