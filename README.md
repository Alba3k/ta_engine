
---

# TA Engine Service

**TA Engine** is a dedicated microservice for computing technical analysis (TA) indicators on trading data.  
It is implemented with **FastAPI** and designed to be lightweight, extensible, and easy to integrate into larger trading or analytics platforms. The service exposes a RESTful API with endpoints for single indicator queries, bulk calculations, and health checks, making it suitable for both real‑time and batch workflows.

---

## ✨ Key Features

- **FastAPI‑based REST API** with automatic documentation (Swagger UI & ReDoc).
- **Wide range of indicators** supported: RSI, MACD, ATR, Ichimoku, and more via TA‑Lib.
- **Bulk calculation** endpoint for multi‑symbol, multi‑indicator requests.
- **Flexible parameters**: timeframe, chart type (candles/heikinashi), backtracking, and window length.
- **JSON request/response format** for seamless integration with trading bots, dashboards, or analytics pipelines.
- **Stateless microservice design** — deploy independently or as part of a larger system.
- **Testing support** with pytest and asyncio for reliability.

---

## 🚀 Getting Started

### Prerequisites
- Python 3.10+
- Installed dependencies (see [Dependencies](#-dependencies))

### Running the Service
Start the service locally using **uvicorn**:

```bash
uvicorn services.ta_engine.app:app --reload --port 8001
```

### API Documentation
Once running, interactive documentation is available at:
- [http://127.0.0.1:8000/docs](http://127.0.0.1:8000/docs) — Swagger UI
- [http://127.0.0.1:8000/redoc](http://127.0.0.1:8000/redoc) — ReDoc

---

## 📂 API Endpoints

### `GET /`
Health check endpoint.  
Returns service status:

```json
{"status": "ok"}
```

---

### `GET /indicator/{name}`
Retrieve the value of a specific technical indicator.

**Parameters:**
- `name` — indicator name (e.g., `rsi`, `macd`, `atr`)
- `symbol` — trading pair (e.g., `BTC-USDT-SWAP`)
- `interval` — timeframe (`1h`, `1d`)
- `period` — window length (default: 14)
- `backtrack` — number of candles back
- `chart` — chart type (`candles` or `heikinashi`)
- `addResultTimestamp` — include timestamp in result

---

### `POST /bulk`
Calculate multiple indicators for multiple symbols in one request.

**Request Example:**
```json
{
  "secret": "MY_SECRET",
  "constructs": [
    {
      "symbol": "AVAX-USDT-SWAP",
      "interval": "1h",
      "indicators": [
        {"indicator": "rsi", "period": 14},
        {"indicator": "macd"}
      ]
    },
    {
      "symbol": "LINEA-USDT-SWAP",
      "interval": "1h",
      "indicators": [
        {"indicator": "ichimoku"},
        {"indicator": "atr", "period": 14}
      ]
    }
  ]
}
```

**Response Example:**
```json
{
  "data": [
    {
      "symbol": "AVAX-USDT-SWAP",
      "interval": "1h",
      "results": {
        "rsi": {"value": 52.3},
        "macd": {"values": {"macd": 0.12, "signal": 0.08}}
      }
    },
    {
      "symbol": "LINEA-USDT-SWAP",
      "interval": "1h",
      "results": {
        "ichimoku": {...},
        "atr": {"value": 1.23}
      }
    }
  ]
}
```

---

### `GET /favicon.ico`
Returns the service icon (or an empty response).

---

## 🧪 Testing

Run tests with **pytest**:

```bash
pytest -v
```

---

## 📦 Dependencies

The service relies on the following libraries:

```text
fastapi==0.121.2
uvicorn==0.38.0
pandas==2.3.3
SQLAlchemy==2.0.44
Django==5.2.7
TA-Lib==0.6.8
numpy==2.3.4
httpx==0.27.2
pytest==9.0.1
pytest-asyncio==0.23.8
```

---

## 🏗️ Design Notes

- **Microservice Architecture**: TA Engine is designed to run independently, making it easy to scale horizontally or integrate into a larger system (e.g., alongside authentication, data ingestion, and strategy execution services).
- **Indicator Extensibility**: New indicators can be added by extending the FastAPI routes and leveraging TA‑Lib or custom implementations.
- **Security**: The bulk endpoint supports a `secret` parameter for basic request validation. For production, consider integrating with a full authentication service.
- **Deployment**: Can be containerized with Docker and orchestrated via Kubernetes or systemd for production reliability.
- **Use Cases**:
  - Trading bots needing real‑time indicator values.
  - Analytics dashboards displaying multiple indicators.
  - Backtesting pipelines requiring bulk indicator calculations.

---

## ✅ Summary

- **TA Engine** provides a robust API for technical indicator calculations.  
- Endpoints are documented with Swagger and ReDoc for easy exploration.  
- Bulk requests allow efficient multi‑symbol analysis.  
- Dependencies are pinned for reproducibility and stability.  
- Designed for integration into trading platforms, analytics dashboards, or automated strategy pipelines.  

---
