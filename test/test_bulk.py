import pytest

@pytest.mark.asyncio
async def test_bulk(client):
    payload = {
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
    resp = await client.post("/bulk", json=payload)
    assert resp.status_code == 200
    data = resp.json()
    assert "data" in data
    assert len(data["data"]) >= 2


