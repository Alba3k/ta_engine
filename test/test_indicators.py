import pytest

@pytest.mark.asyncio
async def test_rsi(client):
    resp = await client.get("/indicator/rsi?symbol=AVAX-USDT-SWAP&interval=1h&period=14")
    assert resp.status_code == 200
    data = resp.json()
    assert "value" in data

@pytest.mark.asyncio
async def test_macd(client):
    resp = await client.get("/indicator/macd?symbol=AVAX-USDT-SWAP&interval=1h")
    assert resp.status_code == 200
    data = resp.json()
    assert "values" in data
    assert "macd" in data["values"]
