import pytest

@pytest.mark.asyncio
async def test_math_sum(client):
    resp = await client.get("/indicator/sum?symbol=AVAX-USDT-SWAP&interval=1h")
    assert resp.status_code == 200
    data = resp.json()
    assert "value" in data
