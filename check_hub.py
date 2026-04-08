import asyncio
import httpx

async def check():
    HUB = "http://10.243.38.174:8000"
    async with httpx.AsyncClient(timeout=30) as c:
        r = await c.get(f'{HUB}/status')
        print("STATUS:", r.json())
        
        r = await c.get(f'{HUB}/clusters')
        print("CLUSTERS:", r.json())
        
        r = await c.get(f'{HUB}/devices')
        print("DEVICES:", r.json())

asyncio.run(check())
