import asyncio
import httpx

async def check():
    async with httpx.AsyncClient(timeout=30) as c:
        r = await c.get('http://localhost:8000/status')
        print("STATUS:", r.json())
        
        r = await c.get('http://localhost:8000/clusters')
        print("CLUSTERS:", r.json())
        
        r = await c.get('http://localhost:8000/devices')
        print("DEVICES:", r.json())
        
        r = await c.get('http://localhost:8000/tasks')
        print("TASKS:", r.json())

asyncio.run(check())
