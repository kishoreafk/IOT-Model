import asyncio
import httpx

async def quick_test():
    try:
        async with httpx.AsyncClient(timeout=10) as client:
            r = await client.get("http://10.243.38.174:8000/health")
            print(f"Status: {r.status_code}")
            print(f"Response: {r.json()}")
    except Exception as e:
        print(f"Error: {e}")

asyncio.run(quick_test())
