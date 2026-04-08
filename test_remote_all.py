"""
Test all hub endpoints against remote hub
"""

import asyncio
import httpx
import sys
import os
sys.path.insert(0, os.getcwd())

REMOTE_HUB_URL = os.getenv("HUB_URL", "http://localhost:8000")

async def test_all_endpoints():
    print(f"Testing all endpoints: {REMOTE_HUB_URL}")
    print("=" * 60)
    
    endpoints = [
        ("GET", "/health"),
        ("GET", "/ready"),
        ("GET", "/status"),
        ("GET", "/clusters"),
        ("GET", "/moe/status"),
        ("GET", "/devices"),
        ("GET", "/adapters/latest/version"),
        ("GET", "/tasks/test"),
    ]
    
    results = []
    
    async with httpx.AsyncClient(timeout=30) as client:
        for method, path in endpoints:
            try:
                if method == "GET":
                    resp = await client.get(f"{REMOTE_HUB_URL}{path}")
                else:
                    resp = await client.request(method, f"{REMOTE_HUB_URL}{path}")
                
                print(f"\n[{method}] {path}")
                print(f"  Status: {resp.status_code}")
                if resp.status_code == 200:
                    try:
                        print(f"  Response: {resp.json()}")
                    except:
                        print(f"  Response: {resp.text[:200]}")
                else:
                    print(f"  Error: {resp.text[:200]}")
                
                results.append((path, resp.status_code))
                
            except Exception as e:
                print(f"\n[{method}] {path}")
                print(f"  Exception: {e}")
                results.append((path, 0))
    
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    
    for path, status in results:
        if status == 200:
            print(f"[OK] {path}: {status}")
        elif status == 404:
            print(f"[WARN] {path}: 404 Not Found")
        elif status == 500:
            print(f"[ERR] {path}: 500 Internal Server Error")
        else:
            print(f"[FAIL] {path}: {status}")

asyncio.run(test_all_endpoints())
