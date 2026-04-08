#!/usr/bin/env python3
"""
Hub Server Connectivity Verification Script
Verifies network connectivity to the central hub server.

Usage:
    python scripts/verify_hub_connectivity.py
    python scripts/verify_hub_connectivity.py http://10.243.38.174:8000
"""
import asyncio
import os
import sys
import time
from typing import Dict, Any

import httpx
from dotenv import load_dotenv


async def test_hub_connectivity(hub_url: str) -> Dict[str, Any]:
    """Test full connectivity to hub server."""
    results = {
        "hub_url": hub_url,
        "dns_resolution": False,
        "tcp_connect": False,
        "http_response": False,
        "health_check": False,
        "latency_ms": 0,
        "errors": []
    }

    print(f"\n[*] Testing hub connectivity: {hub_url}")
    print("=" * 60)

    # Test 1: DNS & TCP Connectivity
    start_time = time.time()
    try:
        async with httpx.AsyncClient(timeout=5.0) as client:
            response = await client.head(hub_url)
            results["tcp_connect"] = True
            results["dns_resolution"] = True
            print("[OK] TCP connection successful")
    except Exception as e:
        results["errors"].append(f"Connection failed: {str(e)}")
        print(f"[ERR] Connection failed: {str(e)}")

    # Test 2: Health check endpoint
    if results["tcp_connect"]:
        try:
            async with httpx.AsyncClient(timeout=10.0) as client:
                start_time = time.time()
                response = await client.get(f"{hub_url.rstrip('/')}/health")
                latency = (time.time() - start_time) * 1000
                results["latency_ms"] = round(latency, 2)
                
                if response.status_code == 200:
                    results["health_check"] = True
                    results["http_response"] = True
                    health_data = response.json()
                    print(f"[OK] Health check passed (status: {health_data.get('status')})")
                    print(f"[INFO] Round trip latency: {results['latency_ms']} ms")
                else:
                    results["errors"].append(f"Health check returned {response.status_code}")
                    print(f"[WARN] Health check returned status: {response.status_code}")
                    
        except Exception as e:
            results["errors"].append(f"Health check failed: {str(e)}")
            print(f"[ERR] Health check failed: {str(e)}")

    print("\n[*] Results Summary:")
    print("-" * 60)
    all_passed = all([
        results["dns_resolution"],
        results["tcp_connect"],
        results["http_response"],
        results["health_check"]
    ])
    
    if all_passed:
        print("[OK] ALL TESTS PASSED - Hub server is reachable and operational")
        print(f"[OK] Hub URL configured correctly: {hub_url}")
    else:
        print("[ERR] Some tests failed")
        if results["errors"]:
            print("\nErrors encountered:")
            for error in results["errors"]:
                print(f"  - {error}")
    
    return results


def main():
    load_dotenv()
    
    if len(sys.argv) > 1:
        hub_url = sys.argv[1]
    else:
        hub_url = os.getenv("HUB_URL", "http://localhost:8000")
    
    if hub_url == "http://localhost:8000":
        print("\n[WARN] NOTICE: Using default localhost hub URL")
        print("   For remote hub use: python scripts/verify_hub_connectivity.py http://10.243.38.174:8000")
    
    asyncio.run(test_hub_connectivity(hub_url))
    
    print("\n[INFO] Troubleshooting tips:")
    print("   1. Verify hub server is running on 10.243.38.174")
    print("   2. Ensure port 8000 is open in Windows Firewall")
    print("   3. Check both machines are on the same network")
    print("   4. Verify you can ping 10.243.38.174")


if __name__ == "__main__":
    main()