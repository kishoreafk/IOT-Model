#!/usr/bin/env python3
"""Standalone liveness and readiness probe for Edge Hub."""
import sys
import argparse
import httpx
import time


def check_health(url: str) -> bool:
    """Check if the hub is alive."""
    try:
        response = httpx.get(f"{url}/health", timeout=5.0)
        if response.status_code == 200:
            data = response.json()
            print(f"[OK] Health check: {data.get('status')}")
            return True
        else:
            print(f"[ERROR] Health check failed: {response.status_code}")
            return False
    except Exception as e:
        print(f"[ERROR] Health check failed: {e}")
        return False


def check_readiness(url: str) -> bool:
    """Check if the hub is ready to accept requests."""
    try:
        response = httpx.get(f"{url}/ready", timeout=5.0)
        if response.status_code == 200:
            data = response.json()
            print(f"[OK] Readiness check: {data.get('status')}")
            print(f"     Index initialized: {data.get('index_initialized')}")
            print(f"     MoE ready: {data.get('moe_ready')}")
            return data.get('index_initialized', False) and data.get('moe_ready', False)
        else:
            print(f"[ERROR] Readiness check failed: {response.status_code}")
            return False
    except Exception as e:
        print(f"[ERROR] Readiness check failed: {e}")
        return False


def main():
    parser = argparse.ArgumentParser(description="Edge Hub Health Check")
    parser.add_argument("--url", default="http://localhost:8000", help="Hub URL")
    parser.add_argument("--health-only", action="store_true", help="Only check health")
    parser.add_argument("--ready-only", action="store_true", help="Only check readiness")
    parser.add_argument("--wait", type=int, default=0, help="Wait N seconds before checking")
    args = parser.parse_args()

    if args.wait > 0:
        print(f"Waiting {args.wait} seconds...")
        time.sleep(args.wait)

    if args.health_only:
        success = check_health(args.url)
    elif args.ready_only:
        success = check_readiness(args.url)
    else:
        health_ok = check_health(args.url)
        ready_ok = check_readiness(args.url) if health_ok else False
        success = health_ok and ready_ok

    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()