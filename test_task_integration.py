import asyncio
import httpx
import sys
import os
sys.path.insert(0, os.getcwd())

# Test task tracker integration with hub
async def test_task_integration():
    from central_hub import hub_server
    
    print("Testing task tracker directly in hub_server...")
    
    # Check if task_tracker is initialized
    print(f"task_tracker exists: {hub_server.task_tracker is not None}")
    
    # Manually create a task
    task_id = "manual_test_001"
    hub_server.task_tracker.create(task_id, "test_type")
    print(f"Created task: {task_id}")
    
    # Get task
    task = hub_server.task_tracker.get(task_id)
    print(f"Get task result: {task}")
    
    # Complete task
    hub_server.task_tracker.complete(task_id, {"result": "test_success"})
    task = hub_server.task_tracker.get(task_id)
    print(f"After completion: {task}")
    
    # Now test via API
    async with httpx.AsyncClient(timeout=30) as client:
        r = await client.get(f'http://localhost:8000/tasks/{task_id}')
        print(f"\nAPI Response status: {r.status_code}")
        print(f"API Response: {r.text}")

asyncio.run(test_task_integration())
