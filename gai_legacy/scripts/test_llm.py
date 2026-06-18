import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from gai.llm.client import LLMClient

client = LLMClient(api_key="test", model="test", base_url="http://localhost:9999")

result = client._execute_tool("run_code", {"code": 'print("hello from tool")'})
print(f"run_code result: {result}")

result2 = client._execute_tool("list_dir", {"path": "."})
print(f"list_dir has content: {bool(result2)}")
print(f"list_dir type: {type(result2)}")

result3 = client._execute_tool("web_search", {"query": "test", "max_results": 2})
print(f"web_search result (first 100): {str(result3)[:100]}")

print("Tool execution: OK")
