import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from gai.agent.tools import WebSearch, CodeExecutor, FileOps


def test_code_executor():
    print("=== Test: CodeExecutor ===")
    exec = CodeExecutor()
    result = exec.run_code("print('hello from agent'); result = 42")
    print(f"  stdout: {result['stdout'].strip()}")
    print(f"  success: {result['success']}")

    result = exec.run_code("1 + 1")
    print(f"  result: {result['stdout'].strip()}, success: {result['success']}")

    result = exec.run_code("import math; print(math.pi)")
    print(f"  math import: {result['stdout'].strip()}, success: {result['success']}")

    result = exec.run_code("import os; print('hack')")
    print(f"  blocked os: success={result['success']}, stderr={result['stderr'][:50]}")
    print("  CodeExecutor: OK" if result['success'] == False else "  FAIL")


def test_file_ops():
    print("\n=== Test: FileOps ===")
    fo = FileOps()
    result = fo.write("test_agent_file.txt", "Hello from GAI agent!")
    print(f"  Write: {result}")

    result = fo.read("test_agent_file.txt")
    print(f"  Read: {result}")

    result = fo.list_dir(".")
    print(f"  List (first 3): {' | '.join(result.split(chr(10))[:3])}")

    fo.delete("test_agent_file.txt")
    print("  Cleanup: OK")


def test_web_search():
    print("\n=== Test: WebSearch ===")
    ws = WebSearch()
    result = ws.search("Python neural network tutorial", max_results=2)
    print(f"  Search results (first 400 chars):\n  {result[:400]}")
    print("  WebSearch: OK")


if __name__ == "__main__":
    test_code_executor()
    test_file_ops()
    test_web_search()
    print("\nAll agent tests done!")
