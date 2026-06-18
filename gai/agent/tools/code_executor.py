import sys
import io
import traceback
import ast
import textwrap
import importlib


class CodeExecutor:
    def __init__(self):
        self.name = "code_executor"
        self.description = "Write and execute Python code with standard library support"
        self._namespace = {}
        self._allowed_modules = {
            "json", "re", "math", "random", "datetime", "collections",
            "itertools", "functools", "typing", "string", "decimal",
            "fractions", "statistics", "uuid", "hashlib", "base64",
            "textwrap", "pprint", "copy", "pathlib", "csv", "html",
            "xml", "urllib", "email", "enum", "numbers",
        }

    def _check_safety(self, code):
        tree = ast.parse(code)
        for node in ast.walk(tree):
            if isinstance(node, ast.Call):
                if isinstance(node.func, ast.Name) and node.func.id in ("eval", "exec", "compile", "__import__"):
                    return False, f"'{node.func.id}' is blocked for security"
                if isinstance(node.func, ast.Attribute):
                    if isinstance(node.func.value, ast.Name):
                        if node.func.value.id in ("os", "subprocess", "shutil", "socket", "ctypes"):
                            return False, f"Module '{node.func.value.id}' is restricted"
                        if node.func.value.id == "sys" and node.func.attr in ("exit", "quit"):
                            return False, "sys.exit/quit is blocked"
            if isinstance(node, ast.Import):
                for alias in node.names:
                    base = alias.name.split(".")[0]
                    if base not in self._allowed_modules and base not in ("gai", "numpy", "pandas", "matplotlib", "requests", "beautifulsoup4", "bs4"):
                        return False, f"Import of '{alias.name}' is not allowed"
            if isinstance(node, ast.ImportFrom):
                base = node.module.split(".")[0] if node.module else ""
                if base not in self._allowed_modules and base not in ("gai", "numpy", "pandas", "matplotlib", "requests", "bs4"):
                    return False, f"Import from '{node.module}' is not allowed"
        return True, ""

    def run_code(self, code, timeout=10):
        code = textwrap.dedent(code)
        safe, msg = self._check_safety(code)
        if not safe:
            return {"stdout": "", "stderr": msg, "success": False}

        safe_builtins = {
            "abs": abs, "all": all, "any": any, "bin": bin, "bool": bool,
            "bytearray": bytearray, "bytes": bytes, "chr": chr, "complex": complex,
            "dict": dict, "dir": dir, "divmod": divmod, "enumerate": enumerate,
            "filter": filter, "float": float, "format": format, "frozenset": frozenset,
            "getattr": getattr, "hasattr": hasattr, "hash": hash, "hex": hex,
            "id": id, "int": int, "isinstance": isinstance, "issubclass": issubclass,
            "iter": iter, "len": len, "list": list, "map": map, "max": max,
            "min": min, "next": next, "object": object, "oct": oct, "ord": ord,
            "pow": pow, "print": print, "range": range, "repr": repr,
            "reversed": reversed, "round": round, "set": set, "slice": slice,
            "sorted": sorted, "str": str, "sum": sum, "tuple": tuple, "type": type,
            "zip": zip, "True": True, "False": False, "None": None,
            "open": open, "input": input,
        }

        def safe_import(name, *args, **kwargs):
            base = name.split(".")[0]
            if base in self._allowed_modules:
                return importlib.import_module(name)
            raise ImportError(f"Module '{base}' is not allowed. Allowed: {sorted(self._allowed_modules)}")

        safe_builtins["__import__"] = safe_import

        stdout_buf = io.StringIO()
        stderr_buf = io.StringIO()
        old_stdout = sys.stdout
        old_stderr = sys.stderr

        result = {"stdout": "", "stderr": "", "success": True}
        try:
            sys.stdout = stdout_buf
            sys.stderr = stderr_buf
            exec(code, {"__builtins__": safe_builtins}, self._namespace)
        except Exception:
            result["stderr"] = traceback.format_exc()
            result["success"] = False
        finally:
            sys.stdout = old_stdout
            sys.stderr = old_stderr

        result["stdout"] = stdout_buf.getvalue()
        result["stderr"] = (stderr_buf.getvalue() or "") + (result.get("stderr", "") or "")
        return result

    def run(self, action, **kwargs):
        if action == "run_code":
            return self.run_code(kwargs.get("code", ""), kwargs.get("timeout", 10))
        return {"stdout": "", "stderr": f"Unknown action: {action}", "success": False}
