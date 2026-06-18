import os
from pathlib import Path


class FileOps:
    def __init__(self):
        self.name = "file_ops"
        self.description = "Read, write, list, and manage files in the workspace"

    def read(self, path):
        p = Path(path)
        if not p.exists():
            return f"File not found: {path}"
        try:
            return p.read_text(encoding="utf-8")
        except Exception as e:
            return f"Error reading {path}: {e}"

    def write(self, path, content):
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(content, encoding="utf-8")
        return f"Written {len(content)} bytes to {path}"

    def list_dir(self, path="."):
        p = Path(path)
        if not p.exists():
            return f"Directory not found: {path}"
        items = []
        for item in p.iterdir():
            prefix = "[DIR] " if item.is_dir() else "[FILE] "
            items.append(f"{prefix}{item.name}")
        return "\n".join(items) if items else "(empty directory)"

    def delete(self, path):
        p = Path(path)
        if not p.exists():
            return f"Not found: {path}"
        if p.is_dir():
            import shutil
            shutil.rmtree(p)
            return f"Deleted directory: {path}"
        else:
            p.unlink()
            return f"Deleted file: {path}"

    def exists(self, path):
        return str(Path(path).exists())

    def run(self, action, **kwargs):
        if action == "read":
            return self.read(kwargs.get("path", ""))
        elif action == "write":
            return self.write(kwargs.get("path", ""), kwargs.get("content", ""))
        elif action == "list":
            return self.list_dir(kwargs.get("path", "."))
        elif action == "delete":
            return self.delete(kwargs.get("path", ""))
        elif action == "exists":
            return self.exists(kwargs.get("path", ""))
        return f"Unknown action: {action}. Available: read, write, list, delete, exists"
