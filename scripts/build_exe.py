"""
Build GAI Agent as a standalone .exe file.
Usage: python scripts/build_exe.py
"""
import sys
import os
import shutil

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def build():
    output_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "dist")

    if os.path.exists(output_dir):
        shutil.rmtree(output_dir)
    os.makedirs(output_dir, exist_ok=True)

    entry_point = os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
        "scripts", "ui.py"
    )

    if not os.path.exists(entry_point):
        print(f"Entry point not found: {entry_point}")
        sys.exit(1)

    print("=" * 60)
    print("  Building GAI Agent .exe")
    print("=" * 60)
    print(f"  Entry point: {entry_point}")
    print(f"  Output: {output_dir}")
    print()

    import PyInstaller.__main__

    args = [
        entry_point,
        "--name=GAI_Agent",
        "--onefile",
        "--windowed",
        f"--distpath={output_dir}",
        "--clean",
        "--noconfirm",
        "--add-data=gai;gai",
        "--hidden-import=numpy",
        "--hidden-import=requests",
        "--hidden-import=bs4",
        "--hidden-import=duckduckgo_search",
        "--hidden-import=ddgs",
        "--hidden-import=customtkinter",
        "--hidden-import=ctypes",
        "--hidden-import=packaging",
        "--collect-all=customtkinter",
    ]

    print("  PyInstaller arguments:")
    for a in args:
        print(f"    {a}")
    print()

    print("  Starting build... (this may take a few minutes)")
    PyInstaller.__main__.run(args)

    exe_path = os.path.join(output_dir, "GAI_Agent.exe")
    if os.path.exists(exe_path):
        size_mb = os.path.getsize(exe_path) / (1024 * 1024)
        print(f"\n  Success! .exe created: {exe_path}")
        print(f"  Size: {size_mb:.1f} MB")
    else:
        print("\n  Build may have failed. Check the dist directory.")
        if os.path.exists(output_dir):
            for f in os.listdir(output_dir):
                print(f"    {f}")


if __name__ == "__main__":
    build()
