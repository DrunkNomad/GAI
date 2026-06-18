import sys
import os

if getattr(sys, "frozen", False):
    base = sys._MEIPASS
else:
    base = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

sys.path.insert(0, base)

from gai.ui import run_ui

if __name__ == "__main__":
    run_ui()
