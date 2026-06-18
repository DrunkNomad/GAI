import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from gai.agent import Agent


def main():
    agent = Agent()
    agent.chat_loop()


if __name__ == "__main__":
    main()
