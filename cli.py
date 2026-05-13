import sys
import os

# 确保可以导入 backend 目录下的模块
current_dir = os.path.dirname(os.path.abspath(__file__))
backend_dir = os.path.dirname(current_dir)
if backend_dir not in sys.path:
    sys.path.append(backend_dir)

from agent_core import ReactAgent
from agent_core.tools import create_admin_tool, create_knowledge_tool
from agent_core.rag import HybridSearcher
from agent_core.config.settings import settings
from database.course_repo import query_course_admin, init_db
from app.utils.logger import setup_logger

def run_cli():
    setup_logger()
    print("=" * 48)
    print("    离散数学智能助教 (ReAct CLI)")
    print("=" * 48)
    print('输入 "exit" 退出，"clear" 重置对话\n')

    init_db()
    hybrid_searcher = HybridSearcher()

    tools = [
        create_admin_tool(query_course_admin),
        create_knowledge_tool(hybrid_searcher.query)
    ]

    agent = ReactAgent(config=settings, tools=tools)

    while True:
        try:
            user_input = input("你：").strip()
        except (KeyboardInterrupt, EOFError):
            break

        if not user_input: continue
        if user_input.lower() in ["exit", "quit", "退出"]: break
        if user_input.lower() in ["clear", "reset", "清空"]:
            agent.reset()
            print("对话已重置。")
            continue

        try:
            print("助教：思考中...", end="\r")
            reply = agent.chat(user_input)
            print(f"助教：{reply}\n")
        except Exception as e:
            print(f"\n[错误] {e}\n")

if __name__ == "__main__":
    run_cli()
