import json
import requests


TOOL_DEFINITIONS = [
    {
        "type": "function",
        "function": {
            "name": "web_search",
            "description": "Search the internet for information on a query. Use this when you need current information.",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {"type": "string", "description": "Search query"},
                    "max_results": {"type": "integer", "description": "Number of results (1-10)", "default": 5},
                },
                "required": ["query"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "fetch_page",
            "description": "Fetch and read the content of a webpage",
            "parameters": {
                "type": "object",
                "properties": {
                    "url": {"type": "string", "description": "Full URL to fetch"},
                },
                "required": ["url"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "run_code",
            "description": "Execute Python code and return the output",
            "parameters": {
                "type": "object",
                "properties": {
                    "code": {"type": "string", "description": "Python code to execute"},
                },
                "required": ["code"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "read_file",
            "description": "Read the contents of a file",
            "parameters": {
                "type": "object",
                "properties": {
                    "path": {"type": "string", "description": "Path to the file"},
                },
                "required": ["path"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "write_file",
            "description": "Write content to a file (creates directories if needed)",
            "parameters": {
                "type": "object",
                "properties": {
                    "path": {"type": "string", "description": "Path to the file"},
                    "content": {"type": "string", "description": "Content to write"},
                },
                "required": ["path", "content"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "list_dir",
            "description": "List files and directories in a folder",
            "parameters": {
                "type": "object",
                "properties": {
                    "path": {"type": "string", "description": "Directory path", "default": "."},
                },
                "required": [],
            },
        },
    },
]


SYSTEM_PROMPT = (
    "Ты — GAI, интеллектуальный ИИ-агент с доступом к инструментам.\n\n"
    "Твои инструменты:\n"
    "- web_search — поиск в интернете (передай query)\n"
    "- fetch_page — загрузить содержимое страницы (передай url)\n"
    "- run_code — выполнить Python код (передай code)\n"
    "- read_file — прочитать файл (передай path)\n"
    "- write_file — записать файл (передай path и content)\n"
    "- list_dir — список файлов в директории (передай path)\n\n"
    "ПРАВИЛА:\n"
    "1. Отвечай ПО-РУССКИ. Всегда на русском языке.\n"
    "2. Если пользователь просит что-то, что требует инструмента — используй его.\n"
    "3. Если для ответа нужна информация из интернета — сначала сделай search.\n"
    "4. После получения результата инструмента — проанализируй его и дай пользователю понятный ответ.\n"
    "5. Будь дружелюбным, полезным и точным.\n"
    "6. Если что-то пошло не так — объясни, что случилось.\n"
    "7. Код должен быть чистым, рабочим, с комментариями на русском.\n"
    "8. Не выдумывай — если не знаешь, лучше поищи в интернете.\n"
    "9. Твоя личность — GAI, ты был создан как экспериментальный ИИ-агент.\n"
)


class LLMClient:
    def __init__(self, api_key="", model="gpt-4o-mini", base_url="https://api.openai.com/v1"):
        self.api_key = api_key
        self.model = model
        self.base_url = base_url.rstrip("/")
        self.messages = [{"role": "system", "content": SYSTEM_PROMPT}]

    def set_config(self, api_key, model, base_url):
        self.api_key = api_key
        self.model = model
        self.base_url = base_url.rstrip("/")

    def reset(self):
        self.messages = [{"role": "system", "content": SYSTEM_PROMPT}]

    def add_message(self, role, content):
        self.messages.append({"role": role, "content": content})

    def add_tool_result(self, tool_call_id, name, result):
        if isinstance(result, dict):
            result = json.dumps(result, ensure_ascii=False)
        self.messages.append({
            "role": "tool",
            "tool_call_id": tool_call_id,
            "content": str(result),
        })

    def chat(self, user_message, tools_enabled=True):
        self.add_message("user", user_message)

        max_turns = 6
        for turn in range(max_turns):
            response = self._call_api(tools_enabled and turn < max_turns - 1)
            msg = response["choices"][0]["message"]

            if not msg.get("tool_calls"):
                self.messages.append(msg)
                return msg["content"]

            self.messages.append(msg)

            for tc in msg["tool_calls"]:
                fn = tc["function"]
                name = fn["name"]
                try:
                    args = json.loads(fn["arguments"])
                except json.JSONDecodeError:
                    args = {}

                result = self._execute_tool(name, args)
                self.add_tool_result(tc["id"], name, result)

        final = self._call_api(False)
        text = final["choices"][0]["message"]["content"]
        self.messages.append({"role": "assistant", "content": text})
        return text

    def _call_api(self, tools_enabled):
        body = {
            "model": self.model,
            "messages": self.messages,
        }
        if tools_enabled:
            body["tools"] = TOOL_DEFINITIONS
            body["tool_choice"] = "auto"

        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self.api_key}",
        }

        resp = requests.post(
            f"{self.base_url}/chat/completions",
            headers=headers,
            json=body,
            timeout=60,
        )

        if resp.status_code == 401:
            raise PermissionError("Неверный API ключ")
        if resp.status_code == 404:
            raise ConnectionError(f"Модель '{self.model}' не найдена. Проверь endpoint и имя модели")
        if resp.status_code != 200:
            raise ConnectionError(f"API ошибка {resp.status_code}: {resp.text[:300]}")

        return resp.json()

    def _execute_tool(self, name, args):
        from ..agent.tools.web_search import WebSearch
        from ..agent.tools.code_executor import CodeExecutor
        from ..agent.tools.file_ops import FileOps

        ws = WebSearch()
        ce = CodeExecutor()
        fo = FileOps()

        tool_map = {
            "web_search": lambda: ws.search(args.get("query", ""), args.get("max_results", 5)),
            "fetch_page": lambda: ws.fetch_page(args.get("url", "")),
            "run_code": lambda: ce.run_code(args.get("code", "")),
            "read_file": lambda: fo.read(args.get("path", "")),
            "write_file": lambda: fo.write(args.get("path", ""), args.get("content", "")),
            "list_dir": lambda: fo.list_dir(args.get("path", ".")),
        }

        fn = tool_map.get(name)
        if fn is None:
            return f"Неизвестный инструмент: {name}"

        try:
            result = fn()
            if isinstance(result, dict) and "stderr" in result:
                if result.get("stderr"):
                    return {"error": result["stderr"], "output": result.get("stdout", "")}
                return {"output": result.get("stdout", "")}
            return result
        except Exception as e:
            return {"error": str(e)}
