import json
import re
from .tools.web_search import WebSearch
from .tools.code_executor import CodeExecutor
from .tools.file_ops import FileOps


class Agent:
    def __init__(self, model=None, tools=None):
        self.model = model
        self.tools = tools or [WebSearch(), CodeExecutor(), FileOps()]
        self.tool_map = {t.name: t for t in self.tools}
        self.history = []
        self.max_history = 20

    def add_message(self, role, content):
        self.history.append({"role": role, "content": content})
        if len(self.history) > self.max_history:
            self.history = self.history[-self.max_history:]

    def _build_prompt(self, user_input):
        tool_descriptions = []
        for t in self.tools:
            tool_descriptions.append(f"- {t.name}: {t.description}")
        tools_str = "\n".join(tool_descriptions)

        system = f"""You are GAI (General Artificial Intelligence) — an autonomous AI agent with access to tools.

You have the following tools available:
{tools_str}

To use a tool, respond with:
TOOL: tool_name
```json
{{"action": "action_name", "param": "value", ...}}
```

Wait for the result, then continue. When done, respond to the user.

You can:
- Search the internet and fetch web pages
- Write and execute Python code
- Read and write files
- Use your neural network for text generation

Rules:
- Write clean, working code
- Explain what you're doing
- Be concise and helpful
- If you hit an error, debug and fix it"""

        history_str = "\n".join(
            f"{'User' if m['role'] == 'user' else 'Assistant'}: {m['content']}"
            for m in self.history[-6:]
        )

        return f"""{system}

=== Conversation ===
{history_str}
User: {user_input}
Assistant:"""

    def _parse_tool_call(self, text):
        m = re.search(r"TOOL:\s*(\w+)\s*\n```(?:json)?\s*\n?(.*?)```", text, re.DOTALL | re.IGNORECASE)
        if m:
            tool_name = m.group(1).strip()
            try:
                params = json.loads(m.group(2).strip())
                return tool_name, params
            except json.JSONDecodeError:
                return None, None
        m2 = re.search(r"TOOL:\s*(\w+)\s*\n(.*)", text, re.DOTALL | re.IGNORECASE)
        if m2:
            try:
                params = json.loads(m2.group(2).strip())
                return m2.group(1).strip(), params
            except json.JSONDecodeError:
                pass
        return None, None

    def run(self, user_input):
        prompt = self._build_prompt(user_input)
        self.add_message("user", user_input)

        max_iterations = 5
        final_response = ""

        for iteration in range(max_iterations):
            if self.model is not None:
                response = self._generate_from_model(prompt)
            else:
                response = input(f"🤖 Agent prompt (iteration {iteration+1}):\n{prompt}\n> ")

            tool_name, tool_params = self._parse_tool_call(response)

            if tool_name is None:
                final_response = response.strip()
                self.add_message("assistant", final_response)
                return final_response

            if tool_name not in self.tool_map:
                result = f"Unknown tool: {tool_name}. Available: {', '.join(self.tool_map.keys())}"
            else:
                tool = self.tool_map[tool_name]
                try:
                    result = tool.run(**tool_params)
                    if isinstance(result, dict):
                        if "stderr" in result and result["stderr"]:
                            result_text = f"STDERR:\n{result['stderr']}"
                        else:
                            result_text = f"STDOUT:\n{result.get('stdout', '')}"
                        if "success" in result and not result["success"]:
                            result_text = f"Error:\n{result.get('stderr', 'Unknown error')}"
                    elif isinstance(result, str):
                        result_text = result
                    else:
                        result_text = str(result)
                except Exception as e:
                    result_text = f"Tool error: {e}"

            prompt += f"\n\nTOOL RESULT ({tool_name}):\n{result_text}\n\nAssistant:"
            if iteration == max_iterations - 1:
                final_response = f"Result: {result_text}"
                self.add_message("assistant", final_response)

        return final_response or "Max iterations reached."

    def _generate_from_model(self, prompt):
        if self.model is None:
            return ""
        tokens = self.model.tokenizer.encode(prompt)
        import numpy as np
        x = np.array([tokens[:self.model.max_seq_len]], dtype=np.int64)
        from ..tensor import Tensor
        x_t = Tensor(x, requires_grad=False)
        out = self.model.generate(x_t, max_new_tokens=200, temperature=0.8, top_k=50)
        generated = out[0, len(tokens[:self.model.max_seq_len]):].tolist()
        return self.model.tokenizer.decode(generated)

    def chat_loop(self):
        print("=" * 60)
        print("  GAI Agent — ваш ИИ-помощник с выходом в интернет")
        print("  Команды: /exit, /tools, /clear, /help")
        print("=" * 60)
        while True:
            try:
                user_input = input("\nВы: ").strip()
            except (EOFError, KeyboardInterrupt):
                print("\nДо свидания!")
                break

            if not user_input:
                continue
            if user_input.lower() in ("/exit", "/quit"):
                print("До свидания!")
                break
            if user_input.lower() == "/tools":
                print("Доступные инструменты:")
                for t in self.tools:
                    print(f"  - {t.name}: {t.description}")
                continue
            if user_input.lower() == "/clear":
                self.history = []
                print("История очищена.")
                continue
            if user_input.lower() == "/help":
                print("Команды: /exit, /tools, /clear, /help")
                print("Просто пишите запрос — агент сам решит, какие инструменты использовать.")
                continue

            response = self.run(user_input)
            print(f"\n🤖 GAI: {response}")
