import customtkinter as ctk
from tkinter import messagebox
import tkinter as tk
import threading
import re
import json
import sys
import os

from ..llm import LLMClient


ctk.set_appearance_mode("dark")
ctk.set_default_color_theme("green")

FONT = ("Segoe UI", 13)
FONT_BOLD = ("Segoe UI", 13, "bold")
FONT_SMALL = ("Segoe UI", 11)
FONT_TITLE = ("Segoe UI", 18, "bold")
FONT_CODE = ("Consolas", 12)

COLOR_BG = "#0b0f1a"
COLOR_SIDEBAR = "#0f1629"
COLOR_INPUT = "#161e33"
COLOR_BORDER = "#1e2a45"
COLOR_USER_MSG = "#162033"
COLOR_AI_MSG = "#0f1629"
COLOR_GREEN = "#22c55e"
COLOR_ACCENT = "#3b82f6"
COLOR_TEXT = "#e2e8f0"
COLOR_MUTED = "#64748b"

HEADER_HEIGHT = 56
INPUT_HEIGHT = 64

TOOL_NAMES = {
    "web_search": "\U0001f50d Поиск в интернете",
    "fetch_page": "\U0001f310 Загрузить страницу",
    "run_code": "\u2328 Выполнить код",
    "read_file": "\U0001f4c4 Читать файл",
    "write_file": "\U0001f4dd Записать файл",
    "list_dir": "\U0001f4c1 Список файлов",
}


class MarkdownLabel(ctk.CTkFrame):
    def __init__(self, master, text, **kwargs):
        super().__init__(master, fg_color="transparent", **kwargs)
        self.pack(fill="x", padx=0, pady=0)
        self._render(text)

    def _render(self, text):
        parts = re.split(r"(`[^`]+`|&lt;[^&gt;]+&gt;|\*\*[^*]+\*\*|\*[^*]+\*)", text)
        for part in parts:
            if part.startswith("`") and part.endswith("`"):
                ctk.CTkLabel(
                    self, text=part[1:-1], font=FONT_CODE,
                    fg_color="#1e293b", text_color="#f472b6",
                    padx=6, pady=2,
                ).pack(side="left", padx=(0, 4))
            elif part.startswith("**") and part.endswith("**"):
                ctk.CTkLabel(
                    self, text=part[2:-2], font=FONT_BOLD,
                    text_color=COLOR_TEXT,
                ).pack(side="left")
            elif part.startswith("*") and part.endswith("*"):
                ctk.CTkLabel(
                    self, text=part[1:-1], font=("Segoe UI", 13, "italic"),
                    text_color=COLOR_TEXT,
                ).pack(side="left")
            elif part.strip():
                ctk.CTkLabel(
                    self, text=part, font=FONT,
                    text_color=COLOR_TEXT, wraplength=600, justify="left",
                ).pack(side="left")


class Avatar(ctk.CTkFrame):
    def __init__(self, master, letter, color, **kwargs):
        super().__init__(
            master, width=36, height=36, fg_color=color,
            corner_radius=18, **kwargs
        )
        self.pack_propagate(False)
        ctk.CTkLabel(
            self, text=letter, font=("Segoe UI", 14, "bold"),
            text_color="white",
        ).place(relx=0.5, rely=0.5, anchor="center")


class MessageBubble(ctk.CTkFrame):
    def __init__(self, master, text, role="user", tool_name=None, **kwargs):
        super().__init__(master, fg_color="transparent", **kwargs)
        self.pack(fill="x", padx=0, pady=(0, 2))

        is_user = role == "user"
        avatar_letter = "T" if tool_name else ("Y" if is_user else "G")
        avatar_color = COLOR_ACCENT if is_user else COLOR_GREEN
        bg_color = COLOR_USER_MSG if is_user else COLOR_AI_MSG

        container = ctk.CTkFrame(self, fg_color="transparent")
        container.pack(fill="x", padx=20, pady=(8, 2))

        if is_user:
            inner = ctk.CTkFrame(container, fg_color="transparent")
            inner.pack(side="right", fill="x", padx=(60, 0))

            Avatar(inner, avatar_letter, avatar_color).pack(side="right", padx=(8, 0), pady=(2, 0))

            bubble = ctk.CTkFrame(
                inner, fg_color=bg_color, corner_radius=16,
                border_width=0,
            )
            bubble.pack(side="right", padx=0, pady=0)

            ctk.CTkLabel(
                bubble, text=text, font=FONT,
                text_color=COLOR_TEXT, wraplength=500, justify="left",
            ).pack(padx=16, pady=10)
        else:
            inner = ctk.CTkFrame(container, fg_color="transparent")
            inner.pack(side="left", fill="x", padx=(0, 60))

            Avatar(inner, avatar_letter, avatar_color).pack(side="left", padx=(0, 8), pady=(2, 0))

            bubble = ctk.CTkFrame(
                inner, fg_color=bg_color, corner_radius=16,
                border_width=0,
            )
            bubble.pack(side="left", fill="x", padx=0, pady=0, expand=True)

            self._render_content(bubble, text, tool_name)

    def _render_content(self, parent, text, tool_name):
        if tool_name:
            name = TOOL_NAMES.get(tool_name, f"\u2699 {tool_name}")
            ctk.CTkLabel(
                parent, text=name, font=FONT_BOLD,
                text_color=COLOR_GREEN,
            ).pack(padx=16, pady=(10, 0), anchor="w")

        code_blocks = re.split(r"(```[\s\S]*?```)", text)
        for block in code_blocks:
            if block.startswith("```"):
                code = re.sub(r"```\w*\n?", "", block).strip()
                code_frame = ctk.CTkFrame(
                    parent, fg_color="#020617", corner_radius=8,
                    border_width=1, border_color="#1e2a45",
                )
                code_frame.pack(fill="x", padx=12, pady=6)

                code_label = ctk.CTkLabel(
                    code_frame, text=code, font=FONT_CODE,
                    text_color="#e2e8f0", justify="left",
                    anchor="w", wraplength=500,
                )
                code_label.pack(padx=12, pady=8)

                lines = code.count("\n") + 1
                if lines > 15:
                    code_frame.configure(height=400)
                    code_label.configure(height=380)
            else:
                if block.strip():
                    ctk.CTkLabel(
                        parent, text=block.strip(), font=FONT,
                        text_color=COLOR_TEXT, wraplength=500,
                        justify="left",
                    ).pack(padx=16, pady=(6, 10), anchor="w")


class SettingsPanel(ctk.CTkFrame):
    def __init__(self, master, app, **kwargs):
        super().__init__(master, fg_color=COLOR_SIDEBAR, **kwargs)
        self.app = app
        self.configure(width=280)
        self.pack_propagate(False)
        self._build()

    def _build(self):
        ctk.CTkLabel(
            self, text="\u2699 Настройки", font=FONT_BOLD,
            text_color=COLOR_GREEN,
        ).pack(pady=(20, 16))

        sections = [
            ("Подключение", [
                ("API ключ", "api_key", "", True),
                ("Модель", "model", "gpt-4o-mini"),
                ("Endpoint", "base_url", "https://api.openai.com/v1"),
            ]),
            ("О системе", [
                ("Версия", "version", "1.0.0"),
                ("Инструментов", "tools", "6"),
            ]),
        ]

        self.entries = {}

        for section_name, fields in sections:
            frame = ctk.CTkFrame(self, fg_color="#0d1322", corner_radius=8)
            frame.pack(fill="x", padx=12, pady=6)

            ctk.CTkLabel(
                frame, text=section_name, font=FONT_SMALL,
                text_color="#94a3b8",
            ).pack(padx=12, pady=(8, 4), anchor="w")

            for label, key, default, *opts in fields:
                is_password = opts[0] if opts else False
                row = ctk.CTkFrame(frame, fg_color="transparent")
                row.pack(fill="x", padx=12, pady=2)

                ctk.CTkLabel(
                    row, text=label, font=FONT_SMALL,
                    text_color=COLOR_TEXT, anchor="w",
                ).pack(anchor="w")

                if is_password:
                    entry = ctk.CTkEntry(
                        row, font=FONT_SMALL, height=30,
                        fg_color=COLOR_INPUT, border_color=COLOR_BORDER,
                        show="*",
                    )
                elif key == "version" or key == "tools":
                    entry = ctk.CTkLabel(
                        row, text=default, font=FONT_SMALL,
                        text_color=COLOR_MUTED, anchor="w",
                    )
                else:
                    entry = ctk.CTkEntry(
                        row, font=FONT_SMALL, height=30,
                        fg_color=COLOR_INPUT, border_color=COLOR_BORDER,
                    )
                entry.pack(fill="x", pady=(0, 4))
                if hasattr(entry, "insert"):
                    entry.insert(0, default)
                self.entries[key] = entry

        self.connect_btn = ctk.CTkButton(
            self,
            text="\U0001f517 Подключить",
            height=36,
            font=("Segoe UI", 12, "bold"),
            fg_color=COLOR_GREEN,
            hover_color="#16a34a",
            text_color="#000000",
            command=self._connect,
        )
        self.connect_btn.pack(fill="x", padx=16, pady=(16, 4))

        self.status_label = ctk.CTkLabel(
            self, text="", font=FONT_SMALL, text_color=COLOR_MUTED,
        )
        self.status_label.pack(pady=(0, 16))

    def _connect(self):
        api_key = self.entries["api_key"].get().strip()
        model = self.entries["model"].get().strip()
        base_url = self.entries["base_url"].get().strip()

        if not api_key:
            self.status_label.configure(text="\u26a0 Введи API ключ", text_color="#f59e0b")
            return
        if not model:
            self.status_label.configure(text="\u26a0 Укажи модель", text_color="#f59e0b")
            return

        self.connect_btn.configure(state="disabled", text="Подключение...")
        self.status_label.configure(text="")

        def connect():
            try:
                client = LLMClient(api_key=api_key, model=model, base_url=base_url)
                test = client._call_api(False)
                self.app.on_connected(api_key, model, base_url)
                self.status_label.configure(
                    text="\u2713 Подключено!", text_color=COLOR_GREEN
                )
            except Exception as e:
                self.status_label.configure(
                    text=f"\u2718 {e}", text_color="#ef4444"
                )
            finally:
                self.connect_btn.configure(state="normal", text="\U0001f517 Подключить")

        threading.Thread(target=connect, daemon=True).start()


class StatusBar(ctk.CTkFrame):
    def __init__(self, master, **kwargs):
        super().__init__(master, fg_color="#070b14", corner_radius=0, height=26, **kwargs)
        self.pack_propagate(False)

        self.dot = ctk.CTkLabel(
            self, text="\u25cf", font=FONT_SMALL,
            text_color="#ef4444",
        )
        self.dot.pack(side="left", padx=(12, 4))

        self.status = ctk.CTkLabel(
            self, text="Не подключено", font=FONT_SMALL,
            text_color=COLOR_MUTED,
        )
        self.status.pack(side="left")

        self.model_label = ctk.CTkLabel(
            self, text="", font=FONT_SMALL,
            text_color=COLOR_MUTED,
        )
        self.model_label.pack(side="right", padx=12)

    def set_connected(self, model_name=""):
        self.dot.configure(text_color=COLOR_GREEN)
        self.status.configure(text="Подключено")
        if model_name:
            self.model_label.configure(text=f"Модель: {model_name}")

    def set_disconnected(self):
        self.dot.configure(text_color="#ef4444")
        self.status.configure(text="Не подключено")
        self.model_label.configure(text="")

    def set_working(self):
        self.dot.configure(text_color="#f59e0b")
        self.status.configure(text="Думаю...")

    def set_ready(self):
        self.dot.configure(text_color=COLOR_GREEN)
        self.status.configure(text="Готов")


class GAIApp(ctk.CTk):
    def __init__(self):
        super().__init__()

        self.title("GAI — Интеллектуальный ИИ-агент")
        self.geometry("1200x780")
        self.minsize(900, 600)

        self.llm = None
        self.connected = False
        self.running = False
        self.message_history = []

        self._build_ui()
        self._bind_shortcuts()
        self.after(300, self._show_welcome)

    def _build_ui(self):
        self.grid_rowconfigure(0, weight=1)
        self.grid_rowconfigure(1, weight=0)
        self.grid_columnconfigure(1, weight=1)
        self.grid_columnconfigure(0, weight=0)

        left_panel = ctk.CTkFrame(self, fg_color=COLOR_SIDEBAR, width=280, corner_radius=0)
        left_panel.grid(row=0, column=0, rowspan=2, sticky="ns")
        left_panel.grid_propagate(False)

        ctk.CTkLabel(
            left_panel,
            text="GAI",
            font=("Segoe UI", 24, "bold"),
            text_color=COLOR_GREEN,
        ).pack(pady=(24, 4))
        ctk.CTkLabel(
            left_panel,
            text="ИИ-агент с инструментами",
            font=FONT_SMALL,
            text_color=COLOR_MUTED,
        ).pack(pady=(0, 16))

        tool_frame = ctk.CTkFrame(left_panel, fg_color="transparent")
        tool_frame.pack(fill="x", padx=12)
        ctk.CTkLabel(
            tool_frame, text="Инструменты", font=FONT_BOLD,
            text_color=COLOR_GREEN,
        ).pack(anchor="w", pady=(0, 6))

        tools_info = [
            ("\U0001f50d", "Поиск", "Интернет"),
            ("\u2328", "Код", "Python"),
            ("\U0001f4c4", "Файлы", "Чтение/запись"),
        ]
        for icon, name, desc in tools_info:
            card = ctk.CTkFrame(tool_frame, fg_color="#0d1322", corner_radius=8)
            card.pack(fill="x", pady=2)
            ctk.CTkLabel(
                card, text=f"{icon} {name}", font=("Segoe UI", 12, "bold"),
                text_color=COLOR_TEXT,
            ).pack(side="left", padx=10, pady=6)
            ctk.CTkLabel(
                card, text=desc, font=("Segoe UI", 10),
                text_color=COLOR_MUTED,
            ).pack(side="right", padx=10)

        self.settings = SettingsPanel(left_panel, self)
        self.settings.pack(fill="both", expand=True, pady=(12, 0))

        main_area = ctk.CTkFrame(self, fg_color=COLOR_BG, corner_radius=0)
        main_area.grid(row=0, column=1, sticky="nsew")
        main_area.grid_rowconfigure(0, weight=1)
        main_area.grid_columnconfigure(0, weight=1)

        self.chat_canvas = tk.Canvas(
            main_area, bg=COLOR_BG, highlightthickness=0, bd=0,
        )
        self.chat_canvas.grid(row=0, column=0, sticky="nsew")

        scrollbar = ctk.CTkScrollbar(main_area, command=self.chat_canvas.yview)
        scrollbar.grid(row=0, column=1, sticky="ns")

        self.chat_container = ctk.CTkFrame(self.chat_canvas, fg_color=COLOR_BG)
        self.chat_window = self.chat_canvas.create_window(
            (0, 0), window=self.chat_container, anchor="nw",
        )

        self.chat_canvas.configure(yscrollcommand=scrollbar.set)
        self.chat_container.bind("<Configure>", self._on_frame_configure)
        self.chat_canvas.bind("<Configure>", self._on_canvas_configure)
        self.chat_canvas.bind("<MouseWheel>",
            lambda e: self.chat_canvas.yview_scroll(int(-1 * (e.delta / 120)), "units"))

        input_frame = ctk.CTkFrame(self, fg_color=COLOR_INPUT, corner_radius=0, height=INPUT_HEIGHT)
        input_frame.grid(row=1, column=1, sticky="ew")
        input_frame.grid_columnconfigure(0, weight=1)
        input_frame.grid_propagate(False)

        input_wrap = ctk.CTkFrame(input_frame, fg_color="transparent")
        input_wrap.pack(fill="x", padx=20, pady=(10, 10))

        self.input_text = ctk.CTkTextbox(
            input_wrap, height=40, wrap="word", font=FONT,
            border_width=1, border_color=COLOR_BORDER,
            fg_color="#0d1322", text_color=COLOR_TEXT,
            corner_radius=12,
        )
        self.input_text.pack(side="left", fill="x", expand=True, padx=(0, 10))
        self.input_text.bind("<Return>", self._on_enter)
        self.input_text.bind("<Shift-Return>", lambda e: self.input_text.insert("insert", "\n"))

        self.send_btn = ctk.CTkButton(
            input_wrap, text="\u27a4", width=48, height=40,
            font=("Segoe UI", 18), corner_radius=12,
            fg_color=COLOR_GREEN, hover_color="#16a34a",
            text_color="#000000",
            command=self._send_message,
        )
        self.send_btn.pack(side="right")

        self.status_bar = StatusBar(self)
        self.status_bar.grid(row=2, column=1, sticky="ew")

        self.progress = ctk.CTkProgressBar(input_frame, height=2, mode="indeterminate",
                                            progress_color=COLOR_GREEN)
        self.progress.place(relx=0, rely=0, relwidth=1)

    def _on_frame_configure(self, event):
        self.chat_canvas.configure(scrollregion=self.chat_canvas.bbox("all"))

    def _on_canvas_configure(self, event):
        self.chat_canvas.itemconfig(self.chat_window, width=event.width)

    def _on_enter(self, event):
        if not (event.state & 0x1):
            self._send_message()
            return "break"

    def _bind_shortcuts(self):
        self.bind("<Control-w>", lambda e: self.destroy())
        self.bind("<Control-q>", lambda e: self.destroy())
        self.bind("<Control-l>", lambda e: self._clear_chat())

    def _show_welcome(self):
        welcome = (
            "Привет! Я GAI — твой ИИ-агент с доступом в интернет.\n\n"
            "Я умею:\n"
            "  \U0001f50d  Искать информацию в интернете\n"
            "  \u2328  Писать и выполнять Python код\n"
            "  \U0001f4c1  Работать с файлами\n\n"
            "**Чтобы начать — подключи API ключ в настройках слева.**\n\n"
            "Поддерживаются любые OpenAI-совместимые API:\n"
            "  - OpenAI (GPT-4, GPT-4o-mini)\n"
            "  - Ollama (локально, через /v1/chat/completions)\n"
            "  - Groq, Together AI, и другие\n\n"
            "Просто напиши мне что-нибудь!"
        )
        self._add_message(welcome, "assistant")

    def _add_message(self, text, role="user", tool_name=None):
        msg = MessageBubble(self.chat_container, text, role, tool_name)
        self.after(50, self._scroll_to_bottom)
        self.message_history.append({"role": role, "content": text, "tool_name": tool_name})

    def _scroll_to_bottom(self):
        self.chat_canvas.yview_moveto(1.0)

    def _clear_chat(self):
        for w in self.chat_container.winfo_children():
            w.destroy()
        self.message_history = []

    def _set_loading(self, loading):
        if loading:
            self.send_btn.configure(state="disabled", text="...")
            self.progress.grid()
            self.progress.start()
            self.status_bar.set_working()
        else:
            self.send_btn.configure(state="normal", text="\u27a4")
            self.progress.stop()
            self.progress.grid_remove()
            self.status_bar.set_ready()

    def on_connected(self, api_key, model, base_url):
        self.llm = LLMClient(api_key=api_key, model=model, base_url=base_url)
        self.connected = True
        self.status_bar.set_connected(model)
        self._add_message(
            f"\u2713 Подключено к **{model}**! Теперь я могу общаться и использовать инструменты. "
            "Просто напиши мне что-нибудь.",
            "assistant"
        )

    def _send_message(self):
        text = self.input_text.get("0.0", "end").strip()
        if not text or self.running:
            return

        self.input_text.delete("0.0", "end")
        self._add_message(text, "user")
        self._set_loading(True)
        self.running = True

        threading.Thread(target=self._process, args=(text,), daemon=True).start()

    def _process(self, text):
        try:
            if self.llm and self.connected:
                response = self.llm.chat(text, tools_enabled=True)
                self.after(0, lambda: self._add_message(response, "assistant"))
            else:
                self._local_response(text)
        except Exception as e:
            self.after(0, lambda: self._add_message(
                f"\u274c Ошибка: {e}\n\nПроверь API ключ или endpoint в настройках.",
                "assistant"
            ))
        finally:
            self.running = False
            self.after(0, lambda: self._set_loading(False))

    def _local_response(self, text):
        import time
        time.sleep(0.5)

        text_lower = text.lower()

        if any(w in text_lower for w in ["привет", "здравствуй", "хай", "hello", "hi"]):
            resp = "Привет! Чем могу помочь?"
        elif any(w in text_lower for w in ["спасиб", "благодар"]):
            resp = "Всегда пожалуйста! Обращайся."
        elif any(w in text_lower for w in ["кто ты", "что ты", "gai"]):
            resp = (
                "Я GAI — экспериментальный ИИ-агент.\n\n"
                "Мои возможности:\n"
                "  \U0001f50d  Поиск в интернете\n"
                "  \u2328  Выполнение Python кода\n"
                "  \U0001f4c1  Работа с файлами\n\n"
                "**Но чтобы я мог реально общаться — подключи API ключ в настройках!**\n"
                "Тогда я буду использовать LLM для осмысленных диалогов."
            )
        elif any(w in text_lower for w in ["что можешь", "помощь", "help", "команд"]):
            resp = (
                "Вот что я умею:\n\n"
                "  \U0001f50d  Найти информацию в интернете\n"
                "  \u2328  Написать и выполнить Python код\n"
                "  \U0001f4c4  Прочитать файл\n"
                "  \U0001f4dd  Записать файл\n"
                "  \U0001f4c1  Показать список файлов\n\n"
                "Подключи API ключ — и я буду делать это сам, понимая естественный язык!"
            )
        else:
            resp = (
                f"Я тебя услышал! Чтобы я мог полноценно ответить, "
                f"подключи API ключ в левой панели.\n\n"
                f"**Сейчас доступен режим команд:**\n"
                f"просто опиши, что нужно сделать, а я постараюсь понять.\n\n"
                f"Или напиши «помощь», чтобы увидеть все возможности."
            )

        self.after(0, lambda: self._add_message(resp, "assistant"))


def run_ui():
    app = GAIApp()
    app.mainloop()
