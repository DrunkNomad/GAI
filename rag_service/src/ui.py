import gradio as gr
import httpx
from src.core.config import settings


API_URL = f"http://{settings.host}:{settings.port}"


def chat(message: str, history: list, use_rag: bool) -> tuple[str, list]:
    try:
        with httpx.Client(base_url=API_URL, timeout=30) as client:
            resp = client.post("/api/v1/chat", json={"question": message, "use_rag": use_rag})
            resp.raise_for_status()
            data = resp.json()
            answer = data["answer"]

            sources = data.get("sources", [])
            if sources:
                answer += "\n\n**Sources:**"
                for s in sources:
                    answer += f"\n> {s['text'][:150]}... *(score: {s['score']:.2f})*"

            history.append((message, answer))
            return "", history

    except Exception as e:
        error_msg = f"Error: {e}"
        history.append((message, error_msg))
        return "", history


def toggle_rag(use_rag: bool) -> str:
    return f"RAG is {'enabled' if use_rag else 'disabled'}"


with gr.Blocks(title="GAI RAG Chat", theme=gr.themes.Soft()) as demo:
    gr.Markdown(
        """
    # GAI RAG Chat
    Chat with a custom GPT model augmented with document retrieval.
    """
    )

    chatbot = gr.Chatbot(label="Conversation")
    msg = gr.Textbox(label="Your question", placeholder="Ask something...")
    with gr.Row():
        use_rag = gr.Checkbox(label="Use RAG", value=True)
        status = gr.Markdown("RAG is enabled")
    clear = gr.Button("Clear")

    msg.submit(chat, [msg, chatbot, use_rag], [msg, chatbot])
    use_rag.change(toggle_rag, use_rag, status)
    clear.click(lambda: [], None, chatbot)


if __name__ == "__main__":
    demo.launch(server_name=settings.host, server_port=7860)
