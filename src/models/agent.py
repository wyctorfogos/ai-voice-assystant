import requests


class OllamaClient:
    def __init__(
        self,
        model: str = "qwen3:0.6b",
        base_url: str = "http://localhost:11434"
    ):
        self.model = model
        self.url = f"{base_url}/api/chat"

        self.system_prompt = (
            "Você é um assistente educado, técnico e claro. "
            "Responda sempre em português brasileiro."
        )

    def chat(self, messages) -> str:
        payload = {
            "model": self.model,
            "messages": [
                {"role": "system", "content": self.system_prompt},
                *messages
            ],
            "stream": False
        }

        response = requests.post(self.url, json=payload, timeout=120)
        response.raise_for_status()

        return response.json()["message"]["content"]
