import ollama

from src.config import Config
from src.logger import Logger

log = Logger()


class Ollama:
    def __init__(self):
        try:
            self.client = ollama.Client(Config().get_ollama_api_endpoint())
            self.models = self.client.list()["models"]
            log.info("Ollama available")
        except Exception as e:
            self.client = None
            log.warning(f"Ollama not available: {e}")
            log.warning(
                "run ollama server to use ollama models otherwise use other models"
            )

    def inference(self, model_id: str, prompt: str) -> str:
        response = self.client.generate(model=model_id, prompt=prompt.strip())
        return response["response"]
