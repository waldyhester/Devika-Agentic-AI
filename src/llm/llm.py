"""
Provides an abstraction layer for interacting with various Large Language Models (LLMs).

This module defines the `LLM` class, which serves as a unified interface for
different LLM clients such as Ollama, Claude, OpenAI, Gemini, Mistral, and Groq.
It handles model selection, prompt logging, and token usage tracking.

Global instances for Ollama, Logger, and AgentState are initialized at the module
level for use by the LLM class and its methods.
"""

import tiktoken
from typing import List, Tuple, Dict, Optional

from src.socket_instance import emit_agent
from .ollama_client import Ollama
from .claude_client import Claude
from .openai_client import OpenAi
from .gemini_client import Gemini
from .mistral_client import MistralAi
from .groq_client import Groq
from .base_client import BaseClient

from src.state import AgentState
from src.config import Config
from src.logger import Logger

# Initialize Tiktoken encoding for token counting
TIKTOKEN_ENC = tiktoken.get_encoding("cl100k_base")

# Global instances, primarily used by the LLM class.
# These are initialized here to be shared across LLM instances if needed,
# or to manage a global state for these services.
ollama_client = Ollama()  # Global Ollama client instance
logger = Logger()  # Global logger instance
agent_state = AgentState()  # Global agent state instance for token tracking


class LLM:
    """
    A class to interact with different Large Language Models (LLMs) through a unified interface.

    This class manages various LLM clients and provides methods for listing available models,
    mapping model IDs to their respective client types, updating token usage, and performing
    inference.

    Attributes:
        model_id (Optional[str]): The identifier of the currently selected model.
        log_prompts (bool): Flag to enable or disable logging of prompts and responses.
        models (Dict[str, List[Tuple[str, str]]]): A dictionary storing available models,
            categorized by LLM provider. Each model is a tuple of (display_name, model_id).
    """

    def __init__(self, model_id: Optional[str] = None) -> None:
        """
        Initialize the LLM class.

        Args:
            model_id (Optional[str]): The identifier of the model to be used for inference.
                                      Defaults to None.
        """
        self.model_id: Optional[str] = model_id
        self.log_prompts: bool = Config().get_logging_prompts()
        self.models: Dict[str, List[Tuple[str, str]]] = {
            "CLAUDE": [
                ("Claude 3 Opus", "claude-3-opus-20240229"),
                ("Claude 3 Sonnet", "claude-3-sonnet-20240229"),
                ("Claude 3 Haiku", "claude-3-haiku-20240307"),
            ],
            "OPENAI": [
                ("GPT-4 Turbo", "gpt-4-0125-preview"),
                ("GPT-3.5", "gpt-3.5-turbo-0125"),
            ],
            "GOOGLE": [
                ("Gemini 1.0 Pro", "gemini-pro"),
            ],
            "MISTRAL": [
                ("Mistral 7b", "open-mistral-7b"),
                ("Mistral 8x7b", "open-mixtral-8x7b"),
                ("Mistral Medium", "mistral-medium-latest"),
                ("Mistral Small", "mistral-small-latest"),
                ("Mistral Large", "mistral-large-latest"),
            ],
            "GROQ": [
                ("GROQ Mixtral", "mixtral-8x7b-32768"),
                ("GROQ LLAMA2 70B", "llama2-70b-4096"),
                ("GROQ GEMMA 7B IT", "gemma-7b-it"),
            ],
            "OLLAMA": [],
        }
        if (
            ollama_client.client
        ):  # Check if the Ollama client was successfully initialized
            self.models["OLLAMA"] = [
                (model["name"].split(":")[0], model["name"])
                for model in ollama_client.models
            ]

    def list_models(self) -> Dict[str, List[Tuple[str, str]]]:
        """
        List all available models categorized by LLM provider.

        Returns:
            Dict[str, List[Tuple[str, str]]]: A dictionary where keys are provider names
                                              (e.g., "OPENAI", "OLLAMA") and values are lists
                                              of (display_name, model_id) tuples.
        """
        return self.models

    def model_id_to_enum_mapping(self) -> Dict[str, str]:
        """
        Create a mapping from model ID to its LLM provider category (enum name).

        This is useful for determining which client to use for a given model ID.

        Returns:
            Dict[str, str]: A dictionary where keys are model_ids and values are
                            their corresponding provider enum names (e.g., "OPENAI").
        """
        mapping: Dict[str, str] = {}
        for enum_name, model_list in self.models.items():
            for _, model_id_val in model_list:  # model_name is not used here
                mapping[model_id_val] = enum_name
        return mapping

    @staticmethod
    def update_global_token_usage(text_content: str, project_name: str) -> None:
        """
        Update the global token usage count for a given project.

        This method calculates the number of tokens in the provided text and updates
        the token count in the AgentState. It also emits a socket event with the
        new total token usage.

        Args:
            text_content (str): The text content for which to count tokens.
            project_name (str): The name of the project to associate the token usage with.
        """
        token_usage: int = len(TIKTOKEN_ENC.encode(text_content))
        agent_state.update_token_usage(project_name, token_usage)

        # Get the latest cumulative token usage for the project
        total_tokens: int = agent_state.get_latest_token_usage(project_name)
        # Note: The previous implementation added token_usage again.
        # Assuming get_latest_token_usage already includes the recent update.
        # If not, the previous logic was:
        # total = agent_state.get_latest_token_usage(project_name) + token_usage
        # Clarified: update_token_usage in AgentState is cumulative, so get_latest is correct.
        emit_agent("tokens", {"token_usage": total_tokens})

    def inference(self, prompt: str, project_name: str) -> str:
        """
        Perform inference using the selected LLM.

        This method sends the prompt to the LLM specified by `self.model_id`,
        updates token usage for both prompt and response, and logs the interaction
        if `self.log_prompts` is enabled.

        Args:
            prompt (str): The prompt to send to the LLM.
            project_name (str): The name of the project for token tracking.

        Returns:
            str: The response text from the LLM, stripped of leading/trailing whitespace.

        Raises:
            ValueError: If the `self.model_id` is not set or if the model provider
                        (enum) is not supported or found in the mapping.
        """
        if not self.model_id:
            raise ValueError("Model ID is not set. Please select a model.")

        self.update_global_token_usage(prompt, project_name)

        model_provider_enum: Optional[str] = self.model_id_to_enum_mapping().get(
            self.model_id
        )

        logger.info(f"Using LLM: {self.model_id} via {model_provider_enum}")

        if model_provider_enum is None:
            raise ValueError(f"Model provider for {self.model_id} not found.")

        # Type hint for the model_mapping dictionary
        model_mapping: Dict[str, BaseClient] = {
            "OLLAMA": ollama_client,
            "CLAUDE": Claude(),
            "OPENAI": OpenAi(),
            "GOOGLE": Gemini(),
            "MISTRAL": MistralAi(),
            "GROQ": Groq(),
        }

        try:
            selected_client: BaseClient = model_mapping[model_provider_enum]
            response: str = selected_client.inference(self.model_id, prompt).strip()
        except KeyError:
            # This case should ideally be caught by the model_provider_enum check earlier,
            # but kept for robustness.
            raise ValueError(
                f"Model provider {model_provider_enum} is not supported in model_mapping."
            )
        except Exception as e:
            logger.error(
                f"Error during LLM inference with {self.model_id} ({model_provider_enum}): {e}"
            )
            raise  # Re-raise the exception after logging

        if self.log_prompts:
            logger.debug(f"LLM Prompt ({self.model_id}): --> {prompt}")
            logger.debug(f"LLM Response ({self.model_id}): --> {response}")

        self.update_global_token_usage(response, project_name)

        return response
