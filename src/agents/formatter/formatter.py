"""
Handles the formatting and cleaning of raw text data.

This module defines the `Formatter` class, which is responsible for taking raw text,
typically extracted from web pages or other sources, and transforming it into a
cleaner, more readable Markdown format. It uses a Large Language Model (LLM)
and a Jinja2 template to guide the formatting process.
"""
import re
from typing import Optional

from jinja2 import Environment, BaseLoader

from src.llm import LLM
from src.logger import Logger

# Load the prompt template from the associated Jinja2 file.
try:
    with open("src/agents/formatter/prompt.jinja2", "r", encoding="utf-8") as f:
        PROMPT_TEMPLATE = f.read().strip()
except FileNotFoundError:
    PROMPT_TEMPLATE = "Error: Formatter prompt template not found."
    # Consider logging this error as well if a logger is available at module level.

logger = Logger()


class Formatter:
    """
    The Formatter agent class.

    This agent takes raw text data, renders it into a structured prompt for an LLM,
    and then processes the LLM's response to extract the formatted Markdown content.

    Attributes:
        llm (LLM): An instance of the Large Language Model client.
    """

    def __init__(self, base_model: str) -> None:
        """
        Initialize the Formatter agent.

        Args:
            base_model (str): The identifier of the base LLM model to be used.
        """
        self.llm: LLM = LLM(model_id=base_model)
        if PROMPT_TEMPLATE == "Error: Formatter prompt template not found.":
            logger.error(
                "Formatter initialized with a missing prompt template. "
                "Functionality will be impaired."
            )

    def render(self, raw_text: str) -> str:
        """
        Render the raw text into the predefined Jinja2 template.

        Args:
            raw_text (str): The raw text content to be formatted.

        Returns:
            str: The fully rendered prompt string to be sent to the LLM.
        """
        if PROMPT_TEMPLATE == "Error: Formatter prompt template not found.":
            return f"Formatter prompt template is missing. Raw text: {raw_text[:200]}..." # Log snippet
        env = Environment(loader=BaseLoader())
        template = env.from_string(PROMPT_TEMPLATE)
        return template.render(raw_text=raw_text)

    def validate_response(self, response: str) -> bool:
        """
        Validate the LLM's response.

        For the Formatter, a valid response is simply a non-empty string,
        as it's expected to be formatted Markdown text. It also checks if the
        response is enclosed in markdown-style triple backticks.

        Args:
            response (str): The raw string response from the LLM.

        Returns:
            bool: True if the response is considered valid, False otherwise.
        """
        if not response or not response.strip():
            logger.warning("Formatter response validation failed: Response is empty.")
            return False

        # Check if the response is enclosed in ```markdown ... ``` or just ``` ... ```
        if not (
            response.strip().startswith("```markdown") or response.strip().startswith("```")
        ) or not response.strip().endswith("```"):
            logger.warning(
                "Formatter response validation failed: Response not enclosed in markdown code blocks."
            )
            return False
        return True

    def parse_response(self, response: str) -> str:
        """
        Parse the LLM's response to extract the formatted Markdown content.

        It attempts to extract content from a Markdown code block.
        If no block is found, it returns the stripped response as a fallback.

        Args:
            response (str): The raw string response from the LLM.

        Returns:
            str: The extracted (or raw) formatted text.
        """
        # Regex to find content within ```markdown ... ``` or ``` ... ```
        match = re.search(r"```(?:markdown\s*)?(.*?)```", response, re.DOTALL)
        if match:
            return match.group(1).strip()
        else:
            logger.warning(
                "No markdown block found in formatter response. Returning stripped raw response."
            )
            return response.strip()

    def execute(self, raw_text: str, project_name: Optional[str] = None) -> str:
        """
        Execute the formatting process.

        This involves rendering the raw text into a prompt, sending it to the LLM,
        validating the response, and then parsing it to extract the formatted content.

        Args:
            raw_text (str): The raw text content to be formatted.
            project_name (Optional[str]): The name of the project, used for LLM context/logging.
                                       Defaults to "formatter_task".

        Returns:
            str: The formatted text as a Markdown string. Returns the original raw_text
                 if the prompt template is missing or if the LLM response is invalid
                 or cannot be parsed meaningfully (though parse_response has a fallback).
        """
        rendered_prompt = self.render(raw_text)
        if "Formatter prompt template is missing" in rendered_prompt:
            logger.error("Cannot execute formatter due to missing prompt template.")
            return raw_text # Fallback to raw_text

        llm_response = self.llm.inference(rendered_prompt, project_name or "formatter_task")

        if not self.validate_response(llm_response):
            logger.warning(
                "LLM response failed validation for formatter. Returning raw text as fallback."
            )
            return raw_text # Fallback to raw_text

        return self.parse_response(llm_response)
