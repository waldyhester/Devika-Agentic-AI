"""
Handles the generation of project reports.

This module defines the `Reporter` class, which is responsible for creating a
comprehensive project report in Markdown format based on the conversation history
and the existing codebase. It uses a Large Language Model (LLM) and a Jinja2
template to guide the report generation process. The LLM is instructed to output
raw Markdown.
"""
import re
from typing import List, Optional

from jinja2 import Environment, BaseLoader

from src.llm import LLM
from src.logger import Logger

# Load the prompt template.
try:
    with open("src/agents/reporter/prompt.jinja2", "r", encoding="utf-8") as f:
        PROMPT_TEMPLATE = f.read().strip()
except FileNotFoundError:
    PROMPT_TEMPLATE = "Error: Reporter prompt template not found."

logger = Logger()


class Reporter:
    """
    The Reporter agent class.

    This agent generates a detailed project report in Markdown format.
    It takes the conversation history and code context, renders them into a
    structured prompt for an LLM, and then processes the LLM's response.
    The LLM is expected to return raw Markdown text.

    Attributes:
        llm (LLM): An instance of the Large Language Model client.
    """

    def __init__(self, base_model: str) -> None:
        """
        Initialize the Reporter agent.

        Args:
            base_model (str): The identifier of the base LLM model to be used.
        """
        self.llm: LLM = LLM(model_id=base_model)
        if PROMPT_TEMPLATE == "Error: Reporter prompt template not found.":
            logger.error(
                "Reporter initialized with a missing prompt template. "
                "Functionality will be impaired."
            )

    def render(self, conversation: List[str], code_markdown: str) -> str:
        """
        Render the conversation and code context into the predefined Jinja2 template.

        Args:
            conversation (List[str]): The history of messages in the conversation.
            code_markdown (str): Markdown formatted string of the project's code.

        Returns:
            str: The fully rendered prompt string to be sent to the LLM.
        """
        if PROMPT_TEMPLATE == "Error: Reporter prompt template not found.":
            return f"Reporter prompt template is missing. Conversation: {conversation}"
        env = Environment(loader=BaseLoader())
        template = env.from_string(PROMPT_TEMPLATE)
        return template.render(conversation=conversation, code_markdown=code_markdown)

    def parse_response(self, response: str) -> str:
        """
        Parse the LLM's response to extract the Markdown report.

        Since the prompt asks for raw Markdown without enclosing backticks,
        this method primarily strips leading/trailing whitespace.
        It also includes a check for accidental markdown code blocks,
        attempting to extract content from them if found, as LLMs sometimes
        still wrap their output.

        Args:
            response (str): The raw string response from the LLM.

        Returns:
            str: The extracted Markdown report string.
        """
        # Check if the LLM accidentally wrapped the output in markdown code blocks
        # Common patterns: ```markdown ... ``` or ``` ... ```
        match = re.search(r"```(?:markdown\s*)?(.*?)```", response, re.DOTALL)
        if match:
            logger.info("Markdown code block found in reporter response, extracting content.")
            return match.group(1).strip()
        
        # If no code block, assume the entire response is the intended raw Markdown.
        return response.strip()

    def execute(
        self,
        conversation: List[str],
        code_markdown: str,
        project_name: Optional[str] = None,
    ) -> Optional[str]:
        """
        Execute the report generation process.

        This involves rendering the context into a prompt, sending it to the LLM,
        and then parsing the response to get the Markdown report.
        The recursive retry logic has been removed.

        Args:
            conversation (List[str]): The history of messages in the conversation.
            code_markdown (str): Markdown formatted string of the project's code.
            project_name (Optional[str]): The name of the project, used for LLM
                                       context/logging. Defaults to "reporter_task".

        Returns:
            Optional[str]: The generated Markdown report string if successful,
                           otherwise None.
        """
        if PROMPT_TEMPLATE == "Error: Reporter prompt template not found.":
            logger.error("Cannot execute reporter due to missing prompt template.")
            return None

        rendered_prompt = self.render(conversation, code_markdown)
        llm_response_str = self.llm.inference(
            rendered_prompt, project_name or "reporter_task"
        )

        if not llm_response_str or not llm_response_str.strip():
            logger.error("LLM returned an empty or whitespace-only response for reporter.")
            return None

        # The validate_response was essentially parsing, so combined.
        parsed_report = self.parse_response(llm_response_str)

        if not parsed_report: # Should not happen if parse_response returns stripped original
            logger.error("Failed to get a valid report from LLM after parsing.")
            return None

        logger.info(f"Generated Report (first 100 chars): {parsed_report[:100]}...")
        return parsed_report
