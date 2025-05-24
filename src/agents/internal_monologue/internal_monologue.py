"""
Handles the generation of the AI agent's internal monologue.

This module defines the `InternalMonologue` class, responsible for creating
a brief, human-like internal thought process based on the current prompt or
context. It uses a Large Language Model (LLM) and a Jinja2 template, expecting
the LLM to return a JSON object containing the monologue string.
"""
import json
import re
from typing import TypedDict, Optional

from jinja2 import Environment, BaseLoader

from src.llm import LLM
from src.logger import Logger

# Load the prompt template.
try:
    with open(
        "src/agents/internal_monologue/prompt.jinja2", "r", encoding="utf-8"
    ) as f:
        PROMPT_TEMPLATE = f.read().strip()
except FileNotFoundError:
    PROMPT_TEMPLATE = "Error: Internal Monologue prompt template not found."

logger = Logger()


class InternalMonologueResponseDict(TypedDict):
    """
    Represents the structured dictionary expected from the LLM's JSON response.

    Attributes:
        internal_monologue (str): The agent's internal monologue text.
    """

    internal_monologue: str


class InternalMonologue:
    """
    The InternalMonologue agent class.

    This agent generates a short, human-like internal monologue based on the
    current prompt or context provided to the LLM. The response is expected
    to be a JSON object containing the monologue.

    Attributes:
        llm (LLM): An instance of the Large Language Model client.
    """

    def __init__(self, base_model: str) -> None:
        """
        Initialize the InternalMonologue agent.

        Args:
            base_model (str): The identifier of the base LLM model to be used.
        """
        self.llm: LLM = LLM(model_id=base_model)
        if PROMPT_TEMPLATE == "Error: Internal Monologue prompt template not found.":
            logger.error(
                "InternalMonologue initialized with a missing prompt template. "
                "Functionality will be impaired."
            )

    def render(self, current_prompt: str) -> str:
        """
        Render the current prompt into the predefined Jinja2 template.

        Args:
            current_prompt (str): The current prompt or context for the agent.

        Returns:
            str: The fully rendered prompt string to be sent to the LLM.
        """
        if PROMPT_TEMPLATE == "Error: Internal Monologue prompt template not found.":
            return f"Internal Monologue prompt template is missing. Prompt: {current_prompt}"
        env = Environment(loader=BaseLoader())
        template = env.from_string(PROMPT_TEMPLATE)
        return template.render(current_prompt=current_prompt)

    def parse_and_validate_response(
        self, llm_response: str
    ) -> Optional[InternalMonologueResponseDict]:
        """
        Parse and validate the LLM's JSON response.

        The method attempts to extract JSON from a markdown code block if present.
        It then validates if the parsed JSON contains the "internal_monologue" key
        and if its value is a non-empty string.

        Args:
            llm_response (str): The raw string response from the LLM.

        Returns:
            Optional[InternalMonologueResponseDict]: The parsed and validated response
                                                     as a dictionary, or None if
                                                     parsing or validation fails.
        """
        json_string = llm_response
        match = re.search(r"```json\s*(\{.*?\})\s*```", llm_response, re.DOTALL)
        if match:
            json_string = match.group(1)
        else:
            logger.info("No JSON code block found in internal_monologue response, attempting to parse entire response.")


        try:
            parsed_json: InternalMonologueResponseDict = json.loads(json_string)
        except json.JSONDecodeError as e:
            logger.error(
                f"Failed to parse internal_monologue JSON: {e}. Response: {json_string[:500]}..."
            )
            return None

        if not isinstance(parsed_json, dict):
            logger.error(f"Internal_monologue response is not a JSON object: {parsed_json}")
            return None

        monologue = parsed_json.get("internal_monologue")

        if not isinstance(monologue, str) or not monologue.strip():
            logger.error(
                "'internal_monologue' field is missing, not a string, or empty. "
                f"Found: '{monologue}'. Response: {json_string[:500]}..."
            )
            return None

        return {"internal_monologue": monologue}

    def execute(
        self, current_prompt: str, project_name: Optional[str] = None
    ) -> Optional[str]:
        """
        Execute the internal monologue generation process.

        This involves rendering the current prompt, sending it to the LLM,
        and then parsing and validating the JSON response to extract the monologue.
        The recursive retry logic has been removed.

        Args:
            current_prompt (str): The current prompt or context for the agent.
            project_name (Optional[str]): The name of the project, used for LLM
                                       context/logging. Defaults to "internal_monologue_task".

        Returns:
            Optional[str]: The generated internal monologue string if successful,
                           otherwise None.
        """
        if PROMPT_TEMPLATE == "Error: Internal Monologue prompt template not found.":
            logger.error(
                "Cannot execute internal monologue due to missing prompt template."
            )
            return None

        rendered_prompt = self.render(current_prompt)
        llm_response_str = self.llm.inference(
            rendered_prompt, project_name or "internal_monologue_task"
        )

        parsed_data = self.parse_and_validate_response(llm_response_str)

        if not parsed_data:
            logger.error(
                "Failed to get a valid structured internal monologue from LLM."
            )
            return None # Caller can decide how to handle this.

        logger.info(f"Generated Internal Monologue: {parsed_data['internal_monologue']}")
        return parsed_data["internal_monologue"]
