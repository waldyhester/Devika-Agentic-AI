"""
Handles the research phase of the AI agent's thought process.

This module defines the `Researcher` class, which is responsible for taking a
step-by-step plan and contextual keywords to generate search queries and potentially
formulate questions for the user to gather necessary information. It uses a
Large Language Model (LLM) and a Jinja2 template for this purpose. The output
is expected to be in a structured JSON format.
"""

import json
import re
from typing import Dict, List, Optional, TypedDict

from jinja2 import BaseLoader, Environment

from src.llm import LLM
from src.logger import Logger

# Load the prompt template from the associated Jinja2 file.
try:
    with open("src/agents/researcher/prompt.jinja2", "r", encoding="utf-8") as f:
        PROMPT_TEMPLATE = f.read().strip()
except FileNotFoundError:
    PROMPT_TEMPLATE = "Error: Researcher prompt template not found."
    # Consider logging this error as well if a logger is available at module level.

logger = Logger()


class ResearcherResponseDict(TypedDict):
    """
    Represents the structured dictionary expected from the LLM's JSON response.

    Attributes:
        queries (List[str]): A list of search queries to be executed.
        ask_user (str): A question to ask the user for more information, or an empty string.
    """

    queries: List[str]
    ask_user: str


class Researcher:
    """
    The Researcher agent class.

    This agent takes a step-by-step plan and contextual keywords, renders them
    into a structured prompt for an LLM, and then processes the LLM's JSON response
    to extract search queries and questions for the user.

    Attributes:
        llm (LLM): An instance of the Large Language Model client.
    """

    def __init__(self, base_model: str) -> None:
        """
        Initialize the Researcher agent.

        Args:
            base_model (str): The identifier of the base LLM model to be used.
        """
        self.llm: LLM = LLM(model_id=base_model)
        if PROMPT_TEMPLATE == "Error: Researcher prompt template not found.":
            logger.error(
                "Researcher initialized with a missing prompt template. "
                "Functionality will be impaired."
            )

    def render(self, step_by_step_plan: str, contextual_keywords: str) -> str:
        """
        Render the plan and keywords into the predefined Jinja2 template.

        Args:
            step_by_step_plan (str): The step-by-step plan for the current task.
            contextual_keywords (str): A string of comma-separated contextual keywords.

        Returns:
            str: The fully rendered prompt string to be sent to the LLM.
        """
        if PROMPT_TEMPLATE == "Error: Researcher prompt template not found.":
            return f"Researcher prompt template is missing. Plan: {step_by_step_plan}"
        env = Environment(loader=BaseLoader())
        template = env.from_string(PROMPT_TEMPLATE)
        return template.render(
            step_by_step_plan=step_by_step_plan,
            contextual_keywords=contextual_keywords,
        )

    def parse_and_validate_response(
        self, response: str
    ) -> Optional[ResearcherResponseDict]:
        """
        Parse and validate the LLM's JSON response.

        The method attempts to extract JSON from a markdown code block if present.
        It then validates if the parsed JSON contains the required keys ("queries", "ask_user")
        and if "queries" is a list and "ask_user" is a string.

        Args:
            response (str): The raw string response from the LLM.

        Returns:
            Optional[ResearcherResponseDict]: The parsed and validated response as a dictionary,
                                              or None if parsing or validation fails.
        """
        json_string = response
        # Attempt to extract JSON from a code block if present
        match = re.search(r"```json\s*(\{.*?\})\s*```", response, re.DOTALL)
        if match:
            json_string = match.group(1)
        else:
            logger.info(
                "No JSON code block found in researcher response, attempting to parse entire response."
            )

        try:
            parsed_json: Dict = json.loads(json_string)
        except json.JSONDecodeError as e:
            logger.error(
                f"Failed to parse researcher response JSON: {e}. Response: {json_string[:500]}..."
            )
            return None

        # Validate structure and types
        if not isinstance(parsed_json, dict):
            logger.error(f"Researcher response is not a JSON object: {parsed_json}")
            return None

        queries = parsed_json.get("queries")
        ask_user = parsed_json.get("ask_user")

        if not isinstance(queries, list) or not all(
            isinstance(q, str) for q in queries
        ):
            logger.error(
                f"'queries' field is missing, not a list, or contains non-string elements. Found: {queries}. Response: {json_string[:500]}..."
            )
            return None

        if not isinstance(ask_user, str):
            logger.error(
                f"'ask_user' field is missing or not a string. Found: {ask_user}. Response: {json_string[:500]}..."
            )
            return None

        return {"queries": queries, "ask_user": ask_user}

    def execute(
        self,
        step_by_step_plan: str,
        contextual_keywords: List[str],
        project_name: str,
    ) -> Optional[ResearcherResponseDict]:
        """
        Execute the research process.

        This involves rendering the prompt with the plan and keywords,
        sending it to the LLM, and then parsing and validating the response.
        The recursive retry logic on invalid response has been removed;
        the method now returns None if parsing/validation fails.

        Args:
            step_by_step_plan (str): The step-by-step plan.
            contextual_keywords (List[str]): A list of contextual keywords.
            project_name (str): The name of the project, used for LLM context/logging.

        Returns:
            Optional[ResearcherResponseDict]: A dictionary containing "queries" (List[str])
                                              and "ask_user" (str), or None if the process fails.
        """
        if PROMPT_TEMPLATE == "Error: Researcher prompt template not found.":
            logger.error("Cannot execute researcher due to missing prompt template.")
            return None

        contextual_keywords_str: str = ", ".join(
            k.capitalize() for k in contextual_keywords
        )
        prompt = self.render(step_by_step_plan, contextual_keywords_str)

        llm_response = self.llm.inference(prompt, project_name)

        parsed_response = self.parse_and_validate_response(llm_response)

        if not parsed_response:
            logger.error(
                "Failed to get a valid structured response from LLM for researcher."
            )
            # Caller can decide to retry or handle the None case.
            return None

        return parsed_response
