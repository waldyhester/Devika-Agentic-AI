"""
Handles the planning phase of the AI agent's thought process.

This module defines the `Planner` class, which is responsible for taking a user's
prompt and generating a structured, step-by-step plan for the AI to follow.
It uses a Large Language Model (LLM) and a Jinja2 template to formulate the plan.
The plan is expected to be in JSON format as specified by the prompt.
"""

import json
import re
from typing import List, Optional, TypedDict, Union

from jinja2 import BaseLoader, Environment

from src.llm import LLM
from src.logger import Logger

# Load the prompt template from the associated Jinja2 file.
# It's good practice to load file resources once when the module is imported.
try:
    with open("src/agents/planner/prompt.jinja2", "r", encoding="utf-8") as f:
        PROMPT_TEMPLATE = f.read().strip()
except FileNotFoundError:
    PROMPT_TEMPLATE = "Error: Planner prompt template not found."
    # Consider logging this error as well if a logger is available at module level
    # or raising an exception if the application cannot function without it.

logger = Logger()


class PlanStep(TypedDict):
    """
    Represents a single step in the generated plan.

    Attributes:
        step (Union[int, str]): The step number (e.g., 1, 2, "N").
        description (str): The description of the action to be taken for this step.
    """

    step: Union[int, str]
    description: str


class PlannerResponseDict(TypedDict):
    """
    Represents the structured dictionary expected from the LLM's JSON response.

    Attributes:
        project_name (str): The name of the project.
        reply (str): A human-like reply to the user's prompt.
        focus (str): The main objective or focus area of the plan.
        plan (List[PlanStep]): A list of steps, each a `PlanStep` dictionary.
        summary (str): A summary of the plan, including considerations or challenges.
    """

    project_name: str
    reply: str
    focus: str
    plan: List[PlanStep]
    summary: str


class Planner:
    """
    The Planner agent class.

    This agent takes a user prompt, renders it into a structured prompt for an LLM,
    and then processes the LLM's response to extract a formal plan.
    The plan is expected to be in JSON format.

    Attributes:
        llm (LLM): An instance of the Large Language Model client.
    """

    def __init__(self, base_model: str) -> None:
        """
        Initialize the Planner agent.

        Args:
            base_model (str): The identifier of the base LLM model to be used.
        """
        self.llm: LLM = LLM(model_id=base_model)
        if PROMPT_TEMPLATE == "Error: Planner prompt template not found.":
            logger.error(
                "Planner initialized with a missing prompt template. "
                "Functionality will be impaired."
            )

    def render(self, prompt: str) -> str:
        """
        Render the user prompt into the predefined Jinja2 template.

        Args:
            prompt (str): The user's input prompt.

        Returns:
            str: The fully rendered prompt string to be sent to the LLM.
        """
        if PROMPT_TEMPLATE == "Error: Planner prompt template not found.":
            # Return a simple message or raise an error if template is critical
            return f"Planner prompt template is missing. User prompt: {prompt}"
        env = Environment(loader=BaseLoader())
        template = env.from_string(PROMPT_TEMPLATE)
        return template.render(prompt=prompt)

    def validate_response(self, response: str) -> bool:
        """
        Validate if the LLM's response is valid JSON.

        Currently, this method checks if the response string can be parsed as JSON.
        It looks for a JSON code block if present.

        Args:
            response (str): The raw string response from the LLM.

        Returns:
            bool: True if the response is valid JSON, False otherwise.
        """
        # Attempt to extract JSON from a code block if present
        match = re.search(r"```json\s*(\{.*?\})\s*```", response, re.DOTALL)
        json_string = response
        if match:
            json_string = match.group(1)

        try:
            json.loads(json_string)
            return True
        except json.JSONDecodeError:
            logger.warning(
                f"Planner response validation failed: Not valid JSON. Response: {response[:500]}..."
            )  # Log first 500 chars
            return False

    def parse_response(self, response: str) -> Optional[PlannerResponseDict]:
        """
        Parse the LLM's JSON response string into a structured dictionary.

        Args:
            response (str): The raw string response from the LLM, expected to be
                            a JSON object, possibly within a code block.

        Returns:
            Optional[PlannerResponseDict]: A dictionary containing the structured plan
                                           if parsing is successful, otherwise None.
                                           The structure includes:
                                           - "project_name": str
                                           - "reply": str
                                           - "focus": str
                                           - "plan": List[Dict[str, Union[int, str]]]
                                             (e.g., [{"step": 1, "description": "..."}])
                                           - "summary": str
        """
        # Attempt to extract JSON from a code block if present
        match = re.search(r"```json\s*(\{.*?\})\s*```", response, re.DOTALL)
        json_string = response
        if match:
            json_string = match.group(1)
        else:
            # If no code block, assume the whole response is JSON or attempt to find JSON structure
            # This might be risky if the LLM adds explanations outside the JSON
            # For now, we'll try parsing the whole string if no block is found.
            # A more robust solution might involve cleaning up common LLM preambles/postambles.
            logger.info(
                "No JSON code block found in planner response, attempting to parse entire response."
            )

        try:
            parsed_json: PlannerResponseDict = json.loads(json_string)
            # Basic validation of expected keys
            required_keys = ["project_name", "reply", "focus", "plan", "summary"]
            if not all(key in parsed_json for key in required_keys):
                logger.error(
                    f"Planner response JSON missing required keys. Found: {parsed_json.keys()}. Response: {json_string[:500]}..."
                )
                return None
            if not isinstance(parsed_json["plan"], list):
                logger.error(
                    f"Planner response 'plan' field is not a list. Response: {json_string[:500]}..."
                )
                return None
            return parsed_json
        except json.JSONDecodeError as e:
            logger.error(
                f"Failed to parse planner response JSON: {e}. Response: {json_string[:500]}..."
            )
            return None

    def execute(
        self, prompt: str, project_name: Optional[str] = None
    ) -> Optional[PlannerResponseDict]:
        """
        Execute the planning process.

        This involves rendering the prompt, sending it to the LLM, validating,
        and parsing the response.

        Args:
            prompt (str): The user's input prompt.
            project_name (Optional[str]): The name of the project, used for LLM context/logging.
                                       Defaults to None.

        Returns:
            Optional[PlannerResponseDict]: A dictionary containing the structured plan
                                           if successful, otherwise None.
        """
        rendered_prompt = self.render(prompt)
        if "Planner prompt template is missing" in rendered_prompt:
            logger.error("Cannot execute planner due to missing prompt template.")
            return None

        llm_response = self.llm.inference(
            rendered_prompt, project_name or "planner_task"
        )

        if not self.validate_response(llm_response):
            # Attempt to re-run or handle error, for now, just log and return None
            logger.error(
                "LLM response failed validation. Cannot proceed with planning."
            )
            # You could try a retry mechanism here or a fallback.
            return None

        return self.parse_response(llm_response)
