"""
Determines the next course of action based on conversation history.

This module defines the `Action` class, which is responsible for interpreting
the user's intent from the conversation and deciding on a specific action
keyword (e.g., "run", "deploy", "answer") and a corresponding textual response
for the agent to convey to the user. It uses a Large Language Model (LLM) and
a Jinja2 template that instructs the LLM to output a JSON object containing
the action and the response.
"""

import json
import re
from typing import Dict, List, Optional, Tuple, TypedDict

from jinja2 import BaseLoader, Environment

from src.llm import LLM
from src.logger import Logger

# Load the prompt template from the associated Jinja2 file.
try:
    with open("src/agents/action/prompt.jinja2", "r", encoding="utf-8") as f:
        PROMPT_TEMPLATE = f.read().strip()
except FileNotFoundError:
    PROMPT_TEMPLATE = "Error: Action prompt template not found."

logger = Logger()


class ActionResponseDict(TypedDict):
    """
    Represents the structured dictionary expected from the LLM's JSON response.

    Attributes:
        response (str): A human-like reply to the user, acknowledging their message
                        and stating the action to be taken.
        action (str): A single keyword representing the determined action.
    """

    response: str
    action: str


class Action:
    """
    The Action agent class.

    This agent processes a conversation history, renders it into a structured
    prompt for an LLM, and then parses the LLM's JSON response to determine
    the next action and a suitable reply.

    Attributes:
        llm (LLM): An instance of the Large Language Model client.
    """

    def __init__(self, base_model: str) -> None:
        """
        Initialize the Action agent.

        Args:
            base_model (str): The identifier of the base LLM model to be used.
        """
        self.llm: LLM = LLM(model_id=base_model)
        # project_dir and Config were not used, so removed.
        if PROMPT_TEMPLATE == "Error: Action prompt template not found.":
            logger.error(
                "Action agent initialized with a missing prompt template. "
                "Functionality will be impaired."
            )

    def render(self, conversation: List[str]) -> str:
        """
        Render the conversation history into the predefined Jinja2 template.

        Args:
            conversation (List[str]): A list of messages representing the
                                      conversation history.

        Returns:
            str: The fully rendered prompt string to be sent to the LLM.
        """
        if PROMPT_TEMPLATE == "Error: Action prompt template not found.":
            return f"Action prompt template is missing. Conversation: {conversation}"
        env = Environment(loader=BaseLoader())
        template = env.from_string(PROMPT_TEMPLATE)
        return template.render(conversation=conversation)

    def parse_and_validate_response(
        self, llm_response: str
    ) -> Optional[ActionResponseDict]:
        """
        Parse and validate the LLM's JSON response.

        The method attempts to extract JSON from a markdown code block if present.
        It then validates if the parsed JSON contains "response" (str) and
        "action" (str) keys.

        Args:
            llm_response (str): The raw string response from the LLM.

        Returns:
            Optional[ActionResponseDict]: The parsed and validated response as a
                                          dictionary, or None if parsing or
                                          validation fails.
        """
        json_string = llm_response
        # Attempt to extract JSON from a code block if present
        match = re.search(r"```json\s*(\{.*?\})\s*```", llm_response, re.DOTALL)
        if match:
            json_string = match.group(1)
        else:
            logger.info(
                "No JSON code block found in action response, attempting to parse entire response."
            )

        try:
            parsed_json: Dict = json.loads(json_string)
        except json.JSONDecodeError as e:
            logger.error(
                f"Failed to parse action response JSON: {e}. Response: {json_string[:500]}..."
            )
            return None

        if not isinstance(parsed_json, dict):
            logger.error(f"Action response is not a JSON object: {parsed_json}")
            return None

        agent_reply = parsed_json.get("response")
        action_keyword = parsed_json.get("action")

        if not isinstance(agent_reply, str) or not isinstance(action_keyword, str):
            logger.error(
                f"'response' or 'action' fields are missing or not strings. "
                f"Found: response='{agent_reply}', action='{action_keyword}'. "
                f"Response: {json_string[:500]}..."
            )
            return None

        # Further validation for known action keywords could be added here if desired.
        # For now, we assume any string is a potentially valid action from the LLM.

        return {"response": agent_reply, "action": action_keyword}

    def execute(
        self, conversation: List[str], project_name: str
    ) -> Optional[Tuple[str, str]]:
        """
        Execute the action determination process.

        This involves rendering the conversation into a prompt, sending it to the LLM,
        and then parsing and validating the response to extract the agent's textual
        response and the determined action keyword. The recursive retry logic
        has been removed.

        Args:
            conversation (List[str]): The conversation history.
            project_name (str): The name of the project, used for LLM context/logging.

        Returns:
            Optional[Tuple[str, str]]: A tuple containing the agent's textual response
                                      and the action keyword if successful, otherwise None.
        """
        if PROMPT_TEMPLATE == "Error: Action prompt template not found.":
            logger.error("Cannot execute action agent due to missing prompt template.")
            return None

        rendered_prompt = self.render(conversation)
        llm_response_str = self.llm.inference(rendered_prompt, project_name)

        parsed_data = self.parse_and_validate_response(llm_response_str)

        if not parsed_data:
            logger.error(
                "Failed to get a valid structured response from LLM for action agent."
            )
            # Caller can decide to retry or handle the None case.
            # For now, returning None signifies failure.
            return None

        logger.info(
            f"Action determined: {parsed_data['action']}. Agent response: {parsed_data['response']}"
        )
        return parsed_data["response"], parsed_data["action"]
