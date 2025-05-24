"""
Handles high-level decision-making or routing based on user prompts.

This module defines the `Decision` class, which is responsible for interpreting
a user's prompt and determining a sequence of one or more function calls
(actions) required to fulfill the request. It uses a Large Language Model (LLM)
and a Jinja2 template that instructs the LLM to output a JSON list of
decision items, each specifying a function, its arguments, and a reply.
"""
import json
import re
from typing import List, Dict, Any, TypedDict, Optional

from jinja2 import Environment, BaseLoader

from src.llm import LLM
from src.logger import Logger

# Load the prompt template.
try:
    with open("src/agents/decision/prompt.jinja2", "r", encoding="utf-8") as f:
        PROMPT_TEMPLATE = f.read().strip()
except FileNotFoundError:
    PROMPT_TEMPLATE = "Error: Decision prompt template not found."

logger = Logger()

DecisionArgsDict = Dict[str, Any]


class DecisionItemDict(TypedDict):
    """
    Represents a single decision item in the list returned by the LLM.

    Attributes:
        function (str): The name of the function to be called.
        args (DecisionArgsDict): A dictionary of arguments for the function.
        reply (str): A human-like reply to the user for this specific action.
    """

    function: str
    args: DecisionArgsDict
    reply: str


DecisionResponseDict = List[DecisionItemDict]


class Decision:
    """
    The Decision agent class.

    This agent processes a user prompt, renders it into a structured format for an LLM,
    and then parses the LLM's JSON response to determine a list of actions (function calls)
    to be executed.

    Attributes:
        llm (LLM): An instance of the Large Language Model client.
    """

    def __init__(self, base_model: str) -> None:
        """
        Initialize the Decision agent.

        Args:
            base_model (str): The identifier of the base LLM model to be used.
        """
        self.llm: LLM = LLM(model_id=base_model)
        if PROMPT_TEMPLATE == "Error: Decision prompt template not found.":
            logger.error(
                "Decision agent initialized with a missing prompt template. "
                "Functionality will be impaired."
            )

    def render(self, prompt: str) -> str:
        """
        Render the user's prompt into the predefined Jinja2 template.

        Args:
            prompt (str): The user's input prompt.

        Returns:
            str: The fully rendered prompt string to be sent to the LLM.
        """
        if PROMPT_TEMPLATE == "Error: Decision prompt template not found.":
            return f"Decision prompt template is missing. User prompt: {prompt}"
        env = Environment(loader=BaseLoader())
        template = env.from_string(PROMPT_TEMPLATE)
        return template.render(prompt=prompt)

    def parse_response(self, response: str) -> Optional[DecisionResponseDict]:
        """
        Parse the LLM's JSON response string into a list of DecisionItemDict.

        Args:
            response (str): The raw string response from the LLM, expected to be
                            a JSON list of decision objects, possibly within a code block.

        Returns:
            Optional[DecisionResponseDict]: A list of `DecisionItemDict` if parsing
                                            is successful and validation passes,
                                            otherwise None.
        """
        json_string = response
        match = re.search(r"```json\s*(\[.*?\])\s*```", response, re.DOTALL)
        if match:
            json_string = match.group(1)
        else:
            logger.info("No JSON code block found in decision response, attempting to parse entire response.")

        try:
            parsed_list: List[Dict[str, Any]] = json.loads(json_string)

            if not isinstance(parsed_list, list):
                logger.error(
                    f"Decision response is not a JSON list. Response: {json_string[:500]}..."
                )
                return None

            validated_list: DecisionResponseDict = []
            for item in parsed_list:
                if (
                    isinstance(item, dict)
                    and "function" in item
                    and isinstance(item["function"], str)
                    and "args" in item
                    and isinstance(item["args"], dict)
                    and "reply" in item
                    and isinstance(item["reply"], str)
                ):
                    validated_list.append(
                        {
                            "function": item["function"],
                            "args": item["args"],
                            "reply": item["reply"],
                        }
                    )
                else:
                    logger.error(
                        f"Invalid item structure in decision response list: {item}. Response: {json_string[:500]}..."
                    )
                    return None  # Or skip invalid items based on desired strictness
            return validated_list
        except json.JSONDecodeError as e:
            logger.error(
                f"Failed to parse decision response JSON: {e}. Response: {json_string[:500]}..."
            )
            return None

    def execute(
        self, prompt: str, project_name: Optional[str] = None
    ) -> Optional[DecisionResponseDict]:
        """
        Execute the decision-making process.

        This involves rendering the prompt, sending it to the LLM, and then parsing
        and validating the JSON response to extract the list of decided actions.

        Args:
            prompt (str): The user's input prompt.
            project_name (Optional[str]): The name of the project, used for LLM
                                       context/logging. Defaults to "decision_task".

        Returns:
            Optional[DecisionResponseDict]: A list of `DecisionItemDict` representing
                                            the decided actions if successful,
                                            otherwise None.
        """
        if PROMPT_TEMPLATE == "Error: Decision prompt template not found.":
            logger.error("Cannot execute decision agent due to missing prompt template.")
            return None

        rendered_prompt = self.render(prompt)
        llm_response_str = self.llm.inference(
            rendered_prompt, project_name or "decision_task"
        )

        parsed_decisions = self.parse_response(llm_response_str)

        if not parsed_decisions:
            logger.error(
                "Failed to get a valid structured decision list from LLM."
            )
            return None

        logger.info(f"LLM decided on {len(parsed_decisions)} action(s).")
        return parsed_decisions
