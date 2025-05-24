"""
Handles the generation of answers to user questions based on conversation and code context.

This module defines the `Answer` class, which is responsible for formulating
a direct and informative answer to a user's question. It uses a Large Language
Model (LLM) and a Jinja2 template, expecting the LLM to return a JSON object
containing the answer string.
"""

import json
import re
from typing import List, Optional, TypedDict

from jinja2 import BaseLoader, Environment

from src.llm import LLM
from src.logger import Logger

# Load the prompt template.
try:
    with open("src/agents/answer/prompt.jinja2", "r", encoding="utf-8") as f:
        PROMPT_TEMPLATE = f.read().strip()
except FileNotFoundError:
    PROMPT_TEMPLATE = "Error: Answer prompt template not found."

logger = Logger()


class AnswerResponseDict(TypedDict):
    """
    Represents the structured dictionary expected from the LLM's JSON response.

    Attributes:
        response (str): The agent's textual answer to the user's question.
    """

    response: str


class Answer:
    """
    The Answer agent class.

    This agent takes the conversation history and code context, renders them into
    a structured prompt for an LLM, and then processes the LLM's JSON response
    to extract the answer.

    Attributes:
        llm (LLM): An instance of the Large Language Model client.
    """

    def __init__(self, base_model: str) -> None:
        """
        Initialize the Answer agent.

        Args:
            base_model (str): The identifier of the base LLM model to be used.
        """
        self.llm: LLM = LLM(model_id=base_model)
        # Config and project_dir were not used, so removed.
        if PROMPT_TEMPLATE == "Error: Answer prompt template not found.":
            logger.error(
                "Answer agent initialized with a missing prompt template. "
                "Functionality will be impaired."
            )

    def render(self, conversation: List[str], code_markdown: str) -> str:
        """
        Render the conversation and code context into the predefined Jinja2 template.

        Args:
            conversation (List[str]): The history of messages in the conversation.
            code_markdown (str): Markdown formatted string of the relevant code context.

        Returns:
            str: The fully rendered prompt string to be sent to the LLM.
        """
        if PROMPT_TEMPLATE == "Error: Answer prompt template not found.":
            return f"Answer prompt template is missing. Conversation: {conversation}, Code: {code_markdown[:200]}"
        env = Environment(loader=BaseLoader())
        template = env.from_string(PROMPT_TEMPLATE)
        return template.render(conversation=conversation, code_markdown=code_markdown)

    def parse_and_validate_response(
        self, llm_response: str
    ) -> Optional[AnswerResponseDict]:
        """
        Parse and validate the LLM's JSON response.

        The method attempts to extract JSON from a markdown code block if present.
        It then validates if the parsed JSON contains the "response" key
        and if its value is a non-empty string.

        Args:
            llm_response (str): The raw string response from the LLM.

        Returns:
            Optional[AnswerResponseDict]: The parsed and validated response as a
                                          dictionary, or None if parsing or
                                          validation fails.
        """
        json_string = llm_response
        match = re.search(r"```json\s*(\{.*?\})\s*```", llm_response, re.DOTALL)
        if match:
            json_string = match.group(1)
        else:
            logger.info(
                "No JSON code block found in answer response, attempting to parse entire response."
            )

        try:
            parsed_json: AnswerResponseDict = json.loads(json_string)
        except json.JSONDecodeError as e:
            logger.error(
                f"Failed to parse answer JSON: {e}. Response: {json_string[:500]}..."
            )
            return None

        if not isinstance(parsed_json, dict):
            logger.error(f"Answer response is not a JSON object: {parsed_json}")
            return None

        answer_text = parsed_json.get("response")

        if not isinstance(answer_text, str) or not answer_text.strip():
            logger.error(
                "'response' field is missing, not a string, or empty. "
                f"Found: '{answer_text}'. Response: {json_string[:500]}..."
            )
            return None

        return {"response": answer_text}

    def execute(
        self,
        conversation: List[str],
        code_markdown: str,
        project_name: Optional[str] = None,
    ) -> Optional[str]:
        """
        Execute the answer generation process.

        This involves rendering the conversation and code context into a prompt,
        sending it to the LLM, and then parsing and validating the JSON response
        to extract the answer. The recursive retry logic has been removed.

        Args:
            conversation (List[str]): The history of messages in the conversation.
            code_markdown (str): Markdown formatted string of the relevant code context.
            project_name (Optional[str]): The name of the project, used for LLM
                                       context/logging. Defaults to "answer_task".

        Returns:
            Optional[str]: The generated answer string if successful, otherwise None.
        """
        if PROMPT_TEMPLATE == "Error: Answer prompt template not found.":
            logger.error("Cannot execute answer agent due to missing prompt template.")
            return None

        rendered_prompt = self.render(conversation, code_markdown)
        llm_response_str = self.llm.inference(
            rendered_prompt, project_name or "answer_task"
        )

        parsed_data = self.parse_and_validate_response(llm_response_str)

        if not parsed_data:
            logger.error("Failed to get a valid structured answer from LLM.")
            return None  # Caller can decide how to handle this.

        logger.info(f"Generated Answer: {parsed_data['response']}")
        return parsed_data["response"]
