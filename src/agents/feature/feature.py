"""
Handles the implementation of new features into an existing codebase.

This module defines the `Feature` class, which is responsible for taking a
feature request (usually from a user through conversation), existing code context,
and system information to generate or modify code files to implement the feature.
It uses a Large Language Model (LLM) and a Jinja2 template that instructs the LLM
to output a JSON list of file objects (similar to the Coder agent).
"""
import os
import time
import json
import re
from typing import List, Dict, TypedDict, Optional

from jinja2 import Environment, BaseLoader

from src.config import Config
from src.llm import LLM
from src.state import AgentState, StateType # Assuming StateType is defined in src.state
from src.logger import Logger

# Load the prompt template.
try:
    with open("src/agents/feature/prompt.jinja2", "r", encoding="utf-8") as f:
        PROMPT_TEMPLATE = f.read().strip()
except FileNotFoundError:
    PROMPT_TEMPLATE = "Error: Feature prompt template not found."

logger = Logger()


# Using a similar structure as Coder for consistency
class FeatureCodeFileDict(TypedDict):
    """
    Represents a single file object with its name and content for a feature.

    Attributes:
        file_name (str): The name/path of the file (new or modified).
        code_content (str): The complete content of the file after adding the feature.
    """

    file_name: str
    code_content: str


FeatureResponseDict = List[FeatureCodeFileDict]


class Feature:
    """
    The Feature agent class.

    This agent implements new features based on user requests, existing code,
    and conversation context. It instructs the LLM to return a JSON list of
    file objects (new or modified), parses this response, and saves the files.

    Attributes:
        project_dir (str): The base directory where projects are stored.
        logger (Logger): An instance of the Logger class.
        llm (LLM): An instance of the Large Language Model client.
    """

    def __init__(self, base_model: str) -> None:
        """
        Initialize the Feature agent.

        Args:
            base_model (str): The identifier of the base LLM model to be used.
        """
        config = Config()
        self.project_dir: str = config.get_projects_dir()
        self.logger: Logger = Logger()
        self.llm: LLM = LLM(model_id=base_model)
        if PROMPT_TEMPLATE == "Error: Feature prompt template not found.":
            self.logger.error(
                "Feature agent initialized with a missing prompt template. "
                "Functionality will be impaired."
            )

    def _get_project_path(self, project_name: str) -> str:
        """
        Generate the absolute path for a project's directory.

        Args:
            project_name (str): The name of the project.

        Returns:
            str: The absolute path to the project directory.
        """
        project_name_slug = project_name.lower().replace(" ", "-")
        return os.path.join(self.project_dir, project_name_slug)

    def render(
        self,
        conversation: List[str],
        code_markdown: str,
        system_os: str,
    ) -> str:
        """
        Render the input data into the predefined Jinja2 template for the Feature agent.

        Args:
            conversation (List[str]): The history of the conversation with the user.
            code_markdown (str): Markdown representation of the existing relevant code.
            system_os (str): The operating system context.

        Returns:
            str: The fully rendered prompt string to be sent to the LLM.
        """
        if PROMPT_TEMPLATE == "Error: Feature prompt template not found.":
            return f"Feature prompt template is missing. Conversation: {conversation[-1] if conversation else 'N/A'}"
        env = Environment(loader=BaseLoader())
        template = env.from_string(PROMPT_TEMPLATE)
        return template.render(
            conversation=conversation,
            code_markdown=code_markdown,
            system_os=system_os,
        )

    def parse_response(self, response: str) -> Optional[FeatureResponseDict]:
        """
        Parse the LLM's JSON response string into a list of FeatureCodeFileDict.

        Args:
            response (str): The raw string response from the LLM, expected to be
                            a JSON list of file objects, possibly within a code block.

        Returns:
            Optional[FeatureResponseDict]: A list of `FeatureCodeFileDict` if parsing is
                                           successful and basic validation passes,
                                           otherwise None.
        """
        json_string = response
        match = re.search(r"```json\s*(\[.*?\])\s*```", response, re.DOTALL)
        if match:
            json_string = match.group(1)
        else:
            self.logger.info("No JSON code block found in feature agent response, attempting to parse entire response.")

        try:
            parsed_list: List[Dict[str, str]] = json.loads(json_string)
            if not isinstance(parsed_list, list):
                self.logger.error(
                    f"Feature response is not a JSON list. Response: {json_string[:500]}..."
                )
                return None

            validated_list: FeatureResponseDict = []
            for item in parsed_list:
                if isinstance(item, dict) and "file_name" in item and "code_content" in item and \
                   isinstance(item["file_name"], str) and isinstance(item["code_content"], str):
                    validated_list.append(
                        {"file_name": item["file_name"], "code_content": item["code_content"]}
                    )
                else:
                    self.logger.error(
                        f"Invalid item structure in feature response list: {item}. Response: {json_string[:500]}..."
                    )
                    return None
            return validated_list
        except json.JSONDecodeError as e:
            self.logger.error(
                f"Failed to parse feature response JSON: {e}. Response: {json_string[:500]}..."
            )
            return None

    def save_code_to_project(
        self, code_files: FeatureResponseDict, project_name: str
    ) -> Optional[str]:
        """
        Save the generated/modified code files to the specified project directory.

        Args:
            code_files (FeatureResponseDict): A list of dictionaries, where each
                                              dictionary contains "file_name" and
                                              "code_content".
            project_name (str): The name of the project.

        Returns:
            Optional[str]: The path to the project directory if files were saved,
                           otherwise None.
        """
        project_path = self._get_project_path(project_name)
        os.makedirs(project_path, exist_ok=True)
        self.logger.info(f"Saving feature code to project directory: {project_path}")

        if not code_files:
            self.logger.warning("No code files provided by feature agent to save.")
            return None

        for file_item in code_files:
            file_name = file_item["file_name"]
            code_content = file_item["code_content"]

            if ".." in file_name or os.path.isabs(file_name):
                self.logger.error(f"Invalid or insecure file path provided by feature agent: {file_name}")
                continue

            full_file_path = os.path.join(project_path, file_name)
            file_dir = os.path.dirname(full_file_path)

            try:
                if file_dir:
                    os.makedirs(file_dir, exist_ok=True)
                with open(full_file_path, "w", encoding="utf-8") as f:
                    f.write(code_content)
                self.logger.info(f"Successfully saved feature code to: {full_file_path}")
            except IOError as e:
                self.logger.error(f"Error saving feature file {full_file_path}: {e}")
            except Exception as e:
                self.logger.error(f"An unexpected error occurred while saving feature file {full_file_path}: {e}")
        return project_path

    def _create_markdown_from_code_set(
        self, code_set: FeatureResponseDict
    ) -> str:
        """
        Convert a list of code file dictionaries to a Markdown string for logging/display.

        Args:
            code_set (FeatureResponseDict): A list of code file dictionaries.

        Returns:
            str: A Markdown formatted string representing the code set.
        """
        markdown_parts = []
        for file_item in code_set:
            lang = file_item["file_name"].split(".")[-1] if "." in file_item["file_name"] else ""
            markdown_parts.append(
                f"File: `{file_item['file_name']}`:\n```{lang}\n{file_item['code_content']}\n```"
            )
        return "\n\n".join(markdown_parts)

    def emulate_code_writing(
        self, code_set: FeatureResponseDict, project_name: str
    ) -> None:
        """
        Emulate the process of writing code to provide visual feedback in the UI.

        Args:
            code_set (FeatureResponseDict): A list of code file dictionaries.
            project_name (str): The name of the project.
        """
        agent_state_manager = AgentState()
        for file_item in code_set:
            file_name = file_item["file_name"]
            code_content = file_item["code_content"]

            current_state: Optional[StateType] = agent_state_manager.get_latest_state(project_name)
            new_state: StateType = agent_state_manager.new_state()

            if current_state and "browser_session" in current_state:
                new_state["browser_session"] = current_state["browser_session"]

            new_state["internal_monologue"] = f"Implementing feature by writing to file: {file_name}..."
            new_state["terminal_session"] = { # type: ignore
                "title": f"Editing {file_name} for new feature",
                "command": f"vim {file_name}", # Emulated command
                "output": code_content,
            }
            agent_state_manager.add_to_current_state(project_name, new_state)
            time.sleep(1)

    def execute(
        self,
        conversation: List[str],
        code_markdown: str,
        system_os: str,
        project_name: str,
    ) -> Optional[FeatureResponseDict]:
        """
        Execute the feature implementation process.

        Renders the prompt, sends it to the LLM, parses the JSON response,
        and emulates code writing for UI feedback.

        Args:
            conversation (List[str]): The conversation history with the user.
            code_markdown (str): Markdown representation of existing relevant code.
            system_os (str): The operating system context.
            project_name (str): The name of the project.

        Returns:
            Optional[FeatureResponseDict]: A list of `FeatureCodeFileDict` containing
                                           the new/modified files and their content
                                           if successful, otherwise None.
        """
        if PROMPT_TEMPLATE == "Error: Feature prompt template not found.":
            self.logger.error("Cannot execute feature agent due to missing prompt template.")
            return None

        rendered_prompt = self.render(conversation, code_markdown, system_os)
        llm_response = self.llm.inference(rendered_prompt, project_name)

        parsed_code_files = self.parse_response(llm_response)

        if not parsed_code_files:
            self.logger.error(
                "Failed to parse valid code files from LLM response for feature agent."
            )
            return None

        self.logger.info(
            f"Successfully parsed {len(parsed_code_files)} file(s) from LLM for feature implementation."
        )
        self.emulate_code_writing(parsed_code_files, project_name)

        return parsed_code_files
