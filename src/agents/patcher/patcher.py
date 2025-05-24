"""
Handles the patching of code based on errors and context.

This module defines the `Patcher` class, responsible for generating code patches
when errors occur during execution. It uses a Large Language Model (LLM) and a
Jinja2 template that instructs the LLM to output a JSON list of file objects,
each containing a filename and its corresponding corrected code content.
"""

import json
import os
import re
import time
from typing import Dict, List, Optional, TypedDict

from jinja2 import BaseLoader, Environment

from src.config import Config
from src.llm import LLM
from src.logger import Logger
from src.state import (  # Assuming StateType is defined in src.state
    AgentState,
    StateType,
)

# Load the prompt template.
try:
    with open("src/agents/patcher/prompt.jinja2", "r", encoding="utf-8") as f:
        PROMPT_TEMPLATE = f.read().strip()
except FileNotFoundError:
    PROMPT_TEMPLATE = "Error: Patcher prompt template not found."

logger = Logger()


# Using a similar structure as Coder and Feature for consistency
class PatcherCodeFileDict(TypedDict):
    """
    Represents a single file object with its name and patched content.

    Attributes:
        file_name (str): The name/path of the file that was patched.
        code_content (str): The complete, corrected content of the file.
    """

    file_name: str
    code_content: str


PatcherResponseDict = List[PatcherCodeFileDict]


class Patcher:
    """
    The Patcher agent class.

    This agent attempts to fix bugs in code by generating patches. It takes
    conversation history, existing code, commands that led to an error, the error
    message itself, and system OS information as context. It instructs the LLM
    to return a JSON list of file objects (with complete, patched code),
    parses this response, and saves the files.

    Attributes:
        project_dir (str): The base directory where projects are stored.
        logger (Logger): An instance of the Logger class.
        llm (LLM): An instance of the Large Language Model client.
    """

    def __init__(self, base_model: str) -> None:
        """
        Initialize the Patcher agent.

        Args:
            base_model (str): The identifier of the base LLM model to be used.
        """
        config = Config()
        self.project_dir: str = config.get_projects_dir()
        self.logger: Logger = Logger()
        self.llm: LLM = LLM(model_id=base_model)
        if PROMPT_TEMPLATE == "Error: Patcher prompt template not found.":
            self.logger.error(
                "Patcher agent initialized with a missing prompt template. "
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
        commands: Optional[List[str]],
        error: str,
        system_os: str,
    ) -> str:
        """
        Render the input data into the predefined Jinja2 template for the Patcher.

        Args:
            conversation (List[str]): The history of the conversation.
            code_markdown (str): Markdown representation of the existing code.
            commands (Optional[List[str]]): Commands that were executed and led to the error.
            error (str): The error message encountered.
            system_os (str): The operating system context.

        Returns:
            str: The fully rendered prompt string to be sent to the LLM.
        """
        if PROMPT_TEMPLATE == "Error: Patcher prompt template not found.":
            return f"Patcher prompt template is missing. Error: {error[:200]}"
        env = Environment(loader=BaseLoader())
        template = env.from_string(PROMPT_TEMPLATE)
        return template.render(
            conversation=conversation,
            code_markdown=code_markdown,
            commands=commands or [],  # Ensure commands is a list for the template
            error=error,
            system_os=system_os,
        )

    def parse_response(self, response: str) -> Optional[PatcherResponseDict]:
        """
        Parse the LLM's JSON response string into a list of PatcherCodeFileDict.

        Args:
            response (str): The raw string response from the LLM, expected to be
                            a JSON list of file objects, possibly within a code block.

        Returns:
            Optional[PatcherResponseDict]: A list of `PatcherCodeFileDict` if parsing is
                                           successful and basic validation passes,
                                           otherwise None.
        """
        json_string = response
        match = re.search(r"```json\s*(\[.*?\])\s*```", response, re.DOTALL)
        if match:
            json_string = match.group(1)
        else:
            self.logger.info(
                "No JSON code block found in patcher response, attempting to parse entire response."
            )

        try:
            parsed_list: List[Dict[str, str]] = json.loads(json_string)
            if not isinstance(parsed_list, list):
                self.logger.error(
                    f"Patcher response is not a JSON list. Response: {json_string[:500]}..."
                )
                return None

            validated_list: PatcherResponseDict = []
            for item in parsed_list:
                if (
                    isinstance(item, dict)
                    and "file_name" in item
                    and "code_content" in item
                    and isinstance(item["file_name"], str)
                    and isinstance(item["code_content"], str)
                ):
                    validated_list.append(
                        {
                            "file_name": item["file_name"],
                            "code_content": item["code_content"],
                        }
                    )
                else:
                    self.logger.error(
                        f"Invalid item structure in patcher response list: {item}. Response: {json_string[:500]}..."
                    )
                    return None
            return validated_list
        except json.JSONDecodeError as e:
            self.logger.error(
                f"Failed to parse patcher response JSON: {e}. Response: {json_string[:500]}..."
            )
            return None

    def save_code_to_project(
        self, code_files: PatcherResponseDict, project_name: str
    ) -> Optional[str]:
        """
        Save the patched code files to the specified project directory.

        Args:
            code_files (PatcherResponseDict): A list of dictionaries, where each
                                              dictionary contains "file_name" and
                                              "code_content" (the patched code).
            project_name (str): The name of the project.

        Returns:
            Optional[str]: The path to the project directory if files were saved,
                           otherwise None.
        """
        project_path = self._get_project_path(project_name)
        os.makedirs(project_path, exist_ok=True)
        self.logger.info(f"Saving patched code to project directory: {project_path}")

        if not code_files:
            self.logger.warning("No patched code files provided to save.")
            return None

        for file_item in code_files:
            file_name = file_item["file_name"]
            code_content = file_item["code_content"]

            if ".." in file_name or os.path.isabs(file_name):
                self.logger.error(
                    f"Invalid or insecure file path provided by patcher: {file_name}"
                )
                continue

            full_file_path = os.path.join(project_path, file_name)
            file_dir = os.path.dirname(full_file_path)

            try:
                if file_dir:
                    os.makedirs(file_dir, exist_ok=True)
                with open(full_file_path, "w", encoding="utf-8") as f:
                    f.write(code_content)
                self.logger.info(
                    f"Successfully saved patched code to: {full_file_path}"
                )
            except IOError as e:
                self.logger.error(f"Error saving patched file {full_file_path}: {e}")
            except Exception as e:
                self.logger.error(
                    f"An unexpected error occurred while saving patched file {full_file_path}: {e}"
                )
        return project_path

    def _create_markdown_from_code_set(self, code_set: PatcherResponseDict) -> str:
        """
        Convert a list of code file dictionaries to a Markdown string for logging/display.

        Args:
            code_set (PatcherResponseDict): A list of code file dictionaries.

        Returns:
            str: A Markdown formatted string representing the code set.
        """
        markdown_parts = []
        for file_item in code_set:
            lang = (
                file_item["file_name"].split(".")[-1]
                if "." in file_item["file_name"]
                else ""
            )
            markdown_parts.append(
                f"File: `{file_item['file_name']}`:\n```{lang}\n{file_item['code_content']}\n```"
            )
        return "\n\n".join(markdown_parts)

    def emulate_code_writing(
        self, code_set: PatcherResponseDict, project_name: str
    ) -> None:
        """
        Emulate the process of writing (patching) code for UI feedback.

        Args:
            code_set (PatcherResponseDict): A list of code file dictionaries.
            project_name (str): The name of the project.
        """
        agent_state_manager = AgentState()
        for file_item in code_set:
            file_name = file_item["file_name"]
            code_content = file_item["code_content"]

            current_state: Optional[StateType] = agent_state_manager.get_latest_state(
                project_name
            )
            new_state: StateType = agent_state_manager.new_state()

            if current_state and "browser_session" in current_state:
                new_state["browser_session"] = current_state["browser_session"]

            new_state["internal_monologue"] = f"Applying patch to file: {file_name}..."
            new_state["terminal_session"] = {  # type: ignore
                "title": f"Patching {file_name}",
                "command": f"vim {file_name}",  # Emulated command
                "output": code_content,  # Shows the new content
            }
            agent_state_manager.add_to_current_state(project_name, new_state)
            time.sleep(1)

    def execute(
        self,
        conversation: List[str],
        code_markdown: str,
        commands: Optional[List[str]],
        error: str,
        system_os: str,
        project_name: str,
    ) -> Optional[PatcherResponseDict]:
        """
        Execute the patching process.

        Renders the prompt with context (conversation, code, error, etc.),
        sends it to the LLM, parses the JSON response containing patched files,
        and emulates code writing for UI feedback.

        Args:
            conversation (List[str]): History of the conversation.
            code_markdown (str): Markdown representation of the existing code.
            commands (Optional[List[str]]): Commands that led to the error.
            error (str): The error message encountered.
            system_os (str): The operating system context.
            project_name (str): The name of the project.

        Returns:
            Optional[PatcherResponseDict]: A list of `PatcherCodeFileDict` containing
                                           the patched files and their content if successful,
                                           otherwise None.
        """
        if PROMPT_TEMPLATE == "Error: Patcher prompt template not found.":
            self.logger.error("Cannot execute patcher due to missing prompt template.")
            return None

        rendered_prompt = self.render(
            conversation, code_markdown, commands, error, system_os
        )
        llm_response = self.llm.inference(rendered_prompt, project_name)

        patched_code_files = self.parse_response(llm_response)

        if not patched_code_files:
            self.logger.error(
                "Failed to parse valid patched code files from LLM response for patcher."
            )
            return None

        self.logger.info(
            f"Successfully parsed {len(patched_code_files)} file(s) from LLM for patching."
        )
        self.emulate_code_writing(patched_code_files, project_name)

        return patched_code_files
