"""
Handles the code generation phase based on a plan, user context, and research.

This module defines the `Coder` class, which is responsible for generating code
files. It uses a Large Language Model (LLM) and a Jinja2 template that instructs
the LLM to output a JSON list of file objects, each containing a filename and
its corresponding code content.
"""
import os
import time
import json
import re
from typing import List, Dict, TypedDict, Optional

from jinja2 import Environment, BaseLoader

from src.config import Config
from src.llm import LLM
from src.state import AgentState, StateType
from src.logger import Logger

# Load the prompt template from the associated Jinja2 file.
try:
    with open("src/agents/coder/prompt.jinja2", "r", encoding="utf-8") as f:
        PROMPT_TEMPLATE = f.read().strip()
except FileNotFoundError:
    PROMPT_TEMPLATE = "Error: Coder prompt template not found."

logger = Logger()


class CodeFileDict(TypedDict):
    """
    Represents a single file object with its name and content.

    Attributes:
        file_name (str): The name/path of the file.
        code_content (str): The complete content of the file.
    """

    file_name: str
    code_content: str


CoderResponseDict = List[CodeFileDict]


class Coder:
    """
    The Coder agent class.

    This agent generates code based on a provided plan, user context, and search
    results. It instructs the LLM to return a JSON list of file objects,
    parses this response, and saves the files to the project directory.

    Attributes:
        project_dir (str): The base directory where projects are stored.
        logger (Logger): An instance of the Logger class.
        llm (LLM): An instance of the Large Language Model client.
    """

    def __init__(self, base_model: str) -> None:
        """
        Initialize the Coder agent.

        Args:
            base_model (str): The identifier of the base LLM model to be used.
        """
        config = Config()
        self.project_dir: str = config.get_projects_dir()
        self.logger: Logger = Logger()
        self.llm: LLM = LLM(model_id=base_model)
        if PROMPT_TEMPLATE == "Error: Coder prompt template not found.":
            self.logger.error(
                "Coder initialized with a missing prompt template. "
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
        step_by_step_plan: str,
        user_context: str,
        search_results: Dict[str, str],
    ) -> str:
        """
        Render the input data into the predefined Jinja2 template for the Coder.

        Args:
            step_by_step_plan (str): The step-by-step plan for code generation.
            user_context (str): Relevant context provided by the user.
            search_results (Dict[str, str]): A dictionary of search query results.

        Returns:
            str: The fully rendered prompt string to be sent to the LLM.
        """
        if PROMPT_TEMPLATE == "Error: Coder prompt template not found.":
            return f"Coder prompt template is missing. Plan: {step_by_step_plan[:200]}..."
        env = Environment(loader=BaseLoader())
        template = env.from_string(PROMPT_TEMPLATE)
        return template.render(
            step_by_step_plan=step_by_step_plan,
            user_context=user_context,
            search_results=search_results,
        )

    def parse_response(self, response: str) -> Optional[CoderResponseDict]:
        """
        Parse the LLM's JSON response string into a list of CodeFileDict.

        Args:
            response (str): The raw string response from the LLM, expected to be
                            a JSON list of file objects, possibly within a code block.

        Returns:
            Optional[CoderResponseDict]: A list of `CodeFileDict` if parsing is successful
                                         and basic validation passes, otherwise None.
        """
        json_string = response
        match = re.search(r"```json\s*(\[.*?\])\s*```", response, re.DOTALL)
        if match:
            json_string = match.group(1)
        else:
            self.logger.info("No JSON code block found in coder response, attempting to parse entire response.")


        try:
            parsed_list: List[Dict[str, str]] = json.loads(json_string)
            if not isinstance(parsed_list, list):
                self.logger.error(
                    f"Coder response is not a JSON list. Response: {json_string[:500]}..."
                )
                return None

            # Validate structure of each item in the list
            validated_list: CoderResponseDict = []
            for item in parsed_list:
                if isinstance(item, dict) and "file_name" in item and "code_content" in item and \
                   isinstance(item["file_name"], str) and isinstance(item["code_content"], str):
                    validated_list.append(
                        {"file_name": item["file_name"], "code_content": item["code_content"]}
                    )
                else:
                    self.logger.error(
                        f"Invalid item structure in coder response list: {item}. Response: {json_string[:500]}..."
                    )
                    return None  # Or skip invalid items, depending on desired strictness
            return validated_list
        except json.JSONDecodeError as e:
            self.logger.error(
                f"Failed to parse coder response JSON: {e}. Response: {json_string[:500]}..."
            )
            return None

    def save_code_to_project(
        self, code_files: CoderResponseDict, project_name: str
    ) -> Optional[str]:
        """
        Save the generated code files to the specified project directory.

        Args:
            code_files (CoderResponseDict): A list of dictionaries, where each dictionary
                                          contains "file_name" and "code_content".
            project_name (str): The name of the project.

        Returns:
            Optional[str]: The path to the project directory if files were saved,
                           otherwise None.
        """
        project_path = self._get_project_path(project_name)
        os.makedirs(project_path, exist_ok=True)
        self.logger.info(f"Saving code to project directory: {project_path}")

        if not code_files:
            self.logger.warning("No code files provided to save.")
            return None

        for file_item in code_files:
            file_name = file_item["file_name"]
            code_content = file_item["code_content"]

            # Ensure file_name is a relative path and secure
            if ".." in file_name or os.path.isabs(file_name):
                self.logger.error(f"Invalid or insecure file path: {file_name}")
                continue # Skip saving this file

            full_file_path = os.path.join(project_path, file_name)
            file_dir = os.path.dirname(full_file_path)

            try:
                if file_dir: # Ensure directory exists only if file_dir is not empty (e.g. for root files)
                    os.makedirs(file_dir, exist_ok=True)
                with open(full_file_path, "w", encoding="utf-8") as f:
                    f.write(code_content)
                self.logger.info(f"Successfully saved code to: {full_file_path}")
            except IOError as e:
                self.logger.error(f"Error saving file {full_file_path}: {e}")
                # Decide if one error should stop all, or continue. For now, continue.
            except Exception as e:
                self.logger.error(f"An unexpected error occurred while saving {full_file_path}: {e}")


        return project_path

    def _create_markdown_from_code_set(
        self, code_set: CoderResponseDict
    ) -> str:
        """
        Convert a list of code file dictionaries to a Markdown string.
        (Used for internal representation or logging, not for LLM response).

        Args:
            code_set (CoderResponseDict): A list of code file dictionaries.

        Returns:
            str: A Markdown formatted string representing the code set.
        """
        markdown_parts = []
        for file_item in code_set:
            # Basic language detection from extension for markdown block
            lang = file_item["file_name"].split(".")[-1] if "." in file_item["file_name"] else ""
            markdown_parts.append(
                f"File: `{file_item['file_name']}`:\n```{lang}\n{file_item['code_content']}\n```"
            )
        # The original prompt used "~~~" as delimiters, but standard markdown is ```
        return "\n\n".join(markdown_parts)


    def emulate_code_writing(
        self, code_set: CoderResponseDict, project_name: str
    ) -> None:
        """
        Emulate the process of writing code to provide visual feedback in the UI.

        This method updates the agent's state for each file being "written",
        simulating a terminal session where the code is being edited.

        Args:
            code_set (CoderResponseDict): A list of code file dictionaries.
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

            if current_state and "browser_session" in current_state: # Preserve browser session
                new_state["browser_session"] = current_state["browser_session"]

            new_state["internal_monologue"] = f"Writing code into file: {file_name}..."
            new_state["terminal_session"] = { # type: ignore
                "title": f"Editing {file_name}",
                "command": f"vim {file_name}", # Emulated command
                "output": code_content,
            }
            agent_state_manager.add_to_current_state(project_name, new_state)
            time.sleep(1) # Short delay for UI update

    def execute(
        self,
        step_by_step_plan: str,
        user_context: str,
        search_results: Dict[str, str],
        project_name: str,
    ) -> Optional[CoderResponseDict]:
        """
        Execute the code generation process.

        This involves rendering the prompt, sending it to the LLM, parsing the
        JSON response, and then optionally emulating the code writing process for UI.

        Args:
            step_by_step_plan (str): The detailed plan for code generation.
            user_context (str): Context provided by the user.
            search_results (Dict[str, str]): Results from prior research.
            project_name (str): The name of the project.

        Returns:
            Optional[CoderResponseDict]: A list of `CodeFileDict` containing the generated
                                         files and their content if successful,
                                         otherwise None.
        """
        if PROMPT_TEMPLATE == "Error: Coder prompt template not found.":
            self.logger.error("Cannot execute coder due to missing prompt template.")
            return None

        rendered_prompt = self.render(
            step_by_step_plan, user_context, search_results
        )
        llm_response = self.llm.inference(rendered_prompt, project_name)

        parsed_code_files = self.parse_response(llm_response)

        if not parsed_code_files:
            self.logger.error(
                "Failed to parse valid code files from LLM response for coder."
            )
            # Consider a retry mechanism or returning an error state
            return None

        self.logger.info(f"Successfully parsed {len(parsed_code_files)} file(s) from LLM response.")
        self.emulate_code_writing(parsed_code_files, project_name)

        return parsed_code_files
