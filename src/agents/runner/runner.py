"""
Handles the execution of code for a project, including reruns with debugging.

This module defines the `Runner` class, which is responsible for:
1.  Determining the commands needed to run a project based on its code and conversation context.
2.  Executing these commands using `subprocess.run`.
3.  Handling errors during execution by prompting an LLM for fixes (either to the
    commands or by requesting a code patch via the Patcher agent).
"""

import json
import re
import subprocess
import time
from typing import List, Optional, TypedDict

from jinja2 import BaseLoader, Environment

from src.agents.patcher import Patcher
from src.filesystem import ReadCode  # Added import
from src.llm import LLM
from src.logger import Logger
from src.project import ProjectManager
from src.state import AgentState, StateType

# Load prompt templates.
try:
    with open("src/agents/runner/prompt.jinja2", "r", encoding="utf-8") as f:
        PROMPT_TEMPLATE = f.read().strip()
except FileNotFoundError:
    PROMPT_TEMPLATE = "Error: Runner prompt template not found."

try:
    with open("src/agents/runner/rerunner.jinja2", "r", encoding="utf-8") as f:
        RERUNNER_PROMPT_TEMPLATE = f.read().strip()
except FileNotFoundError:
    RERUNNER_PROMPT_TEMPLATE = "Error: Rerunner prompt template not found."

logger = Logger()


class RunnerCommandsResponseDict(TypedDict):
    """
    Represents the structured dictionary for commands from `prompt.jinja2`.

    Attributes:
        commands (List[str]): A list of shell commands to execute.
    """

    commands: List[str]


class RerunnerActionResponseDict(TypedDict):
    """
    Represents the structured dictionary for actions from `rerunner.jinja2`.

    Attributes:
        action (str): The action to take ("patch" or "command").
        command (Optional[str]): The corrected command if action is "command".
        response (str): A human-like response describing the situation/fix.
    """

    action: str
    command: Optional[str]  # Only present if action is "command"
    response: str


class Runner:
    """
    The Runner agent class.

    This agent determines commands to run a project, executes them, and handles
    errors by consulting an LLM for fixes (either command changes or code patches).

    Attributes:
        base_model (str): The base model ID for the LLM.
        llm (LLM): An instance of the Large Language Model client.
        agent_state (AgentState): Manages the agent's state.
        project_manager (ProjectManager): Manages project-related data.
    """

    def __init__(self, base_model: str) -> None:
        """
        Initialize the Runner agent.

        Args:
            base_model (str): The identifier of the base LLM model to be used.
        """
        self.base_model: str = base_model
        self.llm: LLM = LLM(model_id=base_model)
        self.agent_state: AgentState = AgentState()
        self.project_manager: ProjectManager = ProjectManager()

        if PROMPT_TEMPLATE == "Error: Runner prompt template not found.":
            logger.error("Runner initialized with missing main prompt template.")
        if RERUNNER_PROMPT_TEMPLATE == "Error: Rerunner prompt template not found.":
            logger.error("Runner initialized with missing rerunner prompt template.")

    def _render_initial_prompt(
        self, conversation: List[str], code_markdown: str, system_os: str
    ) -> str:
        """
        Render the initial prompt for determining run commands.

        Args:
            conversation (List[str]): The conversation history.
            code_markdown (str): Markdown formatted string of the project's code.
            system_os (str): The operating system of the target environment.

        Returns:
            str: The rendered prompt.
        """
        if PROMPT_TEMPLATE == "Error: Runner prompt template not found.":
            return "Runner prompt template is missing."
        env = Environment(loader=BaseLoader())
        template = env.from_string(PROMPT_TEMPLATE)
        return template.render(
            conversation=conversation,
            code_markdown=code_markdown,
            system_os=system_os,
        )

    def _render_rerunner_prompt(
        self,
        conversation: List[str],
        code_markdown: str,
        system_os: str,
        commands: List[str],
        error_output: str,
    ) -> str:
        """
        Render the prompt for handling errors during execution.

        Args:
            conversation (List[str]): The conversation history.
            code_markdown (str): Markdown formatted string of the project's code.
            system_os (str): The operating system of the target environment.
            commands (List[str]): The list of commands that were attempted.
            error_output (str): The error output from the failed command.

        Returns:
            str: The rendered prompt for the rerunner.
        """
        if RERUNNER_PROMPT_TEMPLATE == "Error: Rerunner prompt template not found.":
            return "Rerunner prompt template is missing."
        env = Environment(loader=BaseLoader())
        template = env.from_string(RERUNNER_PROMPT_TEMPLATE)
        return template.render(
            conversation=conversation,
            code_markdown=code_markdown,
            system_os=system_os,
            commands=commands,
            error=error_output,
        )

    def _parse_commands_response(
        self, response: str
    ) -> Optional[RunnerCommandsResponseDict]:
        """
        Parse and validate the LLM's JSON response for initial run commands.

        Args:
            response (str): The raw LLM response string.

        Returns:
            Optional[RunnerCommandsResponseDict]: Parsed data if valid, else None.
        """
        json_string = response
        match = re.search(r"```json\s*(\{.*?\})\s*```", response, re.DOTALL)
        if match:
            json_string = match.group(1)
        else:
            logger.info(
                "No JSON code block found in runner commands response, attempting to parse entire response."
            )

        try:
            parsed_json: RunnerCommandsResponseDict = json.loads(json_string)
            if "commands" not in parsed_json or not isinstance(
                parsed_json["commands"], list
            ):
                logger.error(f"Invalid 'commands' field in LLM response: {parsed_json}")
                return None
            if not all(isinstance(cmd, str) for cmd in parsed_json["commands"]):
                logger.error(
                    f"Not all commands are strings in LLM response: {parsed_json['commands']}"
                )
                return None
            return parsed_json
        except json.JSONDecodeError as e:
            logger.error(
                f"Failed to parse runner commands JSON: {e}. Response: {json_string[:500]}..."
            )
            return None

    def _parse_rerunner_action_response(
        self, response: str
    ) -> Optional[RerunnerActionResponseDict]:
        """
        Parse and validate the LLM's JSON response for error handling actions.

        Args:
            response (str): The raw LLM response string.

        Returns:
            Optional[RerunnerActionResponseDict]: Parsed data if valid, else None.
        """
        json_string = response
        match = re.search(r"```json\s*(\{.*?\})\s*```", response, re.DOTALL)
        if match:
            json_string = match.group(1)
        else:
            logger.info(
                "No JSON code block found in rerunner action response, attempting to parse entire response."
            )

        try:
            parsed_json: RerunnerActionResponseDict = json.loads(json_string)
            if "action" not in parsed_json or "response" not in parsed_json:
                logger.error(
                    f"Missing 'action' or 'response' in rerunner LLM response: {parsed_json}"
                )
                return None
            if parsed_json["action"] == "command" and "command" not in parsed_json:
                logger.error(
                    f"'command' field missing for 'command' action in rerunner response: {parsed_json}"
                )
                return None
            # Ensure command is None if action is not "command" or make it optional in TypedDict
            if parsed_json["action"] != "command":
                parsed_json["command"] = None

            return parsed_json
        except json.JSONDecodeError as e:
            logger.error(
                f"Failed to parse rerunner action JSON: {e}. Response: {json_string[:500]}..."
            )
            return None

    def _execute_commands_in_project(
        self,
        commands_to_run: List[str],
        project_path: str,
        project_name: str,
        conversation_history: List[str],
        code_markdown_context: str,
        os_platform: str,
    ) -> bool:
        """
        Execute a list of shell commands in the project directory and handle errors.

        Args:
            commands_to_run (List[str]): The list of commands to execute.
            project_path (str): The absolute path to the project directory.
            project_name (str): The name of the project.
            conversation_history (List[str]): The current conversation history.
            code_markdown_context (str): Markdown representation of the project's code.
            os_platform (str): The operating system where commands are run.

        Returns:
            bool: True if all commands executed successfully, False otherwise.
        """
        max_retries_per_command = 2
        current_commands = list(commands_to_run)  # Make a mutable copy

        idx = 0
        while idx < len(current_commands):
            command = current_commands[idx]
            retries = 0
            command_failed = True

            while command_failed and retries <= max_retries_per_command:
                self.logger.info(
                    f"Executing command: {command} in {project_path} (Attempt {retries + 1})"
                )
                command_parts = command.split(
                    " "
                )  # Simple split, might need shlex for complex commands

                try:
                    process = subprocess.run(
                        command_parts,
                        stdout=subprocess.PIPE,
                        stderr=subprocess.PIPE,
                        cwd=project_path,
                        text=True,  # For Python 3.7+
                        check=False,  # Don't raise exception on non-zero exit
                    )
                    command_output = process.stdout + process.stderr
                    command_failed = process.returncode != 0
                except Exception as e:
                    self.logger.error(
                        f"Subprocess execution failed for command '{command}': {e}"
                    )
                    command_output = str(e)
                    command_failed = True

                latest_state: StateType = self.agent_state.new_state()
                latest_state["internal_monologue"] = (
                    f"Executing command: {command}"
                    if not command_failed
                    else f"Error executing command: {command}. Output: {command_output[:100]}..."
                )
                latest_state["terminal_session"] = {  # type: ignore
                    "title": "Terminal Output",
                    "command": command,
                    "output": command_output,
                }
                self.agent_state.add_to_current_state(project_name, latest_state)
                time.sleep(0.5)  # Allow UI to update

                if not command_failed:
                    self.logger.info(f"Command '{command}' executed successfully.")
                    idx += 1  # Move to next command
                    break  # Exit retry loop for this command

                # Command failed, attempt to handle error
                retries += 1
                if retries > max_retries_per_command:
                    self.logger.error(
                        f"Command '{command}' failed after {max_retries_per_command} retries. Aborting run."
                    )
                    self.project_manager.add_message_from_devika(
                        project_name,
                        f"I encountered an error with command: `{command}` that I couldn't resolve after multiple attempts. Error: {command_output[:200]}...",
                    )
                    return False  # Overall execution failed

                self.project_manager.add_message_from_devika(
                    project_name,
                    f"I encountered an error with command: `{command}`. Error: {command_output[:200]}. Let me try to fix this.",
                )

                rerunner_prompt = self._render_rerunner_prompt(
                    conversation=conversation_history,
                    code_markdown=code_markdown_context,
                    system_os=os_platform,
                    commands=current_commands,  # Pass current list of commands
                    error_output=command_output,
                )
                if "Rerunner prompt template is missing" in rerunner_prompt:
                    self.logger.error(
                        "Cannot attempt rerun due to missing rerunner prompt template."
                    )
                    return False

                llm_rerun_response = self.llm.inference(rerunner_prompt, project_name)
                action_data = self._parse_rerunner_action_response(llm_rerun_response)

                if not action_data:
                    self.logger.error(
                        "Failed to get a valid action from LLM for error handling."
                    )
                    continue  # Retry the same command if LLM fails to give action

                self.project_manager.add_message_from_devika(
                    project_name, action_data["response"]
                )

                if action_data["action"] == "command":
                    new_command = action_data.get("command")
                    if new_command and isinstance(new_command, str):
                        self.logger.info(f"LLM suggested new command: {new_command}")
                        current_commands[idx] = new_command  # Replace current command
                        command = new_command  # Update command for the next iteration of the retry loop
                        # Reset retries for the new command is implicitly handled by continuing the loop
                    else:
                        self.logger.error(
                            "LLM action 'command' but no valid command provided."
                        )
                        # Continue to retry the original failed command
                elif action_data["action"] == "patch":
                    self.logger.info("LLM suggested patching the code.")
                    patcher = Patcher(base_model=self.base_model)
                    patched_code_files = patcher.execute(
                        conversation=conversation_history,
                        code_markdown=code_markdown_context,
                        commands=current_commands,  # Pass commands that led to error
                        error=command_output,
                        system_os=os_platform,
                        project_name=project_name,
                    )
                    if patched_code_files:
                        patcher.save_code_to_project(patched_code_files, project_name)
                        self.project_manager.add_message_from_devika(
                            project_name,
                            "I've applied the patches. Let's try running the command again.",
                        )
                        # Code has changed, so update context for next potential LLM call
                        code_markdown_context = ReadCode(
                            project_name
                        ).code_set_to_markdown()
                        # Retry the same command after patching
                    else:
                        self.logger.error("Patcher failed to generate valid patches.")
                        self.project_manager.add_message_from_devika(
                            project_name,
                            "I tried to patch the code but encountered an issue. I'll try the command again.",
                        )
                        # Continue to retry the original failed command

        return True  # All commands (potentially after fixes) executed successfully

    def execute(
        self,
        conversation: List[str],
        code_markdown: str,
        os_system: str,
        project_path: str,
        project_name: str,
    ) -> Optional[List[str]]:
        """
        Execute the project running process.

        This involves:
        1. Rendering a prompt to get execution commands from the LLM.
        2. Parsing the LLM's response for these commands.
        3. Executing the commands sequentially using `_execute_commands_in_project`.
           This internal method handles errors, retries, and potential code patching.

        Args:
            conversation (List[str]): The conversation history.
            code_markdown (str): Markdown formatted string of the project's code.
            os_system (str): The operating system of the target environment.
            project_path (str): The absolute path to the project directory.
            project_name (str): The name of the project.

        Returns:
            Optional[List[str]]: The list of commands that were determined and attempted,
                                 or None if the initial command generation failed.
                                 The success of command execution is handled internally
                                 by `_execute_commands_in_project`.
        """
        if PROMPT_TEMPLATE == "Error: Runner prompt template not found.":
            logger.error("Cannot execute runner due to missing main prompt template.")
            return None

        rendered_prompt = self._render_initial_prompt(
            conversation, code_markdown, os_system
        )
        llm_response = self.llm.inference(rendered_prompt, project_name)
        parsed_commands_data = self._parse_commands_response(llm_response)

        if not parsed_commands_data or not parsed_commands_data["commands"]:
            logger.error("Failed to get valid execution commands from LLM.")
            self.project_manager.add_message_from_devika(
                project_name, "I couldn't determine the commands to run the project."
            )
            return None

        commands_to_run = parsed_commands_data["commands"]
        self.logger.info(f"LLM suggested run commands: {commands_to_run}")
        self.project_manager.add_message_from_devika(
            project_name,
            f"Okay, I will try to run the project using the following commands: `{'`, `'.join(commands_to_run)}`",
        )

        success = self._execute_commands_in_project(
            commands_to_run,
            project_path,
            project_name,
            conversation,
            code_markdown,
            os_system,
        )

        if success:
            self.project_manager.add_message_from_devika(
                project_name,
                "The project execution seems to have completed successfully.",
            )
        else:
            self.project_manager.add_message_from_devika(
                project_name,
                "There were issues during project execution that could not be fully resolved.",
            )

        return commands_to_run  # Return the commands that were attempted
