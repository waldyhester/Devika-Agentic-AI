"""
Manages the agent's dynamic state, including interaction history and status.

This module defines how the agent's state is stored and retrieved, primarily using
a SQLite database accessed via SQLModel. The state includes internal monologues,
browser session details, terminal session information, step-by-step progress,
messages, completion status, active status, and token usage.

The `AgentStateModel` class defines the database schema for storing project-specific
state stacks as JSON strings. The `AgentState` class provides methods to interact
with this stored state, such as creating new states, adding to the state stack,
retrieving current or latest states, and updating specific attributes like
agent activity, completion status, and token usage.
"""
import json
from datetime import datetime
from typing import Optional, Dict, Any, List, Union

from sqlalchemy.future.engine import Engine
from sqlmodel import Field, Session, SQLModel, create_engine

from src.socket_instance import emit_agent
from src.config import Config

# Define a type alias for the structured state dictionary.
# This helps in documenting and type-checking the state objects.
StateType = Dict[
    str,
    Union[
        Optional[str],
        Dict[str, Optional[str]],
        Dict[str, Optional[Union[str, List[str]]]],
        bool,
        int,
    ],
]


class AgentStateModel(SQLModel, table=True):
    """
    Database model for storing the agent's state stack for each project.

    Attributes:
        id (Optional[int]): The primary key for the database record.
        project (str): The name of the project this state belongs to. Unique per project.
        state_stack_json (str): A JSON string representing a list (stack) of agent states.
                                Each item in the list is a `StateType` dictionary.
    """

    __tablename__ = "agent_state"

    id: Optional[int] = Field(default=None, primary_key=True)
    project: str = Field(unique=True, index=True)
    state_stack_json: str


class AgentState:
    """
    Manages the agent's state, persisting it in a SQLite database.

    This class provides an interface to create, update, retrieve, and delete
    the state of an agent for various projects. The state is stored as a stack
    (a list of dictionaries) and serialized to JSON before database insertion.

    Attributes:
        engine (Engine): The SQLAlchemy engine instance for database interaction.
    """

    def __init__(self) -> None:
        """
        Initialize the AgentState manager.

        Sets up the database engine and ensures the `AgentStateModel` table
        is created in the database.
        """
        config = Config()
        sqlite_path: str = config.get_sqlite_db()
        self.engine: Engine = create_engine(f"sqlite:///{sqlite_path}")
        SQLModel.metadata.create_all(self.engine)

    def new_state(self) -> StateType:
        """
        Create a new, default state dictionary.

        This state dictionary includes placeholders for internal monologue,
        browser session, terminal session, current step, messages, completion status,
        active status, token usage, and a timestamp.

        Returns:
            StateType: A dictionary representing a new agent state.
                       Expected keys are:
                           - "internal_monologue": Optional[str]
                           - "browser_session": {"url": Optional[str], "screenshot": Optional[str]}
                           - "terminal_session": {"command": Optional[str], "output": Optional[str], "title": Optional[str]}
                           - "step": Optional[str]
                           - "message": Optional[str]
                           - "completed": bool
                           - "agent_is_active": bool
                           - "token_usage": int
                           - "timestamp": str
        """
        timestamp: str = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

        return {
            "internal_monologue": None,
            "browser_session": {"url": None, "screenshot": None},
            "terminal_session": {"command": None, "output": None, "title": None},
            "step": None,
            "message": None,
            "completed": False,
            "agent_is_active": True,
            "token_usage": 0,
            "timestamp": timestamp,
        }

    def delete_state(self, project: str) -> None:
        """
        Delete the entire state stack for a given project.

        Args:
            project (str): The name of the project whose state should be deleted.
        """
        with Session(self.engine) as session:
            agent_state_record = (
                session.query(AgentStateModel)
                .filter(AgentStateModel.project == project)
                .first()
            )
            if agent_state_record:
                session.delete(agent_state_record)
                session.commit()

    def add_to_current_state(self, project: str, state: StateType) -> None:
        """
        Add a new state to the project's state stack.

        If no state exists for the project, a new record is created. Otherwise,
        the new state is appended to the existing stack. The updated stack is
        then emitted via a socket event.

        Args:
            project (str): The name of the project.
            state (StateType): The state dictionary to add.
        """
        with Session(self.engine) as session:
            agent_state_record = (
                session.query(AgentStateModel)
                .filter(AgentStateModel.project == project)
                .first()
            )
            state_stack: List[StateType]
            if agent_state_record:
                state_stack = json.loads(agent_state_record.state_stack_json)
                state_stack.append(state)
                agent_state_record.state_stack_json = json.dumps(state_stack)
                session.commit()
            else:
                state_stack = [state]
                agent_state_record = AgentStateModel(
                    project=project, state_stack_json=json.dumps(state_stack)
                )
                session.add(agent_state_record)
                session.commit()
            emit_agent("agent-state", state_stack)

    def get_current_state(self, project: str) -> Optional[List[StateType]]:
        """
        Retrieve the entire state stack for a project.

        Args:
            project (str): The name of the project.

        Returns:
            Optional[List[StateType]]: A list of state dictionaries if found,
                                       otherwise None.
        """
        with Session(self.engine) as session:
            agent_state_record = (
                session.query(AgentStateModel)
                .filter(AgentStateModel.project == project)
                .first()
            )
            if agent_state_record:
                return json.loads(agent_state_record.state_stack_json)
            return None

    def update_latest_state(self, project: str, state: StateType) -> None:
        """
        Update the most recent state in the project's state stack.

        If no state exists, a new stack is created with the given state.
        The updated stack is emitted via a socket event.

        Args:
            project (str): The name of the project.
            state (StateType): The new state dictionary to replace the latest state.
        """
        with Session(self.engine) as session:
            agent_state_record = (
                session.query(AgentStateModel)
                .filter(AgentStateModel.project == project)
                .first()
            )
            state_stack: List[StateType]
            if agent_state_record:
                state_stack = json.loads(agent_state_record.state_stack_json)
                if state_stack:  # Ensure stack is not empty
                    state_stack[-1] = state
                else: # Should not happen if record exists, but handle defensively
                    state_stack = [state]
                agent_state_record.state_stack_json = json.dumps(state_stack)
                session.commit()
            else:
                state_stack = [state]
                agent_state_record = AgentStateModel(
                    project=project, state_stack_json=json.dumps(state_stack)
                )
                session.add(agent_state_record)
                session.commit()
            emit_agent("agent-state", state_stack)

    def get_latest_state(self, project: str) -> Optional[StateType]:
        """
        Retrieve the most recent state from the project's state stack.

        Args:
            project (str): The name of the project.

        Returns:
            Optional[StateType]: The latest state dictionary if found,
                                 otherwise None.
        """
        with Session(self.engine) as session:
            agent_state_record = (
                session.query(AgentStateModel)
                .filter(AgentStateModel.project == project)
                .first()
            )
            if agent_state_record:
                state_stack = json.loads(agent_state_record.state_stack_json)
                if state_stack:
                    return state_stack[-1]
            return None

    def set_agent_active(self, project: str, is_active: bool) -> None:
        """
        Set the 'agent_is_active' status in the latest state of a project.

        If no state exists, a new state is created with the specified active status.
        The updated state stack is emitted.

        Args:
            project (str): The name of the project.
            is_active (bool): The new active status for the agent.
        """
        with Session(self.engine) as session:
            agent_state_record = (
                session.query(AgentStateModel)
                .filter(AgentStateModel.project == project)
                .first()
            )
            state_stack: List[StateType]
            if agent_state_record:
                state_stack = json.loads(agent_state_record.state_stack_json)
                if state_stack:
                    state_stack[-1]["agent_is_active"] = is_active
                else: # Should not happen if record exists
                    new_s = self.new_state()
                    new_s["agent_is_active"] = is_active
                    state_stack = [new_s]
                agent_state_record.state_stack_json = json.dumps(state_stack)
                session.commit()
            else:
                state_stack = [self.new_state()]
                state_stack[-1]["agent_is_active"] = is_active
                agent_state_record = AgentStateModel(
                    project=project, state_stack_json=json.dumps(state_stack)
                )
                session.add(agent_state_record)
                session.commit()
            emit_agent("agent-state", state_stack)

    def is_agent_active(self, project: str) -> Optional[bool]:
        """
        Check if the agent for a project is currently marked as active.

        Args:
            project (str): The name of the project.

        Returns:
            Optional[bool]: The active status if found, otherwise None.
        """
        latest_state = self.get_latest_state(project)
        if latest_state:
            return latest_state.get("agent_is_active")  # type: ignore
        return None

    def set_agent_completed(self, project: str, is_completed: bool) -> None:
        """
        Set the 'completed' status in the latest state of a project.

        Also updates the internal monologue to reflect task completion.
        If no state exists, a new state is created. The updated stack is emitted.

        Args:
            project (str): The name of the project.
            is_completed (bool): The new completion status for the agent.
        """
        with Session(self.engine) as session:
            agent_state_record = (
                session.query(AgentStateModel)
                .filter(AgentStateModel.project == project)
                .first()
            )
            state_stack: List[StateType]
            if agent_state_record:
                state_stack = json.loads(agent_state_record.state_stack_json)
                if state_stack:
                    state_stack[-1]["internal_monologue"] = "Agent has completed the task."
                    state_stack[-1]["completed"] = is_completed
                else: # Should not happen
                    new_s = self.new_state()
                    new_s["internal_monologue"] = "Agent has completed the task."
                    new_s["completed"] = is_completed
                    state_stack = [new_s]

                agent_state_record.state_stack_json = json.dumps(state_stack)
                session.commit()
            else:
                state_stack = [self.new_state()]
                state_stack[-1]["internal_monologue"] = "Agent has completed the task."
                state_stack[-1]["completed"] = is_completed
                agent_state_record = AgentStateModel(
                    project=project, state_stack_json=json.dumps(state_stack)
                )
                session.add(agent_state_record)
                session.commit()
            emit_agent("agent-state", state_stack)

    def is_agent_completed(self, project: str) -> Optional[bool]:
        """
        Check if the agent for a project is marked as completed.

        Args:
            project (str): The name of the project.

        Returns:
            Optional[bool]: The completion status if found, otherwise None.
        """
        latest_state = self.get_latest_state(project)
        if latest_state:
            return latest_state.get("completed")  # type: ignore
        return None

    def update_token_usage(self, project: str, token_usage: int) -> None:
        """
        Update the 'token_usage' in the latest state of a project by adding to it.

        If no state exists, a new state is created with the given token usage.

        Args:
            project (str): The name of the project.
            token_usage (int): The number of tokens to add to the current usage.
        """
        with Session(self.engine) as session:
            agent_state_record = (
                session.query(AgentStateModel)
                .filter(AgentStateModel.project == project)
                .first()
            )
            state_stack: List[StateType]
            if agent_state_record:
                state_stack = json.loads(agent_state_record.state_stack_json)
                if state_stack:
                    current_tokens = state_stack[-1].get("token_usage", 0)
                    if isinstance(current_tokens, int): # Should always be int
                        state_stack[-1]["token_usage"] = current_tokens + token_usage
                    else: # Fallback if somehow not an int
                        state_stack[-1]["token_usage"] = token_usage
                else: # Should not happen
                    new_s = self.new_state()
                    new_s["token_usage"] = token_usage
                    state_stack = [new_s]
                agent_state_record.state_stack_json = json.dumps(state_stack)
                session.commit()
            else:
                state_stack = [self.new_state()]
                state_stack[-1]["token_usage"] = token_usage
                agent_state_record = AgentStateModel(
                    project=project, state_stack_json=json.dumps(state_stack)
                )
                session.add(agent_state_record)
                session.commit()
            # No emit_agent here as it's handled by LLM.update_global_token_usage

    def get_latest_token_usage(self, project: str) -> int:
        """
        Retrieve the latest total token usage for a project.

        Args:
            project (str): The name of the project.

        Returns:
            int: The total token usage, or 0 if no state is found.
        """
        latest_state = self.get_latest_state(project)
        if latest_state:
            usage = latest_state.get("token_usage")
            if isinstance(usage, int):
                return usage
        return 0
