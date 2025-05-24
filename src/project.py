"""
Manages projects, including their creation, message history, and file operations.

This module provides the `ProjectManager` class to handle project-related
functionalities such as creating new projects, storing and retrieving message
histories associated with projects, and packaging project files into ZIP archives.
It uses SQLModel for database interactions to persist project metadata and message
stacks.
"""
import os
import json
import zipfile
from datetime import datetime
from typing import Optional, List, Dict, TypedDict, Any

from sqlalchemy.future.engine import Engine
from sqlmodel import Field, Session, SQLModel, create_engine

from src.socket_instance import emit_agent
from src.config import Config


class MessageDict(TypedDict):
    """
    Represents the structure of a message dictionary.

    Attributes:
        from_devika (bool): True if the message is from Devika, False if from the user.
        message (Optional[str]): The content of the message.
        timestamp (str): The timestamp of when the message was created.
    """

    from_devika: bool
    message: Optional[str]
    timestamp: str


class Projects(SQLModel, table=True):
    """
    Database model for storing project metadata and message history.

    Attributes:
        id (Optional[int]): The primary key for the database record.
        project (str): The name of the project. This should be unique.
        message_stack_json (str): A JSON string representing a list of `MessageDict`
                                  objects, storing the conversation history for the project.
    """

    id: Optional[int] = Field(default=None, primary_key=True)
    project: str = Field(unique=True, index=True)
    message_stack_json: str


class ProjectManager:
    """
    Manages project creation, message history, and file operations.

    This class interacts with a SQLite database via SQLModel to store and retrieve
    project information and message stacks. It also handles filesystem operations
    for projects, such as creating project directories and zipping project files.

    Attributes:
        project_path (str): The base directory where project files are stored.
        engine (Engine): The SQLAlchemy engine instance for database interaction.
    """

    def __init__(self) -> None:
        """
        Initialize the ProjectManager.

        Sets up the database engine, creates necessary tables if they don't exist,
        and initializes the base path for project storage from the application config.
        """
        config = Config()
        sqlite_path: str = config.get_sqlite_db()
        self.project_path: str = config.get_projects_dir()
        self.engine: Engine = create_engine(f"sqlite:///{sqlite_path}")
        SQLModel.metadata.create_all(self.engine)

    def new_message(self) -> MessageDict:
        """
        Create a new message dictionary with default values.

        Defaults to a message from Devika with a current timestamp.

        Returns:
            MessageDict: A new message dictionary.
        """
        timestamp: str = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        return MessageDict(
            from_devika=True, message=None, timestamp=timestamp
        )

    def create_project(self, project: str) -> None:
        """
        Create a new project record in the database.

        Initializes the project with an empty message stack.

        Args:
            project (str): The name of the project to create.
        """
        with Session(self.engine) as session:
            project_state = Projects(project=project, message_stack_json=json.dumps([]))
            session.add(project_state)
            session.commit()

    def delete_project(self, project: str) -> None:
        """
        Delete a project record from the database.

        Args:
            project (str): The name of the project to delete.
        """
        with Session(self.engine) as session:
            project_state = (
                session.query(Projects).filter(Projects.project == project).first()
            )
            if project_state:
                session.delete(project_state)
                session.commit()

    def add_message_to_project(self, project: str, message: MessageDict) -> None:
        """
        Add a message to a specific project's message stack.

        If the project doesn't exist, it creates a new one. The message is appended
        to the existing stack or a new stack if one doesn't exist.

        Args:
            project (str): The name of the project.
            message (MessageDict): The message dictionary to add.
        """
        with Session(self.engine) as session:
            project_record = (
                session.query(Projects).filter(Projects.project == project).first()
            )
            message_stack: List[MessageDict]
            if project_record:
                message_stack = json.loads(project_record.message_stack_json)
                message_stack.append(message)
                project_record.message_stack_json = json.dumps(message_stack)
                session.commit()
            else:
                message_stack = [message]
                project_record = Projects(
                    project=project, message_stack_json=json.dumps(message_stack)
                )
                session.add(project_record)
                session.commit()

    def add_message_from_devika(self, project: str, message_content: str) -> None:
        """
        Add a message from Devika to the project's message stack.

        A new message dictionary is created, marked as from Devika, and then added.
        A socket event is emitted with the new message.

        Args:
            project (str): The name of the project.
            message_content (str): The content of the message from Devika.
        """
        new_msg: MessageDict = self.new_message()
        new_msg["message"] = message_content
        emit_agent("server-message", {"messages": new_msg})
        self.add_message_to_project(project, new_msg)

    def add_message_from_user(self, project: str, message_content: str) -> None:
        """
        Add a message from the user to the project's message stack.

        A new message dictionary is created, marked as from the user, and then added.
        A socket event is emitted with the new message.

        Args:
            project (str): The name of the project.
            message_content (str): The content of the message from the user.
        """
        new_msg: MessageDict = self.new_message()
        new_msg["message"] = message_content
        new_msg["from_devika"] = False
        emit_agent("server-message", {"messages": new_msg})
        self.add_message_to_project(project, new_msg)

    def get_messages(self, project: str) -> Optional[List[MessageDict]]:
        """
        Retrieve all messages for a specific project.

        Args:
            project (str): The name of the project.

        Returns:
            Optional[List[MessageDict]]: A list of message dictionaries if the project
                                         is found, otherwise None.
        """
        with Session(self.engine) as session:
            project_record = (
                session.query(Projects).filter(Projects.project == project).first()
            )
            if project_record:
                return json.loads(project_record.message_stack_json)
            return None

    def get_latest_message_from_user(self, project: str) -> Optional[MessageDict]:
        """
        Retrieve the most recent message from the user for a specific project.

        Args:
            project (str): The name of the project.

        Returns:
            Optional[MessageDict]: The latest user message dictionary if found,
                                   otherwise None.
        """
        messages = self.get_messages(project)
        if messages:
            for message in reversed(messages):
                if not message["from_devika"]:
                    return message
        return None

    def validate_last_message_is_from_user(self, project: str) -> bool:
        """
        Check if the last message in the project's history is from the user.

        Args:
            project (str): The name of the project.

        Returns:
            bool: True if the last message is from the user, False otherwise (including
                  if there are no messages or the project doesn't exist).
        """
        messages = self.get_messages(project)
        if messages:
            return not messages[-1]["from_devika"]
        return False

    def get_latest_message_from_devika(self, project: str) -> Optional[MessageDict]:
        """
        Retrieve the most recent message from Devika for a specific project.

        Args:
            project (str): The name of the project.

        Returns:
            Optional[MessageDict]: The latest Devika message dictionary if found,
                                   otherwise None.
        """
        messages = self.get_messages(project)
        if messages:
            for message in reversed(messages):
                if message["from_devika"]:
                    return message
        return None

    def get_project_list(self) -> List[str]:
        """
        Retrieve a list of all project names.

        Returns:
            List[str]: A list containing the names of all projects.
        """
        with Session(self.engine) as session:
            project_records = session.query(Projects).all()
            return [record.project for record in project_records]

    def get_all_messages_formatted(self, project: str) -> List[str]:
        """
        Retrieve all messages for a project, formatted as strings with sender prefix.

        Args:
            project (str): The name of the project.

        Returns:
            List[str]: A list of formatted message strings (e.g., "Devika: Hello", "User: Hi").
        """
        formatted_messages: List[str] = []
        messages = self.get_messages(project)
        if messages:
            for message in messages:
                sender = "Devika" if message["from_devika"] else "User"
                formatted_messages.append(f"{sender}: {message['message']}")
        return formatted_messages

    def get_project_path(self, project: str) -> str:
        """
        Get the absolute filesystem path for a given project name.

        Project names are lowercased and spaces are replaced with hyphens
        to form the directory name.

        Args:
            project (str): The name of the project.

        Returns:
            str: The absolute path to the project's directory.
        """
        project_dir_name = project.lower().replace(" ", "-")
        return os.path.join(self.project_path, project_dir_name)

    def project_to_zip(self, project: str) -> str:
        """
        Create a ZIP archive of the specified project's directory.

        The ZIP file will be named after the project and stored in the same
        parent directory as the project directory itself.

        Args:
            project (str): The name of the project to zip.

        Returns:
            str: The absolute path to the created ZIP file.

        Raises:
            FileNotFoundError: If the project path does not exist.
        """
        project_fs_path = self.get_project_path(project)
        if not os.path.isdir(project_fs_path):
            raise FileNotFoundError(f"Project directory not found: {project_fs_path}")

        zip_path = f"{project_fs_path}.zip"

        with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as zipf:
            for root, _, files in os.walk(project_fs_path):
                for file_name in files:
                    file_path = os.path.join(root, file_name)
                    # Create an arcname that is relative to the project_fs_path
                    # e.g., if project_fs_path is /data/projects/my-proj
                    # and file_path is /data/projects/my-proj/src/main.py
                    # arcname will be my-proj/src/main.py
                    arcname = os.path.relpath(file_path, os.path.dirname(project_fs_path))
                    zipf.write(file_path, arcname=arcname)
        return zip_path

    def get_zip_path(self, project: str) -> str:
        """
        Get the expected absolute path for a project's ZIP file.

        Note: This method does not check if the ZIP file actually exists.

        Args:
            project (str): The name of the project.

        Returns:
            str: The absolute path where the ZIP file for the project would be located.
        """
        project_fs_path = self.get_project_path(project)
        return f"{project_fs_path}.zip"
