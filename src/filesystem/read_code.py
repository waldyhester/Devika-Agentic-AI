"""
Handles reading and formatting code files from a project directory.

This module provides the `ReadCode` class, which can walk through a project
directory, read the content of specified file types (or skip ignored ones),
and format the collected code into a single Markdown string.
"""
import os
from typing import List, Dict, Set

from src.config import Config
from src.logger import Logger

logger = Logger()

# Common directories and files to ignore during code reading
DEFAULT_IGNORE_DIRECTORIES: Set[str] = {
    ".git",
    "__pycache__",
    ".venv",
    "venv",
    "node_modules",
    ".vscode",
    ".idea",
    "build",
    "dist",
    "*.egg-info",
}
DEFAULT_IGNORE_FILES: Set[str] = {
    ".DS_Store",
    "*.pyc",
    "*.log",
    "*.swp",
    "*.swo",
}
# For now, we read most text-based files and try to infer language for markdown
# A more sophisticated approach might involve a list of specifically included extensions.


class ReadCode:
    """
    Reads code files from a specified project directory and can format them into Markdown.

    Attributes:
        directory_path (str): The absolute path to the project's root directory.
        ignore_directories (Set[str]): A set of directory names to ignore.
        ignore_files (Set[str]): A set of file names/patterns to ignore.
    """

    def __init__(
        self,
        project_name: str,
        ignore_directories: Optional[Set[str]] = None,
        ignore_files: Optional[Set[str]] = None,
    ) -> None:
        """
        Initialize the ReadCode class.

        Args:
            project_name (str): The name of the project. The actual directory path
                                will be derived from this name and application config.
            ignore_directories (Optional[Set[str]]): A set of directory names to ignore.
                                                     Defaults to `DEFAULT_IGNORE_DIRECTORIES`.
            ignore_files (Optional[Set[str]]): A set of file names/patterns to ignore.
                                               Defaults to `DEFAULT_IGNORE_FILES`.

        Raises:
            FileNotFoundError: If the derived project directory does not exist.
        """
        config = Config()
        project_base_path: str = config.get_projects_dir()
        project_slug = project_name.lower().replace(" ", "-")
        self.directory_path: str = os.path.join(project_base_path, project_slug)

        if not os.path.isdir(self.directory_path):
            logger.error(f"Project directory not found: {self.directory_path}")
            # Consider whether to raise an error or allow an empty ReadCode object
            # For now, let's log and proceed, read_directory will return empty.
            # raise FileNotFoundError(f"Project directory not found: {self.directory_path}")

        self.ignore_directories: Set[str] = (
            ignore_directories if ignore_directories is not None else DEFAULT_IGNORE_DIRECTORIES
        )
        self.ignore_files: Set[str] = (
            ignore_files if ignore_files is not None else DEFAULT_IGNORE_FILES
        )
        logger.info(f"ReadCode initialized for project: {project_name} at {self.directory_path}")


    def read_directory(self) -> List[Dict[str, str]]:
        """
        Recursively reads files from the project directory, skipping ignored ones.

        Returns:
            List[Dict[str, str]]: A list of dictionaries, where each dictionary
                                  has "filename" (relative path to the project root)
                                  and "code" (file content) keys.

        Raises:
            FileNotFoundError: If the project directory does not exist (though __init__ checks this).
        """
        files_list: List[Dict[str, str]] = []
        
        if not os.path.isdir(self.directory_path):
            logger.warning(f"Cannot read directory, path does not exist or is not a directory: {self.directory_path}")
            return files_list

        for root, dirs, files in os.walk(self.directory_path, topdown=True):
            # Filter out ignored directories
            dirs[:] = [d for d in dirs if d not in self.ignore_directories and not d.startswith('.')]

            for file_name in files:
                if file_name in self.ignore_files or file_name.endswith(tuple(self.ignore_files)): # Basic pattern matching for extensions
                    continue

                file_path: str = os.path.join(root, file_name)
                relative_file_path: str = os.path.relpath(file_path, self.directory_path)

                try:
                    with open(file_path, "r", encoding="utf-8", errors="ignore") as file_content:
                        # errors='ignore' is a fallback for non-UTF-8 files, ideally identify binary files better
                        code = file_content.read()
                        files_list.append({"filename": relative_file_path, "code": code})
                        logger.debug(f"Read file: {relative_file_path}")
                except IOError as e:
                    logger.error(f"Could not read file {file_path}: {e}")
                except Exception as e:
                    logger.error(f"An unexpected error occurred while reading file {file_path}: {e}")
        
        logger.info(f"Read {len(files_list)} files from directory: {self.directory_path}")
        return files_list

    def code_set_to_markdown(self) -> str:
        """
        Converts the collected code files into a single Markdown string.

        Each file's content is placed in a Markdown code block, with the
        language inferred from the file extension for syntax highlighting.

        Returns:
            str: A string containing all code files formatted in Markdown.
        """
        code_set = self.read_directory()
        markdown_parts: List[str] = []
        for code_item in code_set:
            file_name = code_item["filename"]
            code_content = code_item["code"]
            
            # Infer language from file extension for Markdown code block
            language_map = {
                ".py": "python", ".js": "javascript", ".ts": "typescript",
                ".java": "java", ".c": "c", ".cpp": "cpp", ".cs": "csharp",
                ".html": "html", ".css": "css", ".scss": "scss", ".json": "json",
                ".xml": "xml", ".md": "markdown", ".sh": "shell", ".rb": "ruby",
                ".go": "go", ".php": "php", ".swift": "swift", ".kt": "kotlin",
                ".rs": "rust", ".toml": "toml", ".yaml": "yaml", ".yml": "yaml",
                ".dockerfile": "dockerfile", "dockerfile": "dockerfile",
                "requirements.txt": "text", ".txt": "text"
            }
            
            # Handle full filename matches first (e.g. Dockerfile)
            lang_indicator = language_map.get(file_name.lower())
            if not lang_indicator:
                _root, ext = os.path.splitext(file_name)
                lang_indicator = language_map.get(ext.lower(), "") # Fallback to empty for unknown


            markdown_parts.append(f"### File: `{file_name}`\n\n")
            markdown_parts.append(f"```{lang_indicator}\n{code_content}\n```\n\n")
            markdown_parts.append("---\n\n")
        
        if not markdown_parts:
            return "No code files found or read from the project directory."
            
        return "".join(markdown_parts)
