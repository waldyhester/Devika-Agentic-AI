"""
Manages the application configuration using a TOML file.

This module provides a singleton class `Config` that loads configuration
from a `config.toml` file. If the file doesn't exist, it's created
by copying `sample.config.toml`. The class offers methods to get and set
various configuration parameters.
"""

import os
from typing import Any, Dict, Optional

import toml
from dotenv import load_dotenv


class Config:
    """
    Singleton class to manage application configuration.

    This class loads configuration settings from a TOML file (`config.toml`).
    It provides methods to access and modify these settings. If `config.toml`
    is not found, it initializes it by copying from `sample.config.toml`.
    """

    _instance: Optional["Config"] = None
    config: Dict[str, Any]

    def __new__(cls) -> "Config":
        """
        Create a new instance of Config if one doesn't exist, or return the existing one.

        Ensures that only one instance of the Config class is created (Singleton pattern).

        Returns:
            Config: The singleton instance of the Config class.
        """
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._load_config()
        return cls._instance

    def _load_config(self) -> None:
        """
        Load configuration from `config.toml`.

        If `config.toml` does not exist, it copies the content from
        `sample.config.toml` to create it. Then, it loads the configuration
        into the `self.config` attribute.
        """
        # If the config file doesn't exist, copy from the sample
        if not os.path.exists("config.toml"):
            with open("sample.config.toml", "r", encoding="utf-8") as f_in, open(
                "config.toml", "w", encoding="utf-8"
            ) as f_out:
                f_out.write(f_in.read())

        load_dotenv()  # Load environment variables from .env file
        self.config = toml.load("config.toml")

    def get_config(self) -> Dict[str, Any]:
        """
        Retrieve the entire configuration dictionary.

        Returns:
            Dict[str, Any]: The complete configuration dictionary.
        """
        return self.config

    def get_bing_api_endpoint(self) -> str:
        """
        Get the Bing API endpoint URL.

        Returns:
            str: The Bing API endpoint.
        """
        return str(self.config["API_ENDPOINTS"]["BING"])

    def get_bing_api_key(self) -> str:
        """
        Get the Bing API key.

        Returns:
            str: The Bing API key.
        """
        return str(self.config["API_KEYS"]["BING"])

    def get_google_search_api_key(self) -> str:
        """
        Get the Google Search API key.

        Returns:
            str: The Google Search API key.
        """
        return str(self.config["API_KEYS"]["GOOGLE_SEARCH"])

    def get_google_search_engine_id(self) -> str:
        """
        Get the Google Search engine ID.

        Returns:
            str: The Google Search engine ID.
        """
        return str(self.config["API_KEYS"]["GOOGLE_SEARCH_ENGINE_ID"])

    def get_google_search_api_endpoint(self) -> str:
        """
        Get the Google Search API endpoint URL.

        Returns:
            str: The Google Search API endpoint.
        """
        return str(self.config["API_ENDPOINTS"]["GOOGLE"])

    def get_ollama_api_endpoint(self) -> str:
        """
        Get the Ollama API endpoint URL.

        Returns:
            str: The Ollama API endpoint.
        """
        return str(self.config["API_ENDPOINTS"]["OLLAMA"])

    def get_claude_api_key(self) -> str:
        """
        Get the Claude API key.

        Returns:
            str: The Claude API key.
        """
        return str(self.config["API_KEYS"]["CLAUDE"])

    def get_openai_api_key(self) -> str:
        """
        Get the OpenAI API key.

        Returns:
            str: The OpenAI API key.
        """
        return str(self.config["API_KEYS"]["OPENAI"])

    def get_gemini_api_key(self) -> str:
        """
        Get the Gemini API key.

        Returns:
            str: The Gemini API key.
        """
        return str(self.config["API_KEYS"]["GEMINI"])

    def get_mistral_api_key(self) -> str:
        """
        Get the Mistral API key.

        Returns:
            str: The Mistral API key.
        """
        return str(self.config["API_KEYS"]["MISTRAL"])

    def get_groq_api_key(self) -> str:
        """
        Get the Groq API key.

        Returns:
            str: The Groq API key.
        """
        return str(self.config["API_KEYS"]["GROQ"])

    def get_netlify_api_key(self) -> str:
        """
        Get the Netlify API key.

        Returns:
            str: The Netlify API key.
        """
        return str(self.config["API_KEYS"]["NETLIFY"])

    def get_sqlite_db(self) -> str:
        """
        Get the SQLite database file path.

        Returns:
            str: The path to the SQLite database file.
        """
        return str(self.config["STORAGE"]["SQLITE_DB"])

    def get_screenshots_dir(self) -> str:
        """
        Get the directory for storing screenshots.

        Returns:
            str: The path to the screenshots directory.
        """
        return str(self.config["STORAGE"]["SCREENSHOTS_DIR"])

    def get_pdfs_dir(self) -> str:
        """
        Get the directory for storing PDFs.

        Returns:
            str: The path to the PDFs directory.
        """
        return str(self.config["STORAGE"]["PDFS_DIR"])

    def get_projects_dir(self) -> str:
        """
        Get the directory for storing projects.

        Returns:
            str: The path to the projects directory.
        """
        return str(self.config["STORAGE"]["PROJECTS_DIR"])

    def get_logs_dir(self) -> str:
        """
        Get the directory for storing logs.

        Returns:
            str: The path to the logs directory.
        """
        return str(self.config["STORAGE"]["LOGS_DIR"])

    def get_repos_dir(self) -> str:
        """
        Get the directory for storing repositories.

        Returns:
            str: The path to the repositories directory.
        """
        return str(self.config["STORAGE"]["REPOS_DIR"])

    def get_logging_rest_api(self) -> bool:
        """
        Check if logging to REST API is enabled.

        Returns:
            bool: True if REST API logging is enabled, False otherwise.
        """
        return self.config["LOGGING"]["LOG_REST_API"] == "true"

    def get_logging_prompts(self) -> bool:
        """
        Check if logging prompts is enabled.

        Returns:
            bool: True if prompt logging is enabled, False otherwise.
        """
        return self.config["LOGGING"]["LOG_PROMPTS"] == "true"

    # New getters for environment-configurable service parameters
    def get_github_api_base_url(self) -> str:
        return os.getenv("GITHUB_API_BASE_URL", "https://api.github.com")

    def get_github_default_user_agent(self) -> str:
        return os.getenv("GITHUB_DEFAULT_USER_AGENT", "DevikaAI/0.1")

    def get_github_default_timeout(self) -> int:
        return int(os.getenv("GITHUB_DEFAULT_TIMEOUT", "10"))

    def get_github_default_per_page(self) -> int:
        return int(os.getenv("GITHUB_DEFAULT_PER_PAGE", "30"))

    def get_git_executable_path(self) -> str:
        return os.getenv("GIT_EXECUTABLE_PATH", "git")

    def get_git_default_timeout(self) -> int:
        return int(os.getenv("GIT_DEFAULT_TIMEOUT", "300"))

    def get_firejail_executable_path(self) -> str:
        return os.getenv("FIREJAIL_EXECUTABLE_PATH", "firejail")

    def get_firejail_default_profile_path(self) -> Optional[str]:
        profile_path = os.getenv("FIREJAIL_DEFAULT_PROFILE_PATH")
        return profile_path if profile_path else None

    def get_browser_default_user_agent(self) -> str:
        return os.getenv("BROWSER_DEFAULT_USER_AGENT", "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36")

    def set_bing_api_key(self, key: str) -> None:
        """
        Set the Bing API key.

        Args:
            key (str): The Bing API key.
        """
        self.config["API_KEYS"]["BING"] = key
        self.save_config()

    def set_bing_api_endpoint(self, endpoint: str) -> None:
        """
        Set the Bing API endpoint URL.

        Args:
            endpoint (str): The Bing API endpoint URL.
        """
        self.config["API_ENDPOINTS"]["BING"] = endpoint
        self.save_config()

    def set_google_search_api_key(self, key: str) -> None:
        """
        Set the Google Search API key.

        Args:
            key (str): The Google Search API key.
        """
        self.config["API_KEYS"]["GOOGLE_SEARCH"] = key
        self.save_config()

    def set_google_search_engine_id(self, key: str) -> None:
        """
        Set the Google Search engine ID.

        Args:
            key (str): The Google Search engine ID.
        """
        self.config["API_KEYS"]["GOOGLE_SEARCH_ENGINE_ID"] = key
        self.save_config()

    def set_google_search_api_endpoint(self, endpoint: str) -> None:
        """
        Set the Google Search API endpoint URL.

        Args:
            endpoint (str): The Google Search API endpoint URL.
        """
        self.config["API_ENDPOINTS"]["GOOGLE_SEARCH"] = endpoint
        self.save_config()

    def set_ollama_api_endpoint(self, endpoint: str) -> None:
        """
        Set the Ollama API endpoint URL.

        Args:
            endpoint (str): The Ollama API endpoint URL.
        """
        self.config["API_ENDPOINTS"]["OLLAMA"] = endpoint
        self.save_config()

    def set_claude_api_key(self, key: str) -> None:
        """
        Set the Claude API key.

        Args:
            key (str): The Claude API key.
        """
        self.config["API_KEYS"]["CLAUDE"] = key
        self.save_config()

    def set_openai_api_key(self, key: str) -> None:
        """
        Set the OpenAI API key.

        Args:
            key (str): The OpenAI API key.
        """
        self.config["API_KEYS"]["OPENAI"] = key
        self.save_config()

    def set_gemini_api_key(self, key: str) -> None:
        """
        Set the Gemini API key.

        Args:
            key (str): The Gemini API key.
        """
        self.config["API_KEYS"]["GEMINI"] = key
        self.save_config()

    def set_mistral_api_key(self, key: str) -> None:
        """
        Set the Mistral API key.

        Args:
            key (str): The Mistral API key.
        """
        self.config["API_KEYS"]["MISTRAL"] = key
        self.save_config()

    def set_groq_api_key(self, key: str) -> None:
        """
        Set the Groq API key.

        Args:
            key (str): The Groq API key.
        """
        self.config["API_KEYS"]["GROQ"] = key
        self.save_config()

    def set_netlify_api_key(self, key: str) -> None:
        """
        Set the Netlify API key.

        Args:
            key (str): The Netlify API key.
        """
        self.config["API_KEYS"]["NETLIFY"] = key
        self.save_config()

    def set_sqlite_db(self, db_path: str) -> None:
        """
        Set the SQLite database file path.

        Args:
            db_path (str): The path to the SQLite database file.
        """
        self.config["STORAGE"]["SQLITE_DB"] = db_path
        self.save_config()

    def set_screenshots_dir(self, directory: str) -> None:
        """
        Set the directory for storing screenshots.

        Args:
            directory (str): The path to the screenshots directory.
        """
        self.config["STORAGE"]["SCREENSHOTS_DIR"] = directory
        self.save_config()

    def set_pdfs_dir(self, directory: str) -> None:
        """
        Set the directory for storing PDFs.

        Args:
            directory (str): The path to the PDFs directory.
        """
        self.config["STORAGE"]["PDFS_DIR"] = directory
        self.save_config()

    def set_projects_dir(self, directory: str) -> None:
        """
        Set the directory for storing projects.

        Args:
            directory (str): The path to the projects directory.
        """
        self.config["STORAGE"]["PROJECTS_DIR"] = directory
        self.save_config()

    def set_logs_dir(self, directory: str) -> None:
        """
        Set the directory for storing logs.

        Args:
            directory (str): The path to the logs directory.
        """
        self.config["STORAGE"]["LOGS_DIR"] = directory
        self.save_config()

    def set_repos_dir(self, directory: str) -> None:
        """
        Set the directory for storing repositories.

        Args:
            directory (str): The path to the repositories directory.
        """
        self.config["STORAGE"]["REPOS_DIR"] = directory
        self.save_config()

    def set_logging_rest_api(self, value: bool) -> None:
        """
        Enable or disable logging to REST API.

        Args:
            value (bool): True to enable, False to disable.
        """
        self.config["LOGGING"]["LOG_REST_API"] = "true" if value else "false"
        self.save_config()

    def set_logging_prompts(self, value: bool) -> None:
        """
        Enable or disable logging prompts.

        Args:
            value (bool): True to enable, False to disable.
        """
        self.config["LOGGING"]["LOG_PROMPTS"] = "true" if value else "false"
        self.save_config()

    def save_config(self) -> None:
        """
        Save the current configuration to `config.toml`.

        Writes the `self.config` dictionary to the `config.toml` file.
        """
        with open("config.toml", "w", encoding="utf-8") as f:
            toml.dump(self.config, f)
