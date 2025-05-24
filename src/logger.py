"""
Provides logging capabilities for the Devika application using Python's standard logging.

This module defines a `Logger` class that configures and provides access to a
standard Python logger instance. It sets up handlers for console and file logging,
with configurable log levels and formats. It also includes a decorator
`route_logger` for logging Flask route access.
"""

import logging
import os
from functools import wraps
from typing import Any, Callable, Dict, Optional

from flask import request  # Assuming Flask is used for route_logger
from werkzeug.wrappers import (
    Response as WerkzeugResponse,
)  # For type checking in route_logger

from src.config import Config

# Default logger configuration
DEFAULT_LOGGER_NAME = "devika"
DEFAULT_LOG_FILENAME = "devika_agent.log"
DEFAULT_LOG_FORMAT = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
DEFAULT_LOG_LEVEL = "INFO"  # Default level if not set in config

# Cache for configured loggers to prevent duplicate handler setup
_configured_loggers: Dict[str, logging.Logger] = {}


class Logger:
    """
    A wrapper class for Python's standard logging module.

    This class configures a logger with handlers for console and file output.
    It is intended to be instantiated where needed, but all instances with the
    same logger name will share the same underlying logging configuration due to
    how `logging.getLogger()` works.

    Attributes:
        logger (logging.Logger): The underlying Python logger instance.
        log_file_path (str): The full path to the log file.
    """

    def __init__(
        self,
        logger_name: str = DEFAULT_LOGGER_NAME,
        filename: str = DEFAULT_LOG_FILENAME,
    ) -> None:
        """
        Initialize the Logger and configure the underlying Python logger.

        If a logger with the given `logger_name` has already been configured by
        this class, its existing configuration will be used. Otherwise, a new
        logger is configured with console and file handlers.

        Args:
            logger_name (str): The name for the logger. Defaults to "devika".
            filename (str): The name of the log file. Defaults to "devika_agent.log".
        """
        self.logger: logging.Logger = logging.getLogger(logger_name)
        config = Config()

        # Configure logger only if it hasn't been configured by this class before
        if logger_name not in _configured_loggers:
            log_level_str: str = (
                config.get_log_level() or DEFAULT_LOG_LEVEL
            )  # Assuming get_log_level in Config
            numeric_log_level: int = getattr(
                logging, log_level_str.upper(), logging.INFO
            )
            self.logger.setLevel(numeric_log_level)

            # Create formatter
            formatter = logging.Formatter(DEFAULT_LOG_FORMAT)

            # Console Handler
            if (
                config.get_console_logging_enabled()
            ):  # Assuming this config option exists
                console_handler = logging.StreamHandler()
                console_handler.setFormatter(formatter)
                self.logger.addHandler(console_handler)

            # File Handler
            if config.get_file_logging_enabled():  # Assuming this config option exists
                logs_dir: str = config.get_logs_dir()
                os.makedirs(logs_dir, exist_ok=True)
                self.log_file_path: str = os.path.join(logs_dir, filename)

                file_handler = logging.FileHandler(self.log_file_path, encoding="utf-8")
                file_handler.setFormatter(formatter)
                self.logger.addHandler(file_handler)
            else:
                self.log_file_path = ""  # No file path if file logging is disabled

            _configured_loggers[logger_name] = self.logger
            self.logger.info(
                f"Logger '{logger_name}' configured. Level: {log_level_str}."
            )
        else:
            # If already configured, retrieve the path from one of its file handlers
            self.log_file_path = ""
            for handler in self.logger.handlers:
                if isinstance(handler, logging.FileHandler) and handler.baseFilename:
                    self.log_file_path = handler.baseFilename
                    break
            self.logger.debug(f"Logger '{logger_name}' already configured.")

    def read_log_file(self) -> Optional[str]:
        """
        Read the content of the configured log file.

        Returns:
            Optional[str]: The content of the log file, or None if file logging
                           is disabled, the file doesn't exist, or an error occurs.
        """
        if not self.log_file_path:
            self.logger.warning(
                "File logging is not enabled or file path not set; cannot read log file."
            )
            return None
        try:
            with open(self.log_file_path, "r", encoding="utf-8") as file:
                return file.read()
        except FileNotFoundError:
            self.logger.error(f"Log file not found at: {self.log_file_path}")
            return None
        except IOError as e:
            self.logger.error(f"Error reading log file {self.log_file_path}: {e}")
            return None

    def info(self, message: str, *args: Any, **kwargs: Any) -> None:
        """Log an informational message."""
        self.logger.info(message, *args, **kwargs)

    def error(self, message: str, *args: Any, **kwargs: Any) -> None:
        """Log an error message. Consider using `exception()` for errors with exceptions."""
        self.logger.error(message, *args, **kwargs)

    def warning(self, message: str, *args: Any, **kwargs: Any) -> None:
        """Log a warning message."""
        self.logger.warning(message, *args, **kwargs)

    def debug(self, message: str, *args: Any, **kwargs: Any) -> None:
        """Log a debug message."""
        self.logger.debug(message, *args, **kwargs)

    def exception(self, message: str, *args: Any, **kwargs: Any) -> None:
        """
        Log an error message with exception information (stack trace).
        Should be called from an exception handler.
        """
        kwargs.setdefault("exc_info", True)
        self.logger.error(message, *args, **kwargs)

    def critical(self, message: str, *args: Any, **kwargs: Any) -> None:
        """Log a critical error message."""
        self.logger.critical(message, *args, **kwargs)


def route_logger(logger_instance: Logger) -> Callable:
    """
    Decorator factory that creates a decorator to log Flask route entry and exit points.

    This decorator uses the provided `Logger` instance to log information about
    incoming requests and outgoing responses. Logging is controlled by the
    `LOG_REST_API` setting in the application configuration.

    Args:
        logger_instance (Logger): The logger instance to use for logging.

    Returns:
        Callable: A decorator function.
    """
    config = Config()
    log_enabled: bool = config.get_logging_rest_api()

    def decorator(func: Callable) -> Callable:
        @wraps(func)
        def wrapper(*args: Any, **kwargs: Any) -> Any:
            if log_enabled:
                # Log entry point
                try:
                    logger_instance.debug(
                        f"Request: {request.method} {request.path} - "
                        f"Headers: {dict(request.headers)} - "
                        f"Body: {request.get_data(as_text=True)[:200]}..."  # Log first 200 chars of body
                    )
                except Exception as e:
                    logger_instance.exception(
                        f"Error logging request details for {request.path}: {e}"
                    )

            # Call the actual route function
            response = func(*args, **kwargs)

            if log_enabled:
                # Log exit point, including response summary if possible
                try:
                    if (
                        isinstance(response, WerkzeugResponse)
                        and response.direct_passthrough
                    ):
                        logger_instance.debug(
                            f"Response: {request.method} {request.path} - Status: {response.status_code} - Type: File response"
                        )
                    elif isinstance(response, tuple) and isinstance(
                        response[0], WerkzeugResponse
                    ):  # Handle (response, status_code)
                        res_obj = response[0]
                        response_summary = res_obj.get_data(as_text=True)[
                            :200
                        ]  # Log first 200 chars
                        logger_instance.debug(
                            f"Response: {request.method} {request.path} - Status: {res_obj.status_code} - Body: {response_summary}..."
                        )
                    elif isinstance(response, WerkzeugResponse):
                        response_summary = response.get_data(as_text=True)[
                            :200
                        ]  # Log first 200 chars
                        logger_instance.debug(
                            f"Response: {request.method} {request.path} - Status: {response.status_code} - Body: {response_summary}..."
                        )
                    else:  # For non-Response objects, e.g. direct string or dict returns from Flask routes
                        logger_instance.debug(
                            f"Response: {request.method} {request.path} - Body: {str(response)[:200]}..."
                        )
                except Exception as e:
                    logger_instance.exception(
                        f"Error logging response details for {request.path}: {e}"
                    )
            return response

        return wrapper

    return decorator
