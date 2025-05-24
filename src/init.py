"""
Handles initial setup tasks for the Devika application.

This module includes the `init_devika` function which is responsible for
creating necessary directories for application data (logs, databases, projects, etc.)
and pre-loading machine learning models like SentenceBERT to ensure they are
ready for use when the application starts. This helps in smoother first-time
operation and identifies potential issues early.
"""

import os
from typing import List, Tuple  # Tuple might be needed if extract_keywords returns it

from src.bert.sentence import SentenceBert  # Moved to top-level imports
from src.config import Config
from src.logger import Logger

# Initialize logger here if it's to be used at module level,
# or ensure it's properly initialized before this script runs if imported.
# For simplicity, assuming Logger() can be called directly as it is.
logger = Logger()


def init_devika() -> None:
    """
    Initializes the Devika application environment.

    This function performs the following setup tasks:
    1.  Retrieves directory paths from the application configuration.
    2.  Creates all necessary directories if they do not already exist,
        including directories for SQLite database, screenshots, PDFs,
        projects, and logs. Includes error handling for directory creation.
    3.  Pre-loads the SentenceBERT model by performing a dummy keyword
        extraction. This helps in downloading/caching the model on first run
        and ensures it's ready for subsequent operations. Includes error
        handling for model loading.

    Returns:
        None
    """
    config: Config = Config()

    logger.info("Initializing Devika environment...")

    # Define all required directory paths
    # The parent of sqlite_db needs to exist for create_engine
    sqlite_db_path: str = config.get_sqlite_db()
    sqlite_db_dir: str = (
        os.path.dirname(sqlite_db_path) if sqlite_db_path else ""
    )  # Handle if path is empty

    dir_paths_to_create: List[str] = [
        sqlite_db_dir,  # Ensure parent of DB file exists
        config.get_screenshots_dir(),
        config.get_pdfs_dir(),
        config.get_projects_dir(),
        config.get_logs_dir(),  # Logger might create this, but good to ensure
    ]
    # Filter out empty paths that might result from empty config values
    dir_paths_to_create = [path for path in dir_paths_to_create if path]

    logger.info("Creating required directories...")
    for dir_path in dir_paths_to_create:
        try:
            os.makedirs(dir_path, exist_ok=True)
            logger.info(f"Ensured directory exists: {dir_path}")
        except OSError as e:
            logger.error(f"Error creating directory {dir_path}: {e}", exc_info=True)
            # Depending on the severity, one might choose to raise the exception
            # or exit the application if a critical directory cannot be created.
        except Exception as e:
            logger.error(
                f"Unexpected error creating directory {dir_path}: {e}", exc_info=True
            )

    logger.info("Pre-loading sentence-transformer BERT models...")
    try:
        # A simple, short prompt for pre-loading.
        # The content doesn't matter as much as triggering the model download/load.
        test_prompt: str = "Initialize keyword extraction model."
        # The result of extract_keywords is List[Tuple[str, float]]
        keywords: List[Tuple[str, float]] = SentenceBert(test_prompt).extract_keywords()
        if keywords:  # Check if keywords were extracted (model likely loaded)
            logger.info(
                f"BERT model loaded successfully. Test keywords: {keywords[:2]}"
            )
        else:
            logger.info(
                "BERT model loaded, but test keyword extraction returned empty (might be normal for short prompt)."
            )
    except ImportError as e:
        logger.error(
            f"Failed to import SentenceBert or its dependencies. "
            f"Ensure 'keybert' and 'sentence-transformers' are installed: {e}",
            exc_info=True,
        )
    except Exception as e:
        logger.error(
            f"An error occurred during BERT model pre-loading or test extraction: {e}",
            exc_info=True,
        )
        # Depending on how critical BERT is at startup, you might want to
        # re-raise this or handle it more gracefully.

    logger.info("Devika initialization process completed.")
