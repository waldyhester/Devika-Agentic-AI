"""
Provides functionalities for interacting with the GitHub API.

This module defines the `GitHubService` class, which allows for operations such as
listing user repositories by making authenticated requests to the GitHub API.
It uses the `requests` library for HTTP communication and handles API responses,
including pagination and error conditions.
"""
import os
from typing import List, Optional, TypedDict, Dict, Any

import requests
from requests.exceptions import RequestException, HTTPError

from src.config import Config
from src.logger import Logger

logger = Logger()

GITHUB_API_BASE_URL = "https://api.github.com"
DEFAULT_USER_AGENT = "DevikaAI/0.1" # Recommended by GitHub
DEFAULT_TIMEOUT = 10  # seconds
DEFAULT_PER_PAGE = 30 # Default items per page for paginated requests


class RepositoryDict(TypedDict, total=False):
    """
    Represents the structure of a GitHub repository object (subset of fields).

    Attributes:
        id (int): The unique identifier of the repository.
        name (str): The name of the repository.
        full_name (str): The full name of the repository (e.g., "owner/repo").
        html_url (str): The URL to the repository on GitHub.
        description (Optional[str]): A description of the repository.
        private (bool): Whether the repository is private.
        fork (bool): Whether the repository is a fork.
        url (str): The API URL for the repository.
        # Add other relevant fields as needed
    """
    id: int
    name: str
    full_name: str
    html_url: str
    description: Optional[str]
    private: bool
    fork: bool
    url: str


class GitHubService:
    """
    Service class for interacting with the GitHub API.

    Handles authentication, request construction, response parsing, and error handling
    for GitHub API operations.

    Attributes:
        token (Optional[str]): The GitHub Personal Access Token for authentication.
        session (requests.Session): A requests session with pre-configured headers.
    """

    def __init__(self, token: Optional[str] = None) -> None:
        """
        Initialize the GitHubService.

        Args:
            token (Optional[str]): A GitHub Personal Access Token. If not provided,
                                   it attempts to fetch one from the application config.
        """
        config = Config()
        self.token: Optional[str] = token or config.get_github_api_key() # Assuming a new config method
        
        self.session = requests.Session()
        self.session.headers["User-Agent"] = DEFAULT_USER_AGENT
        self.session.headers["Accept"] = "application/vnd.github.v3+json"
        if self.token:
            self.session.headers["Authorization"] = f"token {self.token}"
            logger.info("GitHubService initialized with API token.")
        else:
            logger.warning(
                "GitHubService initialized without an API token. "
                "Access to private repositories and some actions will be restricted."
            )

    def _log_rate_limit_info(self, response: requests.Response) -> None:
        """
        Log rate limit information from GitHub API response headers.

        Args:
            response (requests.Response): The HTTP response object from GitHub.
        """
        limit = response.headers.get("X-RateLimit-Limit")
        remaining = response.headers.get("X-RateLimit-Remaining")
        reset_timestamp = response.headers.get("X-RateLimit-Reset")
        if limit and remaining and reset_timestamp:
            reset_time_str = ""
            try:
                # Convert Unix timestamp to human-readable format
                from datetime import datetime
                reset_time = datetime.fromtimestamp(int(reset_timestamp))
                reset_time_str = reset_time.isoformat()
            except ValueError: # pragma: no cover
                pass # If timestamp is invalid, just log the raw value
            
            logger.debug(
                f"GitHub API Rate Limit: {remaining}/{limit} requests remaining. "
                f"Resets at: {reset_time_str or reset_timestamp}."
            )

    def get_repositories(
        self, per_page: int = DEFAULT_PER_PAGE, max_pages: Optional[int] = None
    ) -> Optional[List[RepositoryDict]]:
        """
        Fetch the list of repositories for the authenticated user.

        Handles pagination to retrieve all repositories if `max_pages` is not set,
        or up to `max_pages` pages of results.

        Args:
            per_page (int): Number of items to return per page (max 100).
            max_pages (Optional[int]): Maximum number of pages to fetch. If None, fetches all.

        Returns:
            Optional[List[RepositoryDict]]: A list of repository dictionaries,
                                            or None if an error occurs.
        """
        if not self.token:
            logger.error("Cannot fetch repositories without a GitHub API token.")
            return None

        all_repos: List[RepositoryDict] = []
        page = 1
        url = f"{GITHUB_API_BASE_URL}/user/repos"
        
        while True:
            params = {"per_page": per_page, "page": page, "sort": "updated"}
            logger.debug(f"Fetching repositories page {page} with {per_page} items per page.")
            
            try:
                response = self.session.get(url, params=params, timeout=DEFAULT_TIMEOUT)
                self._log_rate_limit_info(response)
                response.raise_for_status()  # Raises HTTPError for bad responses
                
                current_page_repos = response.json()
                if not isinstance(current_page_repos, list):
                    logger.error(f"Unexpected response format from GitHub API: {current_page_repos}")
                    break 
                
                all_repos.extend(current_page_repos) # type: ignore

                # Check for next page using 'Link' header
                if 'next' not in response.links or (max_pages and page >= max_pages):
                    break 
                page += 1
                # Some APIs might return the next page URL directly in response.links['next']['url']
                # but /user/repos uses simple page increment.

            except HTTPError as e:
                logger.error(f"HTTP error fetching repositories (page {page}): {e.response.status_code} - {e.response.text}")
                return None # Or return partial results: all_repos
            except RequestException as e:
                logger.error(f"Request error fetching repositories (page {page}): {e}")
                return None # Or return partial results: all_repos
            except ValueError as e: # JSONDecodeError inherits from ValueError
                logger.error(f"Error decoding JSON response for repositories (page {page}): {e}")
                return None # Or return partial results: all_repos
            except Exception as e:
                logger.error(f"An unexpected error occurred while fetching repositories (page {page}): {e}", exc_info=True)
                return None

        logger.info(f"Successfully fetched {len(all_repos)} repositories.")
        return all_repos

    # Placeholder for other GitHub methods like:
    # def create_issue(self, repo_full_name: str, title: str, body: Optional[str] = None, ...) -> Optional[IssueDict]:
    #     pass
    # def get_pull_requests(self, repo_full_name: str, state: str = "open", ...) -> Optional[List[PullRequestDict]]:
    #     pass
