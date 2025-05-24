"""
Provides functionalities for interacting with Git repositories using subprocess calls.

This module defines the `GitService` class (renamed from `Git` for clarity)
which allows cloning repositories, pulling changes, listing branches, getting commit
history, and viewing file content at specific commits, all by executing `git`
commands as subprocesses. This approach avoids direct dependency on GitPython library
as per refactoring objective.
"""
import os
import subprocess
import shutil
from typing import List, Tuple, Optional, TypedDict, Dict, Any

from src.config import Config
from src.logger import Logger

logger = Logger()
GIT_EXECUTABLE = "git" # Assumes 'git' is in PATH
DEFAULT_TIMEOUT = 300  # 5 minutes for git operations


class GitOperationResult(TypedDict):
    """
    Represents the result of a Git operation.

    Attributes:
        success (bool): True if the operation was successful, False otherwise.
        message (str): A human-readable message describing the outcome.
        details (Optional[str]): Additional details, often stdout or stderr from git.
    """
    success: bool
    message: str
    details: Optional[str]


class CommitDetails(TypedDict):
    """
    Represents details of a Git commit.

    Attributes:
        hash (str): The full commit hash.
        author_name (str): Author's name.
        author_email (str): Author's email.
        author_date (str): Author date in ISO format.
        committer_name (str): Committer's name.
        committer_email (str): Committer's email.
        committer_date (str): Committer date in ISO format.
        subject (str): The commit subject (first line of message).
        body (str): The full commit message body.
    """
    hash: str
    author_name: str
    author_email: str
    author_date: str
    committer_name: str
    committer_email: str
    committer_date: str
    subject: str
    body: str


class GitService:
    """
    Service class for interacting with Git repositories via subprocess calls.

    This class provides methods for common Git operations. It requires Git
    to be installed and accessible in the system's PATH.

    Attributes:
        config (Config): Application configuration instance.
        repos_base_dir (str): The base directory where repositories are stored.
    """

    def __init__(self) -> None:
        """
        Initialize the GitService.

        Loads configuration to determine the base directory for repositories.
        """
        self.config: Config = Config()
        self.repos_base_dir: str = self.config.get_repos_dir()
        os.makedirs(self.repos_base_dir, exist_ok=True)
        logger.info(f"GitService initialized. Repositories base directory: {self.repos_base_dir}")

    def _get_repo_path(self, repo_name: str) -> str:
        """
        Construct the absolute path for a given repository name.

        Args:
            repo_name (str): The name of the repository (will be used as directory name).

        Returns:
            str: The absolute path to the repository directory.
        """
        # Sanitize repo_name to prevent path traversal issues, though less critical if repo_name is internal
        safe_repo_name = os.path.basename(repo_name)
        return os.path.join(self.repos_base_dir, safe_repo_name)

    def _run_git_command(
        self, command: List[str], repo_path: Optional[str] = None
    ) -> Tuple[int, str, str]:
        """
        Execute a Git command as a subprocess.

        Args:
            command (List[str]): The Git command and its arguments as a list.
            repo_path (Optional[str]): The path to the repository directory. If None,
                                       the command is run in the current working directory
                                       (or as specified by other means if `git clone`).

        Returns:
            Tuple[int, str, str]: A tuple containing the return code, stdout, and stderr.
        """
        full_command = [GIT_EXECUTABLE] + command
        logger.debug(f"Executing Git command: {' '.join(full_command)} in {repo_path or 'current dir'}")
        try:
            process = subprocess.run(
                full_command,
                capture_output=True,
                text=True, # Decodes output as system default (usually UTF-8)
                cwd=repo_path,
                timeout=DEFAULT_TIMEOUT,
                check=False, # Do not raise CalledProcessError for non-zero exit codes
                encoding="utf-8", errors="replace" # Specify encoding
            )
            logger.debug(f"Git command finished. RC: {process.returncode}")
            if process.stdout: logger.debug(f"Stdout: {process.stdout[:200]}...")
            if process.stderr: logger.debug(f"Stderr: {process.stderr[:200]}...")
            return process.returncode, process.stdout, process.stderr
        except subprocess.TimeoutExpired:
            err_msg = f"Git command timed out after {DEFAULT_TIMEOUT}s: {' '.join(full_command)}"
            logger.error(err_msg)
            return -1, "", err_msg # Using -1 for timeout
        except FileNotFoundError:
            err_msg = f"Git executable '{GIT_EXECUTABLE}' not found. Please ensure Git is installed and in PATH."
            logger.error(err_msg)
            return -1, "", err_msg
        except Exception as e:
            err_msg = f"Unexpected error running Git command {' '.join(full_command)}: {e}"
            logger.error(err_msg, exc_info=True)
            return -1, "", err_msg

    def clone_repo(self, url: str, repo_name: str, overwrite: bool = False) -> GitOperationResult:
        """
        Clone a Git repository into the managed repositories directory.

        Args:
            url (str): The URL of the Git repository to clone.
            repo_name (str): The desired name for the repository directory.
            overwrite (bool): If True and the repository directory already exists,
                              it will be deleted and re-cloned. Defaults to False.

        Returns:
            GitOperationResult: Dictionary indicating success or failure.
        """
        repo_path = self._get_repo_path(repo_name)

        if os.path.exists(repo_path):
            if overwrite:
                logger.info(f"Repository '{repo_name}' already exists at {repo_path}. Overwriting.")
                try:
                    shutil.rmtree(repo_path)
                except OSError as e:
                    msg = f"Error deleting existing repository '{repo_name}': {e}"
                    logger.error(msg)
                    return {"success": False, "message": msg, "details": str(e)}
            else:
                msg = f"Repository '{repo_name}' already exists at {repo_path}. Clone aborted."
                logger.warning(msg)
                return {"success": False, "message": msg, "details": "Directory exists."}
        
        # Note: For `git clone`, the `repo_path` is the target directory, so `cwd` for subprocess
        # should be the parent of `repo_path` (i.e., `self.repos_base_dir`) or None.
        # The command itself specifies the target directory.
        command = ["clone", url, repo_path]
        return_code, stdout, stderr = self._run_git_command(command) # cwd defaults to None

        if return_code == 0:
            msg = f"Repository '{url}' cloned successfully as '{repo_name}' at {repo_path}."
            logger.info(msg)
            return {"success": True, "message": msg, "details": stdout}
        else:
            msg = f"Failed to clone repository '{url}' as '{repo_name}'."
            logger.error(f"{msg} Error: {stderr}")
            return {"success": False, "message": msg, "details": stderr}

    def pull_repo(self, repo_name: str, branch: Optional[str] = None) -> GitOperationResult:
        """
        Pull changes from the remote repository for the specified branch.

        Args:
            repo_name (str): The name of the repository directory.
            branch (Optional[str]): The branch to pull. If None, pulls the current branch.

        Returns:
            GitOperationResult: Dictionary indicating success or failure.
        """
        repo_path = self._get_repo_path(repo_name)
        if not os.path.isdir(repo_path):
            msg = f"Repository '{repo_name}' not found at {repo_path}."
            logger.error(msg)
            return {"success": False, "message": msg, "details": "Directory not found."}

        command = ["pull"]
        if branch:
            command.extend(["origin", branch])
        
        return_code, stdout, stderr = self._run_git_command(command, repo_path=repo_path)

        if return_code == 0:
            msg = f"Pulled changes for repository '{repo_name}' (branch: {branch or 'current'})."
            logger.info(msg)
            return {"success": True, "message": msg, "details": stdout}
        else:
            msg = f"Failed to pull changes for repository '{repo_name}' (branch: {branch or 'current'})."
            logger.error(f"{msg} Error: {stderr}")
            return {"success": False, "message": msg, "details": stderr}

    def get_branches(self, repo_name: str) -> Optional[List[str]]:
        """
        Get a list of all local and remote branches for a repository.

        Args:
            repo_name (str): The name of the repository directory.

        Returns:
            Optional[List[str]]: A list of branch names, or None if an error occurs.
        """
        repo_path = self._get_repo_path(repo_name)
        if not os.path.isdir(repo_path):
            logger.error(f"Repository '{repo_name}' not found at {repo_path}.")
            return None

        command = ["branch", "-a"]
        return_code, stdout, stderr = self._run_git_command(command, repo_path=repo_path)

        if return_code == 0:
            branches = [line.strip().lstrip("* ") for line in stdout.splitlines() if line.strip()]
            # Filter out remote HEAD pointers like 'remotes/origin/HEAD -> origin/main'
            branches = [b for b in branches if not "->" in b]
            logger.info(f"Found branches for '{repo_name}': {branches}")
            return branches
        else:
            logger.error(f"Failed to get branches for '{repo_name}'. Error: {stderr}")
            return None

    def get_current_branch(self, repo_name: str) -> Optional[str]:
        """
        Get the current active branch of a repository.

        Args:
            repo_name (str): The name of the repository directory.

        Returns:
            Optional[str]: The name of the current branch, or None if an error occurs
                           or not on a branch (e.g., detached HEAD).
        """
        repo_path = self._get_repo_path(repo_name)
        if not os.path.isdir(repo_path):
            logger.error(f"Repository '{repo_name}' not found at {repo_path}.")
            return None

        command = ["rev-parse", "--abbrev-ref", "HEAD"] # More reliable than 'git branch --show-current'
        return_code, stdout, stderr = self._run_git_command(command, repo_path=repo_path)

        if return_code == 0:
            current_branch = stdout.strip()
            if current_branch == "HEAD": # Detached HEAD state
                logger.info(f"Repository '{repo_name}' is in a detached HEAD state.")
                return None 
            logger.info(f"Current branch for '{repo_name}': {current_branch}")
            return current_branch
        else:
            logger.error(f"Failed to get current branch for '{repo_name}'. Error: {stderr}")
            return None

    def get_commits(
        self, repo_name: str, branch: Optional[str] = None, count: int = 10
    ) -> Optional[List[CommitDetails]]:
        """
        Get a list of commit details for a repository and branch.

        Args:
            repo_name (str): The name of the repository directory.
            branch (Optional[str]): The branch to get commits from. Defaults to current branch.
            count (int): The maximum number of commits to retrieve. Defaults to 10.

        Returns:
            Optional[List[CommitDetails]]: A list of commit details, or None if an error occurs.
        """
        repo_path = self._get_repo_path(repo_name)
        if not os.path.isdir(repo_path):
            logger.error(f"Repository '{repo_name}' not found at {repo_path}.")
            return None

        # Format: HASH<tab>AUTHOR_NAME<tab>AUTHOR_EMAIL<tab>AUTHOR_DATE_ISO<tab>SUBJECT
        log_format = "%H%x09%an%x09%ae%x09%ad%x09%s"
        command = ["log", f"--pretty=format:{log_format}", f"-n{count}", "--date=iso"]
        if branch:
            command.append(branch)

        return_code, stdout, stderr = self._run_git_command(command, repo_path=repo_path)

        if return_code == 0:
            commits: List[CommitDetails] = []
            for line in stdout.splitlines():
                if not line.strip(): continue
                parts = line.split("\t", 4)
                if len(parts) == 5:
                    # For simplicity, we're not fetching committer details or full body here.
                    # A full `git show` would be needed per commit for that.
                    commits.append(CommitDetails(
                        hash=parts[0], author_name=parts[1], author_email=parts[2],
                        author_date=parts[3], subject=parts[4],
                        committer_name="", committer_email="", committer_date="", body="" # Placeholder
                    ))
            logger.info(f"Retrieved {len(commits)} commits for '{repo_name}' (branch: {branch or 'current'}).")
            return commits
        else:
            logger.error(f"Failed to get commits for '{repo_name}'. Error: {stderr}")
            return None

    def get_commit_details(self, repo_name: str, commit_hash: str) -> Optional[CommitDetails]:
        """
        Get detailed information for a specific commit.

        Args:
            repo_name (str): The name of the repository.
            commit_hash (str): The hash of the commit.

        Returns:
            Optional[CommitDetails]: A dictionary with commit details, or None if error.
        """
        repo_path = self._get_repo_path(repo_name)
        if not os.path.isdir(repo_path):
            logger.error(f"Repository '{repo_name}' not found at {repo_path}.")
            return None

        # Format: FullHash<NL>AuthorName<NL>AuthorEmail<NL>AuthorDateISO<NL>CommitterName<NL>CommitterEmail<NL>CommitterDateISO<NL>Subject<NL>Body
        # Using %x00 (null byte) as a separator for robustness, then splitting.
        log_format = "%H%x00%an%x00%ae%x00%ad%x00%cn%x00%ce%x00cd%x00%s%x00%b"
        command = ["show", "--quiet", f"--pretty=format:{log_format}", "--date=iso", commit_hash]
        
        return_code, stdout, stderr = self._run_git_command(command, repo_path=repo_path)

        if return_code == 0 and stdout.strip():
            parts = stdout.strip().split("\x00", 8)
            if len(parts) == 9:
                return CommitDetails(
                    hash=parts[0], author_name=parts[1], author_email=parts[2],
                    author_date=parts[3], committer_name=parts[4], committer_email=parts[5],
                    committer_date=parts[6], subject=parts[7], body=parts[8]
                )
            else:
                logger.error(f"Could not parse commit details for {commit_hash} in {repo_name}. Output parts: {len(parts)}")
                return None
        else:
            logger.error(f"Failed to get details for commit {commit_hash} in '{repo_name}'. Error: {stderr}")
            return None


    def get_file_content_at_commit(
        self, repo_name: str, file_path: str, commit_hash: Optional[str] = None
    ) -> Optional[str]:
        """
        Get the content of a file at a specific commit, or the latest if no commit_hash.

        Args:
            repo_name (str): The name of the repository.
            file_path (str): The relative path to the file within the repository.
            commit_hash (Optional[str]): The commit hash. If None, gets the latest version.

        Returns:
            Optional[str]: The content of the file, or None if an error occurs.
        """
        repo_path = self._get_repo_path(repo_name)
        if not os.path.isdir(repo_path):
            logger.error(f"Repository '{repo_name}' not found at {repo_path}.")
            return None

        target = f"{commit_hash}:{file_path}" if commit_hash else file_path
        command = ["show", target]
        
        return_code, stdout, stderr = self._run_git_command(command, repo_path=repo_path)

        if return_code == 0:
            return stdout
        else:
            logger.error(f"Failed to get file content for '{target}' in '{repo_name}'. Error: {stderr}")
            return None
