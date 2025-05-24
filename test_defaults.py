from src.config import Config
from src.services.github import GitHubService
from src.services.git import GitService
from src.sandbox.code_runner import CodeRunner
# from src.browser.search import Browser # Browser class exists, but for simplicity, let's focus on the others.

config = Config()

print(f"GitHub API URL (default): {config.get_github_api_base_url()}")
assert config.get_github_api_base_url() == "https://api.github.com"

print(f"GitHub Timeout (default): {config.get_github_default_timeout()}")
assert config.get_github_default_timeout() == 10

print(f"Git Executable (default): {config.get_git_executable_path()}")
assert config.get_git_executable_path() == "git"

# Attempt to instantiate classes
try:
    GitHubService() 
    GitService()    
    CodeRunner()    
    # Browser() # Skipping Browser for now to avoid dealing with its specific dependencies if any for simple instantiation
    print("Classes instantiated successfully with default configs.")
except Exception as e:
    print(f"Error instantiating classes with default configs: {e}")
    raise
