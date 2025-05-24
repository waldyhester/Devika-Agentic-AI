"""
Provides functionalities for interacting with the Netlify API, primarily for deploying projects.

This module defines the `NetlifyService` class, which uses the `netlify_py` library
to create new sites and deploy project directories to Netlify.
"""

import os  # Added import
from typing import Any, Dict, Optional, TypedDict

from netlify_py import NetlifyPy

from src.config import Config
from src.logger import Logger
from src.project import ProjectManager

# Assuming netlify_py might raise specific exceptions, though its documentation is sparse.
# from netlify_py.exceptions import NetlifyError # Example, if such an exception exists.


logger = Logger()


class NetlifySiteDict(TypedDict, total=False):
    """
    Represents a subset of data for a Netlify site object.

    Attributes:
        id (str): The unique identifier of the site.
        name (str): The name of the site.
        url (str): The primary URL of the site.
        admin_url (str): The URL to the site's admin dashboard on Netlify.
        ssl_url (str): The SSL-enabled URL of the site.
        state (str): The current state of the site (e.g., "current", "building").
    """

    id: str
    name: str
    url: str
    admin_url: str
    ssl_url: str
    state: str


class NetlifyDeployDict(TypedDict, total=False):
    """
    Represents a subset of data for a Netlify deploy object.

    Attributes:
        id (str): The unique identifier of the deploy.
        site_id (str): The ID of the site this deploy belongs to.
        deploy_url (str): The unique URL for this specific deploy (often a permalink).
        ssl_deploy_url (str): The SSL-enabled unique URL for this specific deploy.
        state (str): The current state of the deploy (e.g., "uploading", "uploaded", "processing", "ready").
        logs_url (Optional[str]): A URL to access the deploy logs (might not always be present).
        # The raw netlify_py response has more fields like 'required', 'required_functions', etc.
    """

    id: str
    site_id: str
    deploy_url: str  # This is often the 'permalink' or 'url' in deploy object
    ssl_deploy_url: str  # Often the 'ssl_url' or 'ssl_permalink'
    state: str
    logs_url: Optional[str]


class NetlifyDeployResult(TypedDict):
    """
    Structured result for a deployment operation.

    Attributes:
        success (bool): True if the deployment process was initiated successfully, False otherwise.
        message (str): A human-readable message about the outcome.
        site_id (Optional[str]): The ID of the created or used Netlify site.
        deploy_id (Optional[str]): The ID of the deployment.
        deploy_url (Optional[str]): The primary URL of the deployed site.
        logs_url (Optional[str]): URL to the deployment logs.
        details (Optional[Dict[str, Any]]): Raw response or additional details from Netlify.
    """

    success: bool
    message: str
    site_id: Optional[str]
    deploy_id: Optional[str]
    deploy_url: Optional[
        str
    ]  # This would typically be site.ssl_url or site.url after deploy
    logs_url: Optional[str]  # Often part of the deploy object itself
    details: Optional[Dict[str, Any]]


class NetlifyService:
    """
    Service class for interacting with the Netlify API using the `netlify_py` library.

    Handles authentication and provides methods for deploying projects.

    Attributes:
        netlify_client (Optional[NetlifyPy]): An instance of the NetlifyPy client.
                                              None if API key is missing.
    """

    def __init__(self, api_key: Optional[str] = None) -> None:
        """
        Initialize the NetlifyService.

        Args:
            api_key (Optional[str]): A Netlify Personal Access Token. If not provided,
                                     it attempts to fetch one from the application config.
        """
        config = Config()
        token: Optional[str] = api_key or config.get_netlify_api_key()

        if not token:
            logger.warning(
                "Netlify API key not found in configuration. "
                "NetlifyService will not be able to deploy projects."
            )
            self.netlify_client: Optional[NetlifyPy] = None
        else:
            try:
                self.netlify_client = NetlifyPy(access_token=token)
                logger.info("NetlifyService initialized with API token.")
            except Exception as e:  # netlify_py might raise various errors on init
                logger.error(f"Failed to initialize NetlifyPy client: {e}")
                self.netlify_client = None

    def deploy_project(self, project_name: str) -> NetlifyDeployResult:
        """
        Deploy a project to Netlify.

        This method currently creates a new site on Netlify for each deployment
        and then deploys the project files to this new site. The `netlify_py`
        library handles the zipping and uploading of the project directory.

        Args:
            project_name (str): The name of the project to deploy. The project's
                                directory path is obtained via `ProjectManager`.

        Returns:
            NetlifyDeployResult: A dictionary containing the outcome of the deployment.
                                 Includes success status, messages, site ID, deploy ID,
                                 and relevant URLs.

        Note on Site Creation:
            This implementation creates a new Netlify site for each call to `deploy_project`.
            For deploying updates to an *existing* site, the `site_id` would need to be
            retrieved (e.g., from config or a previous deployment) and used directly
            with `self.netlify_client.deploys.deploy_site(existing_site_id, project_path)`.
            This behavior could be made configurable if needed.
        """
        if not self.netlify_client:
            return NetlifyDeployResult(
                success=False,
                message="NetlifyService not initialized or API token missing.",
                site_id=None,
                deploy_id=None,
                deploy_url=None,
                logs_url=None,
                details=None,
            )

        project_manager = ProjectManager()
        project_path: str = project_manager.get_project_path(project_name)

        if not os.path.isdir(project_path):
            return NetlifyDeployResult(
                success=False,
                message=f"Project directory not found at: {project_path}",
                site_id=None,
                deploy_id=None,
                deploy_url=None,
                logs_url=None,
                details=None,
            )

        site_id: Optional[str] = None
        deploy_id: Optional[str] = None
        final_site_url: Optional[str] = None
        deploy_logs_url: Optional[str] = (
            None  # Usually part of deploy object or constructed
        )

        try:
            # Step 1: Create a new site on Netlify
            # netlify_py's create_site doesn't take a name, one is auto-generated.
            # For custom domain or specific site name, further API calls or site ID reuse is needed.
            logger.info(f"Creating a new Netlify site for project: {project_name}...")
            site_info_raw: Dict[str, Any] = self.netlify_client.sites.create_site()  # type: ignore

            # Cast to TypedDict for better type checking, though netlify_py returns raw dict
            site_info = NetlifySiteDict(**site_info_raw)

            site_id = site_info.get("id")
            final_site_url = site_info.get("ssl_url") or site_info.get("url")
            logger.info(
                f"Netlify site created. ID: {site_id}, Name: {site_info.get('name')}, URL: {final_site_url}"
            )
            logger.debug(f"Full site creation response: {site_info}")

            if not site_id:
                return NetlifyDeployResult(
                    success=False,
                    message="Failed to create Netlify site: No site ID returned.",
                    site_id=None,
                    deploy_id=None,
                    deploy_url=None,
                    logs_url=None,
                    details=site_info,
                )

            # Step 2: Deploy the project directory to the newly created site
            logger.info(
                f"Deploying project from path: {project_path} to site ID: {site_id}..."
            )
            deploy_info_raw: Dict[str, Any] = self.netlify_client.deploys.deploy_site(site_id, project_path)  # type: ignore

            # Cast to TypedDict
            deploy_info = NetlifyDeployDict(**deploy_info_raw)

            deploy_id = deploy_info.get("id")
            # The main site URL is from `site_info`. The `deploy_info` often contains a permalink.
            # `deploy_url` in NetlifyDeployResult will refer to the main site URL.
            # `logs_url` might be part of deploy_info or constructed.
            # Example: deploy_info.get("links", {}).get("log") or deploy_info.get("log_access_attributes", {}).get("url")
            # For simplicity, if not directly available, we'll use the admin URL.
            deploy_logs_url = deploy_info.get(
                "admin_url", site_info.get("admin_url")
            )  # Fallback to site admin

            logger.info(
                f"Deployment initiated. Deploy ID: {deploy_id}, State: {deploy_info.get('state')}"
            )
            logger.debug(f"Full deployment response: {deploy_info}")

            return NetlifyDeployResult(
                success=True,
                message=f"Project '{project_name}' deployment initiated successfully to site '{site_info.get('name')}'.",
                site_id=site_id,
                deploy_id=deploy_id,
                deploy_url=final_site_url,
                logs_url=deploy_logs_url,
                details={"site_info": site_info, "deploy_info": deploy_info},
            )

        except (
            Exception
        ) as e:  # Catching general exception as netlify_py specific errors are not well documented
            logger.error(
                f"Netlify deployment failed for project '{project_name}': {e}",
                exc_info=True,
            )
            return NetlifyDeployResult(
                success=False,
                message=f"Netlify deployment failed: {e}",
                site_id=site_id,  # Include site_id if site creation succeeded but deploy failed
                deploy_id=None,
                deploy_url=final_site_url if site_id else None,
                logs_url=None,
                details={"error": str(e)},
            )
