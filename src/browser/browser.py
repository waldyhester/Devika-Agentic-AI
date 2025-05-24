"""
Provides core browser automation functionalities using Playwright.

This module defines the `Browser` class, which encapsulates common browser
operations such as launching a browser instance, navigating to URLs,
taking screenshots, extracting page content (HTML, Markdown, PDF, text),
and managing the Playwright lifecycle. It is designed to be used asynchronously.
"""
import base64
import os
from typing import Optional, Tuple, Dict, Any

from playwright.async_api import (
    async_playwright,
    TimeoutError as PlaywrightTimeoutError,
    Error as PlaywrightError,
    Playwright,
    Browser as PlaywrightBrowser,
    Page,
)
from markdownify import markdownify as md
from pdfminer.high_level import extract_text

from src.config import Config
from src.state import AgentState, StateType
from src.logger import Logger
from src.socket_instance import emit_agent # Assuming this is used for agent communication

logger = Logger()


class Browser:
    """
    Asynchronous browser automation class using Playwright.

    Manages a Playwright instance, a browser instance, and a page object
    to perform web browsing tasks.

    Attributes:
        playwright (Optional[Playwright]): The Playwright context manager instance.
        browser (Optional[PlaywrightBrowser]): The Playwright browser instance.
        page (Optional[Page]): The Playwright page instance.
        agent_state_manager (AgentState): Manages the agent's state.
        config (Config): Application configuration instance.
    """

    def __init__(self) -> None:
        """Initialize the Browser class with default None values for Playwright objects."""
        self.playwright: Optional[Playwright] = None
        self.browser: Optional[PlaywrightBrowser] = None
        self.page: Optional[Page] = None
        self.agent_state_manager: AgentState = AgentState()
        self.config: Config = Config()

    async def start(self) -> "Browser":
        """
        Start the Playwright instance and launch a new browser page.

        Initializes `self.playwright`, `self.browser`, and `self.page`.
        The browser is launched in headless mode.

        Returns:
            Browser: The current instance of the Browser class.

        Raises:
            PlaywrightError: If Playwright fails to start or launch the browser.
        """
        try:
            self.playwright = await async_playwright().start()
            self.browser = await self.playwright.chromium.launch(headless=True)
            self.page = await self.browser.new_page()
            logger.info("Browser started successfully.")
        except PlaywrightError as e:
            logger.error(f"Error starting browser: {e}")
            raise
        return self

    async def go_to(self, url: str, timeout: int = 30000) -> bool:
        """
        Navigate the current page to the specified URL.

        Args:
            url (str): The URL to navigate to.
            timeout (int): Maximum navigation time in milliseconds. Defaults to 30000.

        Returns:
            bool: True if navigation was successful, False otherwise.
        """
        if not self.page:
            logger.error("Page not initialized. Call start() first.")
            return False
        try:
            await self.page.goto(url, timeout=timeout)
            logger.info(f"Successfully navigated to: {url}")
            return True
        except PlaywrightTimeoutError:
            logger.warning(f"TimeoutError: Navigating to {url} timed out after {timeout}ms.")
            return False
        except PlaywrightError as e:
            logger.error(f"PlaywrightError navigating to {url}: {e}")
            return False
        except Exception as e: # Catch any other unexpected errors
            logger.error(f"Unexpected error navigating to {url}: {e}")
            return False


    async def screenshot(
        self, project_name: str
    ) -> Optional[Tuple[str, str]]:
        """
        Take a screenshot of the current page and save it.

        The screenshot is saved to the directory specified in the application config.
        Agent state is updated with the screenshot URL and path.

        Args:
            project_name (str): The name of the project, used for context and state updates.

        Returns:
            Optional[Tuple[str, str]]: A tuple containing the path to the saved screenshot
                                      and the base64 encoded screenshot string, or None if an error occurs.
        """
        if not self.page:
            logger.error("Page not initialized for screenshot. Call start() first.")
            return None

        screenshots_save_path: str = self.config.get_screenshots_dir()
        os.makedirs(screenshots_save_path, exist_ok=True)

        try:
            page_metadata: Dict[str, str] = await self.page.evaluate(
                "() => { return { url: document.location.href, title: document.title } }"
            )
            page_url: str = page_metadata.get("url", "unknown_url")
            page_title: str = page_metadata.get("title", "unknown_title").replace(" ", "_")[:50] # Sanitize title

            random_filename_part: str = os.urandom(8).hex() # Shorter random part
            filename_to_save = f"{page_title}_{random_filename_part}.png"
            path_to_save: str = os.path.join(screenshots_save_path, filename_to_save)

            await self.page.emulate_media(media="screen")
            screenshot_bytes: bytes = await self.page.screenshot(path=path_to_save)
            screenshot_base64: str = base64.b64encode(screenshot_bytes).decode()

            # Update agent state
            new_state: StateType = self.agent_state_manager.new_state()
            new_state["internal_monologue"] = f"Took a screenshot of {page_url}"
            # Ensure browser_session structure matches StateType definition
            new_state["browser_session"] = {"url": page_url, "screenshot": path_to_save} # type: ignore
            self.agent_state_manager.add_to_current_state(project_name, new_state)
            
            # Emit screenshot data (optional, based on original logic)
            emit_agent("screenshot", {"data": screenshot_base64, "project_name": project_name}, broadcast=False)
            logger.info(f"Screenshot saved to: {path_to_save}")
            return path_to_save, screenshot_base64
        except PlaywrightError as e:
            logger.error(f"PlaywrightError taking screenshot: {e}")
            return None
        except Exception as e:
            logger.error(f"Unexpected error taking screenshot: {e}")
            return None

    async def get_html(self) -> Optional[str]:
        """
        Get the HTML content of the current page.

        Returns:
            Optional[str]: The HTML content as a string, or None if an error occurs.
        """
        if not self.page:
            logger.error("Page not initialized to get HTML. Call start() first.")
            return None
        try:
            return await self.page.content()
        except PlaywrightError as e:
            logger.error(f"PlaywrightError getting HTML content: {e}")
            return None

    async def get_markdown(self) -> Optional[str]:
        """
        Convert the HTML content of the current page to Markdown.

        Returns:
            Optional[str]: The Markdown content as a string, or None if an error occurs.
        """
        html_content = await self.get_html()
        if html_content:
            return md(html_content)
        return None

    async def get_pdf_content_and_save(self, project_name: str) -> Optional[str]:
        """
        Generate a PDF of the current page, save it, and extract text from it.

        Note: `pdfminer.high_level.extract_text` is a synchronous (blocking) operation.
        In a highly asynchronous application, consider running it in a separate thread
        or process using `asyncio.to_thread` (Python 3.9+) or `loop.run_in_executor`.

        Args:
            project_name (str): The name of the project, used for naming the PDF.

        Returns:
            Optional[str]: Extracted text from the PDF, or None if an error occurs.
        """
        if not self.page:
            logger.error("Page not initialized to get PDF. Call start() first.")
            return None

        pdfs_save_path: str = self.config.get_pdfs_dir()
        os.makedirs(pdfs_save_path, exist_ok=True)

        try:
            page_metadata: Dict[str, str] = await self.page.evaluate(
                "() => { return { url: document.location.href, title: document.title } }"
            )
            # Sanitize title for filename
            safe_title = "".join(c if c.isalnum() else "_" for c in page_metadata.get("title", "untitled"))[:50]
            filename_to_save = f"{safe_title}_{os.urandom(4).hex()}.pdf"
            path_to_save: str = os.path.join(pdfs_save_path, filename_to_save)

            await self.page.pdf(path=path_to_save)
            logger.info(f"PDF saved to: {path_to_save}")

            # pdf_to_text is synchronous, consider executor for async environments
            extracted_text = self.pdf_to_text(path_to_save)
            return extracted_text
        except PlaywrightError as e:
            logger.error(f"PlaywrightError generating or saving PDF: {e}")
            return None
        except Exception as e: # Catch errors from pdf_to_text as well
            logger.error(f"Error extracting text from PDF {path_to_save}: {e}")
            return None


    def pdf_to_text(self, pdf_path: str) -> Optional[str]:
        """
        Extract text content from a local PDF file.

        Args:
            pdf_path (str): The file path to the PDF.

        Returns:
            Optional[str]: The extracted text, or None if an error occurs.
        """
        try:
            return extract_text(pdf_path).strip()
        except Exception as e: # Catch specific pdfminer exceptions if known
            logger.error(f"Error extracting text from PDF {pdf_path}: {e}")
            return None

    async def extract_text(self) -> Optional[str]:
        """
        Extract the visible text (innerText) from the body of the current page.

        Returns:
            Optional[str]: The extracted text, or None if an error occurs.
        """
        if not self.page:
            logger.error("Page not initialized to extract text. Call start() first.")
            return None
        try:
            return await self.page.evaluate("() => document.body.innerText")
        except PlaywrightError as e:
            logger.error(f"PlaywrightError extracting text: {e}")
            return None

    async def close(self) -> None:
        """
        Close the page, browser, and stop the Playwright instance.
        Handles cases where components might not have been initialized.
        """
        if self.page:
            try:
                await self.page.close()
                logger.info("Playwright page closed.")
            except PlaywrightError as e:
                logger.error(f"Error closing Playwright page: {e}")
            self.page = None
        if self.browser:
            try:
                await self.browser.close()
                logger.info("Playwright browser closed.")
            except PlaywrightError as e:
                logger.error(f"Error closing Playwright browser: {e}")
            self.browser = None
        if self.playwright:
            try:
                await self.playwright.stop()
                logger.info("Playwright context stopped.")
            except PlaywrightError as e: # Though stop() itself doesn't usually throw
                logger.error(f"Error stopping Playwright context: {e}")
            self.playwright = None
