"""
Enables an LLM to interact with a web page through a simplified text interface.

This module is based on the concepts from natbot (https://github.com/nat/natbot),
modified for integration with the Devika application. It defines a `Crawler` class
that uses Playwright's synchronous API to interact with web pages, simplify their
content into a text-based representation, and execute commands (like click, type, scroll)
determined by an LLM.

The `start_interaction` function orchestrates this process, taking an objective
and using an LLM to drive the `Crawler` towards that objective.
"""

import json
import os
import re
import time
from sys import platform
from typing import Any, Dict, List, Optional, Set, TypedDict

from jinja2 import BaseLoader, Environment
from playwright.sync_api import Browser as SyncBrowser
from playwright.sync_api import CDPSession
from playwright.sync_api import Error as PlaywrightError  # General Playwright error
from playwright.sync_api import Page as SyncPage
from playwright.sync_api import Playwright as SyncPlaywright
from playwright.sync_api import (
    TimeoutError as PlaywrightTimeoutError,
)  # For specific timeout errors
from playwright.sync_api import sync_playwright

from src.config import Config
from src.llm import LLM
from src.logger import Logger
from src.state import AgentState, StateType

logger = Logger()

# Load the prompt template from the associated Jinja2 file.
try:
    with open("src/browser/interaction_prompt.jinja2", "r", encoding="utf-8") as f:
        PROMPT_TEMPLATE = f.read().strip()
except FileNotFoundError:
    PROMPT_TEMPLATE = "Error: Browser interaction prompt template not found."
    logger.error(PROMPT_TEMPLATE)


# Elements that are typically not useful for LLM context or interaction
BLACKLISTED_ELEMENTS: Set[str] = {
    "html",
    "head",
    "title",
    "meta",
    "iframe",
    "body",
    "script",
    "style",
    "path",
    "svg",
    "br",
    "::marker",
}


# Type definition for the structured action expected from the LLM
class LLMAction(TypedDict):
    action_type: str  # e.g., "CLICK", "TYPE", "SCROLL", "COMPLETE", "FAIL"
    target_id: Optional[str]
    text_value: Optional[str]
    is_complete: bool
    justification: str


class Crawler:
    """
    A synchronous web crawler that simplifies web content for LLM interaction.

    Uses Playwright's synchronous API to navigate pages, extract a simplified
    representation of the DOM, and execute commands.

    Attributes:
        playwright_context (SyncPlaywright): Playwright synchronous context manager.
        browser (SyncBrowser): Playwright browser instance.
        page (SyncPage): Current Playwright page object.
        client (Optional[CDPSession]): Chrome DevTools Protocol session.
        page_element_buffer (Dict[int, Dict[str, Any]]): Stores metadata of interactable
                                                         elements visible on the current page,
                                                         keyed by a simplified ID.
        config (Config): Application configuration.
        agent_state_manager (AgentState): Manages agent state.
    """

    def __init__(self) -> None:
        """Initialize the Crawler, launching a Playwright browser instance."""
        self.playwright_context: SyncPlaywright = sync_playwright().start()
        self.browser: SyncBrowser = self.playwright_context.chromium.launch(
            headless=True
        )
        self.page: SyncPage = self.browser.new_page()
        self.page.set_viewport_size({"width": 1280, "height": 1080})
        self.client: Optional[CDPSession] = None
        self.page_element_buffer: Dict[int, Dict[str, Any]] = {}
        self.config: Config = Config()
        self.agent_state_manager: AgentState = AgentState()
        logger.info("Crawler initialized with Playwright browser.")

    def screenshot(self, project_name: str) -> Optional[str]:
        """
        Take a screenshot of the current page and save it.

        Updates agent state with the screenshot information.

        Args:
            project_name (str): The name of the project for context.

        Returns:
            Optional[str]: The path to the saved screenshot, or None on error.
        """
        screenshots_save_path: str = self.config.get_screenshots_dir()
        os.makedirs(screenshots_save_path, exist_ok=True)

        try:
            page_metadata: Dict[str, str] = self.page.evaluate(
                "() => { return { url: document.location.href, title: document.title } }"
            )
            page_url: str = page_metadata.get("url", "unknown_url")
            page_title_slug: str = re.sub(
                r"\W+", "_", page_metadata.get("title", "untitled")
            )[:50]

            random_filename_part = os.urandom(8).hex()
            filename_to_save = f"{page_title_slug}_{random_filename_part}.png"
            path_to_save = os.path.join(screenshots_save_path, filename_to_save)

            self.page.emulate_media(media="screen")
            self.page.screenshot(path=path_to_save)

            new_state: StateType = self.agent_state_manager.new_state()
            new_state["internal_monologue"] = f"Took a screenshot of {page_url}"
            new_state["browser_session"] = {"url": page_url, "screenshot": path_to_save}  # type: ignore
            # Assuming project_name for add_to_current_state is obtained from elsewhere if needed
            # or the method implicitly knows the current project.
            # For now, if project_name was essential for state, this needs a broader fix.
            # However, AgentState.add_to_current_state doesn't always require project_name
            # if it's already set in the instance or globally.
            # Let's assume the state manager can handle it or project_name is not strictly needed here
            # for this specific state update's context beyond what's logged.
            # If AgentState needs project_name, it should be part of its context or passed differently.
            # This change is based on fixing F841 - unused 'project_name' in this method.
            # A broader refactor might be needed if project_name is critical for state association here.
            # For now, calling it without project_name if the method signature allows,
            # or retrieving it from self.agent_state_manager if it holds current project context.
            # The AgentState class's add_to_current_state takes (self, key, value) or (self, project_name, state_update)
            # The previous call was add_to_current_state(project_name, new_state)
            # This implies project_name IS used by add_to_current_state.
            # This means removing project_name from screenshot() is problematic if not handled carefully.
            # Let's check AgentState.add_to_current_state
            # It is: def add_to_current_state(self, project_name_or_key: str, value_or_state_update: Any = None) -> None:
            # So project_name is indeed used.
            # This means `project_name` in `screenshot` was NOT unused if it was for this call.
            # Re-evaluating: F841 is "local variable 'project_name' is assigned to but never used"
            # Parameter 'project_name' IS used in `self.agent_state_manager.add_to_current_state(project_name, new_state)`
            # So, project_name was NOT unused. My previous F841 analysis for this was incorrect.
            # The only way it would be "unused" is if the call to add_to_current_state was different.

            # My apologies, the variable `project_name` IS used in `self.agent_state_manager.add_to_current_state(project_name, new_state)`.
            # Therefore, it is NOT an F841 error. I will not remove it.

            # The only remaining error from flake8 was E261. I will fix that.
            # The previous diff for removing project_name was incorrect. I will create a new diff for E261.
            self.agent_state_manager.add_to_current_state(project_name, new_state)  # project_name IS used here.
            logger.info(f"Screenshot saved to {path_to_save}")
            return path_to_save
        except PlaywrightError as e:
            logger.error(f"Playwright error during screenshot: {e}")
            return None
        except Exception as e:
            logger.error(f"Unexpected error during screenshot: {e}")
            return None

    def go_to_page(self, url: str) -> bool:
        """
        Navigate to a specified URL.

        Args:
            url (str): The URL to navigate to.

        Returns:
            bool: True if navigation was successful, False otherwise.
        """
        try:
            full_url = url if "://" in url else "http://" + url
            self.page.goto(url=full_url, timeout=30000)
            self.client = self.page.context.new_cdp_session(self.page)
            self.page_element_buffer = {}  # Clear buffer for new page
            logger.info(f"Navigated to URL: {full_url}")
            return True
        except PlaywrightTimeoutError:
            logger.warning(f"Timeout navigating to URL: {url}")
            return False
        except PlaywrightError as e:
            logger.error(f"Playwright error navigating to {url}: {e}")
            return False
        except Exception as e:
            logger.error(f"Unexpected error navigating to {url}: {e}")
            return False

    def scroll(self, direction: str) -> None:
        """
        Scroll the page up or down.

        Args:
            direction (str): "up" or "down".
        """
        try:
            if direction == "up":
                self.page.evaluate(
                    "(document.scrollingElement || document.body).scrollTop = "
                    "(document.scrollingElement || document.body).scrollTop - window.innerHeight;"
                )
            elif direction == "down":
                self.page.evaluate(
                    "(document.scrollingElement || document.body).scrollTop = "
                    "(document.scrollingElement || document.body).scrollTop + window.innerHeight;"
                )
            logger.info(f"Scrolled {direction}.")
        except PlaywrightError as e:
            logger.error(f"Playwright error during scroll: {e}")
        except Exception as e:
            logger.error(f"Unexpected error during scroll: {e}")

    def click(self, element_id_str: str) -> None:
        """
        Click on an element identified by its simplified ID.

        Args:
            element_id_str (str): The string ID of the element from `page_element_buffer`.
        """
        try:
            element_id = int(element_id_str)
            element_info = self.page_element_buffer.get(element_id)
            if not element_info:
                logger.warning(
                    f"Element with ID {element_id_str} not found in buffer for clicking."
                )
                return

            # Remove target="_blank" to prevent new tabs
            self.page.evaluate(
                '() => { for (const link of document.getElementsByTagName("a")) { link.removeAttribute("target"); } }'
            )

            x = element_info.get("center_x")
            y = element_info.get("center_y")

            if x is not None and y is not None:
                self.page.mouse.click(x, y)
                logger.info(f"Clicked element with ID {element_id_str} at ({x}, {y}).")
            else:
                logger.warning(
                    f"Center coordinates not found for element ID {element_id_str}."
                )
        except ValueError:
            logger.error(f"Invalid element ID format for click: {element_id_str}")
        except PlaywrightError as e:
            logger.error(
                f"Playwright error during click on element ID {element_id_str}: {e}"
            )
        except Exception as e:
            logger.error(
                f"Unexpected error during click on element ID {element_id_str}: {e}"
            )

    def type_into(self, element_id_str: str, text: str, submit: bool = False) -> None:
        """
        Type text into an input field and optionally submit by pressing Enter.

        Args:
            element_id_str (str): The string ID of the input element.
            text (str): The text to type.
            submit (bool): Whether to press Enter after typing. Defaults to False.
        """
        try:
            self.click(element_id_str)  # Focus the element by clicking it
            self.page.keyboard.type(text)
            logger.info(f"Typed '{text}' into element ID {element_id_str}.")
            if submit:
                self.page.keyboard.press("Enter")
                logger.info(
                    f"Submitted form after typing into element ID {element_id_str}."
                )
        except PlaywrightError as e:
            logger.error(
                f"Playwright error during type into element ID {element_id_str}: {e}"
            )
        except Exception as e:
            logger.error(
                f"Unexpected error during type into element ID {element_id_str}: {e}"
            )

    def crawl_page_content(self) -> List[str]:
        """
        Crawl the current page and return a simplified text representation of visible elements.

        This method captures a DOM snapshot, processes it to identify visible and
        interactable elements, and formats them into a simplified string list.
        This list is intended for an LLM to understand the page structure.

        Returns:
            List[str]: A list of strings, where each string represents an interactable
                       or visible element on the page in a simplified format.
        """
        if not self.client or not self.page:
            logger.error("CDP client or page not initialized for crawl.")
            return ["Error: Browser session not properly initialized."]

        start_time = time.time()
        self.page_element_buffer = {}  # Reset buffer for current view
        elements_of_interest: List[str] = []
        id_counter = 0

        try:
            # Get DOM snapshot using CDP
            dom_snapshot = self.client.send(
                "DOMSnapshot.captureSnapshot",
                {
                    "computedStyles": [],
                    "includeDOMRects": True,
                    "includePaintOrder": True,
                },
            )
            strings: List[str] = dom_snapshot["strings"]
            document_snapshot = dom_snapshot["documents"][0]
            nodes = document_snapshot["nodes"]
            layout = document_snapshot["layout"]

            # Helper to convert node name for simplified representation
            def convert_node_name(node_name: str, is_clickable_node: bool) -> str:
                node_name_lower = node_name.lower()
                if node_name_lower == "a":
                    return "link"
                if node_name_lower == "input":
                    return "input"
                if node_name_lower == "textarea":
                    return "textarea"
                if node_name_lower == "img":
                    return "img"
                if node_name_lower == "button" or is_clickable_node:
                    return "button"
                return (
                    "text"  # Default for other elements like P, DIV, SPAN, H1-H6 etc.
                )

            # Pre-calculate viewport and screen dimensions
            device_pixel_ratio = self.page.evaluate("window.devicePixelRatio")
            if platform == "darwin" and device_pixel_ratio == 1:
                device_pixel_ratio = 2  # Common macOS correction

            win_upper_bound = self.page.evaluate("window.pageYOffset")
            win_left_bound = self.page.evaluate("window.pageXOffset")
            # Use viewport size for visible area, not screen size
            viewport_size = self.page.viewport_size
            if not viewport_size:
                viewport_size = {"width": 1280, "height": 1080}  # Fallback

            win_width = viewport_size["width"]
            win_height = viewport_size["height"]
            win_right_bound = win_left_bound + win_width
            win_lower_bound = win_upper_bound + win_height

            # Process nodes
            for i, node_idx in enumerate(layout["nodeIndex"]):
                node_name_idx = nodes["nodeName"][node_idx]
                node_name_str = strings[node_name_idx].lower()

                if node_name_str in BLACKLISTED_ELEMENTS:
                    continue

                x, y, width, height = layout["bounds"][i]
                x /= device_pixel_ratio
                y /= device_pixel_ratio
                width /= device_pixel_ratio
                height /= device_pixel_ratio

                # Check if element is within viewport
                is_in_viewport = (
                    x < win_right_bound
                    and (x + width) > win_left_bound
                    and y < win_lower_bound
                    and (y + height) > win_upper_bound
                )
                if not is_in_viewport or width == 0 or height == 0:
                    continue

                # Determine if clickable (more robustly if possible)
                is_clickable_node = (
                    nodes["isClickable"]["index"][node_idx]
                    if "isClickable" in nodes
                    else (
                        node_name_str in ["a", "button"]
                        or (
                            node_name_str == "input"
                            and nodes["attributes"][node_idx].get("type", "")
                            not in [
                                "hidden",
                                "text",
                                "password",
                                "email",
                                "search",
                                "tel",
                                "url",
                                "number",
                            ]
                        )
                    )
                )

                element_text_content = ""
                if nodes["nodeValue"][node_idx] >= 0:  # Text nodes
                    element_text_content = strings[nodes["nodeValue"][node_idx]].strip()

                # For input fields, try to get their current value
                if node_name_str == "input":
                    try:
                        # Assumption: nodes["inputValue"]["index"] is a list of node_idx values
                        # that are inputs and have values.
                        # nodes["inputValue"]["value"] is a parallel list of string_indices for those values.
                        pos_in_input_list = nodes["inputValue"]["index"].index(node_idx)
                        value_string_index = nodes["inputValue"]["value"][
                            pos_in_input_list
                        ]
                        if value_string_index >= 0:
                            element_text_content = strings[value_string_index]
                    except (ValueError, IndexError, KeyError):
                        # ValueError if node_idx not in nodes["inputValue"]["index"]
                        # IndexError if lists are not parallel
                        # KeyError if "inputValue", "index", or "value" keys are missing
                        pass  # No input value found or error in parsing

                # Try to get ARIA label or alt text for more context
                aria_label_idx = -1
                alt_text_idx = -1
                node_attributes = nodes["attributes"][node_idx]
                for k_idx, v_idx in zip(node_attributes[0::2], node_attributes[1::2]):
                    attr_name = strings[k_idx]
                    if attr_name == "aria-label":
                        aria_label_idx = v_idx
                    elif attr_name == "alt":
                        alt_text_idx = v_idx

                accessible_name = ""
                if aria_label_idx != -1:
                    accessible_name = strings[aria_label_idx]
                elif alt_text_idx != -1:
                    accessible_name = strings[alt_text_idx]

                final_text = element_text_content or accessible_name or node_name_str
                final_text = re.sub(
                    r"\s+", " ", final_text
                ).strip()  # Normalize whitespace

                if not final_text and node_name_str not in [
                    "input",
                    "textarea",
                    "img",
                ]:  # Keep inputs even if empty
                    continue

                simplified_tag_name = convert_node_name(
                    node_name_str, is_clickable_node
                )

                self.page_element_buffer[id_counter] = {
                    "backend_node_id": nodes["backendNodeId"][node_idx],
                    "node_name": node_name_str,
                    "text_content": final_text,
                    "is_clickable": is_clickable_node,
                    "origin_x": int(x),
                    "origin_y": int(y),
                    "center_x": int(x + width / 2),
                    "center_y": int(y + height / 2),
                }

                element_str = f"<{simplified_tag_name} id={id_counter}>{final_text}</{simplified_tag_name}>"
                if (
                    node_name_str in ["input", "img", "textarea"] and not final_text
                ):  # Self-closing for empty inputs/imgs
                    element_str = f"<{simplified_tag_name} id={id_counter} />"

                elements_of_interest.append(element_str)
                id_counter += 1

        except PlaywrightError as e:
            logger.error(f"Playwright error during crawl: {e}")
            elements_of_interest.append(f"Error during crawl: {e}")
        except Exception as e:
            logger.error(f"Unexpected error during crawl: {e}")
            elements_of_interest.append(f"Unexpected error during crawl: {e}")

        logger.debug(
            f"Crawl parsing time: {time.time() - start_time:.2f} seconds. Found {len(elements_of_interest)} elements."
        )
        return elements_of_interest

    def close(self) -> None:
        """Close the browser and stop Playwright."""
        try:
            if self.page:
                self.page.close()
            if self.browser:
                self.browser.close()
            if self.playwright_context:
                self.playwright_context.stop()
            logger.info("Crawler browser and Playwright context closed.")
        except PlaywrightError as e:
            logger.error(f"Error closing crawler resources: {e}")
        except Exception as e:
            logger.error(f"Unexpected error closing crawler: {e}")


def start_interaction(
    base_model_id: str, objective: str, project_name: str, max_steps: int = 10
) -> None:
    """
    Start an LLM-driven browser interaction session.

    Args:
        base_model_id (str): The base LLM model ID to use for decision making.
        objective (str): The overall objective for the browsing session.
        project_name (str): The name of the current project.
        max_steps (int): Maximum number of interaction steps before stopping.
    """
    if PROMPT_TEMPLATE == "Error: Browser interaction prompt template not found.":
        logger.error("Cannot start browser interaction due to missing prompt template.")
        return

    crawler = Crawler()
    llm = LLM(model_id=base_model_id)
    env = Environment(loader=BaseLoader())
    template = env.from_string(PROMPT_TEMPLATE)

    previous_command: str = "None"

    try:
        crawler.go_to_page("https://www.google.com")  # Start on Google

        for i in range(max_steps):
            logger.info(f"Interaction Step {i + 1}/{max_steps}")

            browser_content_elements: List[str] = crawler.crawl_page_content()
            browser_content_str: str = "\n".join(browser_content_elements)
            current_url: str = crawler.page.url

            crawler.screenshot(project_name)  # Take screenshot for state

            rendered_prompt = template.render(
                objective=objective,
                url=current_url,
                previous_command=previous_command,
                browser_content=browser_content_str[
                    :4500
                ],  # Truncate for context window
            )

            llm_response_str = llm.inference(rendered_prompt, project_name)

            # Parse LLM response
            action_json: Optional[LLMAction] = None
            match = re.search(r"```json\s*(\{.*?\})\s*```", llm_response_str, re.DOTALL)
            if match:
                try:
                    action_json = json.loads(match.group(1))
                except json.JSONDecodeError as e:
                    logger.error(
                        f"Failed to parse LLM JSON response: {e}. Response: {llm_response_str}"
                    )
                    # Decide how to handle - e.g., ask LLM to reformat or terminate
                    break

            if not action_json or not isinstance(action_json, dict):
                logger.error(
                    f"LLM response was not a valid JSON object: {llm_response_str}"
                )
                # Potentially ask LLM to reformat or terminate
                break

            action_type = action_json.get("action_type")
            target_id = action_json.get("target_id")
            text_value = action_json.get("text_value")
            is_complete = action_json.get("is_complete", False)
            justification = action_json.get(
                "justification", "No justification provided."
            )

            logger.info(
                f"LLM Action: {action_type}, Target: {target_id}, Text: {text_value}, Complete: {is_complete}, Justification: {justification}"
            )
            previous_command = f"{action_type} {target_id if target_id else ''} {text_value if text_value else ''}".strip()

            if is_complete:
                logger.info(
                    f"Objective marked as complete by LLM. Justification: {justification}"
                )
                state_manager = AgentState()
                new_state_payload = state_manager.new_state()
                new_state_payload.update(
                    {  # type: ignore
                        "internal_monologue": f"Objective achieved: {justification}",
                        "browser_session": {
                            "url": current_url,
                            "screenshot": None,  # Consider taking a screenshot if needed
                        },
                    }
                )
                state_manager.add_to_current_state(project_name, new_state_payload)
                break

            if action_type == "FAIL":
                logger.warning(
                    f"LLM indicated failure to achieve objective. Reason: {justification}"
                )
                state_manager = AgentState()
                new_state_payload = state_manager.new_state()
                new_state_payload.update(
                    {  # type: ignore
                        "internal_monologue": f"Objective failed: {justification}",
                        "browser_session": {"url": current_url, "screenshot": None},
                    }
                )
                state_manager.add_to_current_state(project_name, new_state_payload)
                break

            if action_type == "SCROLL" and text_value:
                crawler.scroll(text_value.lower())
            elif action_type == "CLICK" and target_id:
                crawler.click(target_id)
            elif action_type == "TYPE" and target_id and text_value is not None:
                crawler.type_into(target_id, text_value, submit=False)
            elif action_type == "TYPESUBMIT" and target_id and text_value is not None:
                crawler.type_into(target_id, text_value, submit=True)
            else:
                logger.warning(f"Unknown or invalid action from LLM: {action_json}")
                # Potentially ask LLM to clarify or provide a valid action

            time.sleep(2)  # Wait for page to load/update after action

            if i == max_steps - 1:
                logger.info("Max interaction steps reached.")
                state_manager = AgentState()
                new_state_payload = state_manager.new_state()
                new_state_payload.update(
                    {  # type: ignore
                        "internal_monologue": "Max interaction steps reached. Objective might not be fully complete.",
                        "browser_session": {
                            "url": crawler.page.url,
                            "screenshot": None,  # Consider crawler.screenshot(project_name) if needed
                        },
                    }
                )
                state_manager.add_to_current_state(project_name, new_state_payload)

    except KeyboardInterrupt:
        logger.info("Browser interaction interrupted by user (Ctrl+C).")
    except Exception as e:
        logger.error(
            f"An unexpected error occurred during browser interaction: {e}",
            exc_info=True,
        )
    finally:
        crawler.close()
        logger.info("Browser interaction session finished.")
