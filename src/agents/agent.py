"""
Defines the core Agent class that orchestrates various sub-agents to process user prompts.

The Agent class is the central engine of the Devika application. It initializes
all necessary components, including planners, researchers, coders, and other
specialized agents. It manages the overall workflow, from understanding the user's
initial prompt to generating code, documentation, and reports.
"""

import asyncio
import json
import platform
import time
from typing import Any, Dict, List, Optional, Tuple

import tiktoken
from tiktoken.core import Encoding

from src.bert.sentence import SentenceBert
from src.browser import Browser, start_interaction

# from src.memory import KnowledgeBase # Not used currently
from src.browser.search import BaseSearch, BingSearch, DuckDuckGoSearch, GoogleSearch
from src.documenter.pdf import PDF
from src.filesystem import ReadCode
from src.logger import Logger
from src.project import ProjectManager
from src.services import Netlify
from src.socket_instance import emit_agent
from src.state import AgentState, StateType

from .action import Action
from .answer import Answer
from .coder import Coder
from .decision import Decision
from .feature import Feature
from .formatter import Formatter
from .internal_monologue import InternalMonologue
from .patcher import Patcher
from .planner import Planner
from .reporter import Reporter
from .researcher import Researcher
from .runner import Runner


class Agent:
    """
    The central agent orchestrating the workflow of sub-agents to process user prompts.

    This class initializes and coordinates various specialized agents (Planner, Researcher,
    Coder, etc.) to handle tasks like planning, research, code generation, and reporting.
    It manages the project's state, message history, and interaction with external
    services like search engines and deployment platforms.

    Attributes:
        logger (Logger): Instance for logging messages.
        collected_context_keywords (List[str]): A list to accumulate contextual keywords.
        planner (Planner): Agent for generating step-by-step plans.
        researcher (Researcher): Agent for conducting research based on plans.
        formatter (Formatter): Agent for formatting research results.
        coder (Coder): Agent for generating code.
        action (Action): Agent for deciding the next action based on conversation.
        internal_monologue (InternalMonologue): Agent for generating internal thoughts.
        answer (Answer): Agent for answering user questions based on context.
        runner (Runner): Agent for executing code.
        feature (Feature): Agent for implementing new features.
        patcher (Patcher): Agent for fixing bugs in code.
        reporter (Reporter): Agent for generating reports.
        decision (Decision): Agent for making high-level decisions.
        project_manager (ProjectManager): Manages project-related data.
        agent_state (AgentState): Manages the agent's state for projects.
        search_engine_name (str): Name of the search engine to use (e.g., "bing", "google").
        tokenizer (Encoding): Tiktoken tokenizer for counting tokens.
        base_model (str): The base LLM model ID being used.
    """

    def __init__(self, base_model: str, search_engine: str):
        """
        Initialize the Agent.

        Args:
            base_model (str): The base model ID for LLMs. Must not be empty.
            search_engine (str): The name of the search engine to use.

        Raises:
            ValueError: If base_model is not provided.
        """
        if not base_model:
            raise ValueError("base_model is required")

        self.logger: Logger = Logger()
        self.base_model: str = (
            base_model  # Store base_model for use in browser_interaction
        )

        self.collected_context_keywords: List[str] = []

        # Initialize Agents
        self.planner: Planner = Planner(base_model=base_model)
        self.researcher: Researcher = Researcher(base_model=base_model)
        self.formatter: Formatter = Formatter(base_model=base_model)
        self.coder: Coder = Coder(base_model=base_model)
        self.action: Action = Action(base_model=base_model)
        self.internal_monologue: InternalMonologue = InternalMonologue(
            base_model=base_model
        )
        self.answer: Answer = Answer(base_model=base_model)
        self.runner: Runner = Runner(base_model=base_model)
        self.feature: Feature = Feature(base_model=base_model)
        self.patcher: Patcher = Patcher(base_model=base_model)
        self.reporter: Reporter = Reporter(base_model=base_model)
        self.decision: Decision = Decision(base_model=base_model)

        # Initialize Managers and Services
        self.project_manager: ProjectManager = ProjectManager()
        self.agent_state: AgentState = AgentState()
        self.search_engine_name: str = search_engine
        self.tokenizer: Encoding = tiktoken.get_encoding("cl100k_base")

    async def _open_page_and_extract_data(
        self, project_name: str, url: str
    ) -> Tuple[str, str]:
        """
        Open a web page, take a screenshot, and extract text content.

        Args:
            project_name (str): The name of the current project.
            url (str): The URL to open.

        Returns:
            Tuple[str, str]: A tuple containing the screenshot data (base64) and
                             extracted text content.
        """
        browser_instance = await Browser().start()
        try:
            await browser_instance.go_to(url)
            _, raw_screenshot_data = await browser_instance.screenshot(project_name)
            extracted_text = await browser_instance.extract_text()
        finally:
            await browser_instance.close()
        return raw_screenshot_data, extracted_text

    def _get_search_engine(self) -> BaseSearch:
        """
        Get an instance of the specified search engine.

        Returns:
            BaseSearch: An instance of the search engine client.
        """
        if self.search_engine_name == "bing":
            return BingSearch()
        if self.search_engine_name == "google":
            return GoogleSearch()
        return DuckDuckGoSearch()

    def search_queries(self, queries: List[str], project_name: str) -> Dict[str, str]:
        """
        Perform web searches for a list of queries and format the results.

        Args:
            queries (List[str]): A list of search queries.
            project_name (str): The name of the current project.

        Returns:
            Dict[str, str]: A dictionary where keys are queries and values are
                            the formatted search results.
        """
        results: Dict[str, str] = {}
        # knowledge_base = KnowledgeBase() # Not currently used
        web_search_client = self._get_search_engine()

        self.logger.info(f"Using search engine: {self.search_engine_name}")

        for query in queries:
            query_stripped = query.strip().lower()
            if not query_stripped:
                continue

            # TODO: Re-evaluate knowledge base integration if needed
            # knowledge = knowledge_base.get_knowledge(tag=query_stripped)
            # if knowledge:
            #     results[query_stripped] = knowledge
            #     continue

            web_search_client.search(query_stripped)
            link = web_search_client.get_first_link()

            if not link:
                self.logger.warning(f"No link found for query: {query_stripped}")
                results[query_stripped] = "No information found."
                continue

            self.logger.info(f"Processing link for query '{query_stripped}': {link}")

            try:
                # Run async method in a new event loop for sync context
                loop = asyncio.new_event_loop()
                asyncio.set_event_loop(loop)
                raw_screenshot, data = loop.run_until_complete(
                    self._open_page_and_extract_data(project_name, link)
                )
                loop.close()

                emit_agent(
                    "screenshot",
                    {"data": raw_screenshot, "project_name": project_name},
                    broadcast=False,  # Assuming screenshot is specific to a session
                )
                results[query_stripped] = self.formatter.execute(data, project_name)
                self.logger.info(
                    f"Successfully processed search results for: {query_stripped}"
                )
            except Exception as e:
                self.logger.error(
                    f"Error processing query '{query_stripped}' with link {link}: {e}"
                )
                results[query_stripped] = f"Error processing search results: {e}"

            # knowledge_base.add_knowledge(tag=query_stripped, contents=results[query_stripped])
        return results

    def update_contextual_keywords(self, sentence: str) -> List[str]:
        """
        Update and return the list of collected contextual keywords from a sentence.

        Args:
            sentence (str): The sentence/prompt to extract keywords from.

        Returns:
            List[str]: The updated list of collected context keywords.
        """
        keywords: List[Tuple[str, float]] = SentenceBert(sentence).extract_keywords()
        self.collected_context_keywords.extend([keyword[0] for keyword in keywords])
        return self.collected_context_keywords

    def _handle_decision_function(
        self, function_name: str, args: Dict[str, Any], project_name: str
    ) -> None:
        """
        Handle the execution of a function determined by the Decision agent.

        Args:
            function_name (str): The name of the function to execute.
            args (Dict[str, Any]): The arguments for the function.
            project_name (str): The name of the current project.
        """
        if function_name == "git_clone":
            url = args.get("url")
            if url:
                # TODO: Implement git clone functionality
                self.logger.info(f"Placeholder: Git clone URL: {url}")
                self.project_manager.add_message_from_devika(
                    project_name,
                    f"Attempting to clone repository from {url} (not implemented yet).",
                )
            else:
                self.project_manager.add_message_from_devika(
                    project_name, "Git clone URL not provided."
                )

        elif function_name == "generate_pdf_document":
            user_prompt = args.get("user_prompt", "Default prompt for PDF generation.")
            markdown = self.reporter.execute(
                [user_prompt], "", project_name
            )  # Assuming empty code markdown for this context
            pdf_file_path = PDF().markdown_to_pdf(
                markdown, project_name
            )  # Returns path
            self.logger.info(f"Generated PDF document at: {pdf_file_path}")

            project_name_space_url = project_name.replace(" ", "%20")
            pdf_download_url = f"http://127.0.0.1:1337/api/download-project-pdf?project_name={project_name_space_url}"
            response_msg = f"I have generated the PDF document. You can download it from here: {pdf_download_url}"
            self.project_manager.add_message_from_devika(project_name, response_msg)

        elif function_name == "browser_interaction":
            user_prompt = args.get(
                "user_prompt", "Default prompt for browser interaction."
            )
            # Assuming start_interaction is an async function or handles its own loop
            asyncio.run(
                start_interaction(self.base_model, user_prompt, project_name)
            )  # Pass base_model

        elif function_name == "coding_project":
            user_prompt = args.get("user_prompt", "Default prompt for coding project.")
            self._execute_coding_project_flow(user_prompt, project_name)
        else:
            self.project_manager.add_message_from_devika(
                project_name,
                f"Function {function_name} is not recognized or implemented.",
            )

    def _execute_coding_project_flow(self, user_prompt: str, project_name: str) -> None:
        """
        Execute the full coding project flow: plan, research, code.

        Args:
            user_prompt (str): The user's prompt for the coding project.
            project_name (str): The name of the project.
        """
        plan = self.planner.execute(user_prompt, project_name)
        # planner_response = self.planner.parse_response(plan) # Not used

        research = self.researcher.execute(
            plan, self.collected_context_keywords, project_name
        )
        search_results: Dict[str, str] = {}
        if research.get("queries"):
            search_results = self.search_queries(research["queries"], project_name)

        code = self.coder.execute(
            step_by_step_plan=plan,
            user_context=research.get("ask_user", ""),
            search_results=search_results,
            project_name=project_name,
        )
        self.coder.save_code_to_project(code, project_name)
        self.project_manager.add_message_from_devika(
            project_name, "I have finished coding the project based on the plan."
        )

    def make_decision(self, prompt: str, project_name: str) -> None:
        """
        Execute the decision-making flow based on the user's prompt.

        This involves using the Decision agent to determine the appropriate function
        to call and then executing that function.

        Args:
            prompt (str): The user's prompt.
            project_name (str): The name of the current project.
        """
        decision_result = self.decision.execute(prompt, project_name)

        for item in decision_result:
            function_name = item.get("function")
            args = item.get("args", {})
            reply = item.get("reply")

            if reply:
                self.project_manager.add_message_from_devika(project_name, reply)

            if function_name:
                self._handle_decision_function(function_name, args, project_name)
            else:
                self.logger.warning(f"No function specified in decision item: {item}")

    def subsequent_execute(self, prompt: str, project_name: str) -> None:
        """
        Handle subsequent interactions after the initial project setup and execution.

        This method determines the appropriate action (e.g., answer, run, deploy)
        based on the conversation history and the latest user prompt.

        Args:
            prompt (str): The latest user prompt.
            project_name (str): The name of the current project.
        """
        os_system: str = platform.platform()
        self.agent_state.set_agent_active(project_name, True)

        conversation: List[str] = self.project_manager.get_all_messages_formatted(
            project_name
        )
        code_markdown: str = ReadCode(project_name).code_set_to_markdown()

        agent_action_response, agent_action = self.action.execute(
            conversation, project_name
        )
        self.project_manager.add_message_from_devika(
            project_name, agent_action_response
        )
        self.logger.info(f"Action decided: {agent_action}")

        if agent_action == "answer":
            response_text = self.answer.execute(
                conversation=conversation,
                code_markdown=code_markdown,
                project_name=project_name,
            )
            self.project_manager.add_message_from_devika(project_name, response_text)

        elif agent_action == "run":
            project_path = self.project_manager.get_project_path(project_name)
            # Runner agent's execute method is expected to handle its own messaging
            self.runner.execute(
                conversation=conversation,
                code_markdown=code_markdown,
                os_system=os_system,
                project_path=project_path,
                project_name=project_name,
            )

        elif agent_action == "deploy":
            deploy_metadata = Netlify().deploy(project_name)
            deploy_url = deploy_metadata.get(
                "deploy_url", "Deployment URL not available."
            )
            response_data = {
                "message": "Done! I deployed your project on Netlify.",
                "deploy_url": deploy_url,
            }
            self.project_manager.add_message_from_devika(
                project_name, json.dumps(response_data, indent=4)
            )

        elif agent_action == "feature":
            code_generated = self.feature.execute(
                conversation=conversation,
                code_markdown=code_markdown,
                system_os=os_system,
                project_name=project_name,
            )
            self.logger.info(f"Feature code generated:\n{code_generated}")
            self.feature.save_code_to_project(code_generated, project_name)
            self.project_manager.add_message_from_devika(
                project_name, "I've added the new features to the project."
            )

        elif agent_action == "bug":
            # Assuming 'prompt' here is the error message or bug description
            code_patched = self.patcher.execute(
                conversation=conversation,
                code_markdown=code_markdown,
                commands=None,  # Or provide relevant commands if available
                error=prompt,
                system_os=os_system,
                project_name=project_name,
            )
            self.logger.info(f"Patched code generated:\n{code_patched}")
            self.patcher.save_code_to_project(code_patched, project_name)
            self.project_manager.add_message_from_devika(
                project_name, "I've attempted to fix the bug."
            )

        elif agent_action == "report":
            markdown_report = self.reporter.execute(
                conversation, code_markdown, project_name
            )
            pdf_file_path = PDF().markdown_to_pdf(markdown_report, project_name)
            self.logger.info(f"Generated PDF report at: {pdf_file_path}")

            project_name_url_encoded = project_name.replace(" ", "%20")
            pdf_download_url = f"http://127.0.0.1:1337/api/download-project-pdf?project_name={project_name_url_encoded}"
            response_msg = f"I have generated the PDF document. You can download it from here: {pdf_download_url}"
            self.project_manager.add_message_from_devika(project_name, response_msg)

        self.agent_state.set_agent_active(project_name, False)
        self.agent_state.set_agent_completed(
            project_name, True
        )  # Mark as completed after each subsequent action for now

    def execute(
        self, prompt: str, project_name_from_user: Optional[str] = None
    ) -> None:
        """
        The main execution flow of the agent.

        This method handles the initial prompt from the user, orchestrates the planning,
        research, and coding phases, and manages interactions with the user.

        Args:
            prompt (str): The user's initial prompt or query.
            project_name_from_user (Optional[str]): The name of the project if provided by the user.
                                                    If None, a new project name is generated.
        """
        current_project_name: str
        if project_name_from_user:
            current_project_name = project_name_from_user
            self.project_manager.add_message_from_user(current_project_name, prompt)
        else:
            # This path might need review: if project_name_from_user is None,
            # planner.execute gets None, which might not be intended.
            # Assuming planner can handle this or a default name is used.
            temp_plan_for_name = self.planner.execute(
                prompt, None
            )  # Temporary call to get project name
            temp_planner_response = self.planner.parse_response(temp_plan_for_name)
            current_project_name = temp_planner_response.get(
                "project", "new-project"
            )  # Fallback name
            self.project_manager.create_project(current_project_name)
            self.project_manager.add_message_from_user(current_project_name, prompt)

        self.logger.info(f"Executing agent for project: {current_project_name}")
        self.agent_state.set_agent_active(current_project_name, True)

        # Initial planning phase
        plan_str: str = self.planner.execute(prompt, current_project_name)
        self.logger.info(f"Initial plan:\n{plan_str}")

        planner_response: Dict[str, Any] = self.planner.parse_response(plan_str)
        reply: str = planner_response.get("reply", "Okay, I'm working on it.")
        focus: str = planner_response.get("focus", "")
        plans: List[Dict[str, str]] = planner_response.get("plans", [])
        # summary: str = planner_response.get("summary", "") # Not currently used

        self.project_manager.add_message_from_devika(current_project_name, reply)
        if plans:  # Only add plans if they exist
            self.project_manager.add_message_from_devika(
                current_project_name, f"Here's my plan:\n{json.dumps(plans, indent=4)}"
            )

        if focus:
            self.update_contextual_keywords(focus)
            self.logger.info(
                f"Updated context keywords: {self.collected_context_keywords}"
            )

        internal_monologue_text: str = self.internal_monologue.execute(
            current_prompt=plan_str, project_name=current_project_name
        )
        self.logger.info(f"Internal Monologue:\n{internal_monologue_text}")

        current_state: StateType = self.agent_state.new_state()
        current_state["internal_monologue"] = internal_monologue_text
        self.agent_state.add_to_current_state(current_project_name, current_state)

        # Research phase
        research_info: Dict[str, Any] = self.researcher.execute(
            plan_str, self.collected_context_keywords, project_name=current_project_name
        )
        self.logger.info(f"Research Information:\n{research_info}")

        queries: List[str] = research_info.get("queries", [])
        ask_user_query: str = research_info.get("ask_user", "")
        search_results: Dict[str, str] = {}

        if queries:
            queries_combined = ", ".join(queries)
            self.project_manager.add_message_from_devika(
                current_project_name,
                f"I am browsing the web to research the following queries: {queries_combined}.\n"
                "If I need anything, I will make sure to ask you.",
            )
            search_results = self.search_queries(queries, current_project_name)
        elif not ask_user_query:  # Only if no queries AND no ask_user
            self.project_manager.add_message_from_devika(
                current_project_name, "I think I can proceed without searching the web."
            )

        user_feedback: str = "Nothing specific from the user at this phase."
        if ask_user_query:
            self.project_manager.add_message_from_devika(
                current_project_name, ask_user_query
            )
            self.agent_state.set_agent_active(current_project_name, False)
            self.logger.info("Waiting for user feedback...")
            got_user_feedback = False
            while not got_user_feedback:
                time.sleep(5)  # Polling interval
                latest_user_msg = self.project_manager.get_latest_message_from_user(
                    current_project_name
                )
                is_user_msg_new = (
                    self.project_manager.validate_last_message_is_from_user(
                        current_project_name
                    )
                )
                if (
                    latest_user_msg
                    and is_user_msg_new
                    and latest_user_msg["message"] != ask_user_query
                ):  # Ensure it's a new message
                    user_feedback = latest_user_msg["message"]
                    got_user_feedback = True
                    self.project_manager.add_message_from_devika(
                        current_project_name, "Thanks for the feedback! 🙌"
                    )
                    self.logger.info(f"Received user feedback: {user_feedback}")

            self.agent_state.set_agent_active(current_project_name, True)

        # Coding phase
        generated_code: str = self.coder.execute(
            step_by_step_plan=plan_str,
            user_context=user_feedback,
            search_results=search_results,
            project_name=current_project_name,
        )
        self.logger.info(f"Generated Code:\n{generated_code}")
        self.coder.save_code_to_project(generated_code, current_project_name)

        self.agent_state.set_agent_active(current_project_name, False)
        self.agent_state.set_agent_completed(current_project_name, True)
        self.project_manager.add_message_from_devika(
            current_project_name,
            "I have completed the initial setup and coding based on your request. "
            "If you would like me to do anything else, please let me know.",
        )
