"""
Provides classes for interacting with various search engines.

This module defines a base abstract class `BaseSearch` and concrete implementations
for Bing, Google, and DuckDuckGo search engines. Each search engine class is
responsible for fetching and parsing search results from its respective API or source.
"""
import re
import orjson
from abc import ABC, abstractmethod
from html import unescape
from typing import List, Dict, Optional, Any, Tuple
from urllib.parse import unquote

import requests
from curl_cffi.requests import AsyncSession, RequestsError, TimeoutError as CurlTimeoutError


from src.config import Config
from src.logger import Logger

logger = Logger()
DEFAULT_USER_AGENT = (
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
    "(KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36"
)
DEFAULT_TIMEOUT = 10  # seconds


SearchResult = Dict[str, str] # Type alias for a single search result item


class BaseSearch(ABC):
    """
    Abstract base class for search engine integrations.

    Defines a common interface that all search engine classes must implement.
    """

    def __init__(self) -> None:
        """Initialize BaseSearch."""
        self.query_result: Optional[List[SearchResult]] = None

    @abstractmethod
    def search(self, query: str) -> Optional[List[SearchResult]]:
        """
        Perform a search for the given query.

        Args:
            query (str): The search query string.

        Returns:
            Optional[List[SearchResult]]: A list of search result dictionaries,
                                          where each dictionary contains 'title',
                                          'href', and 'body' keys, or None if an error occurs.
        """
        pass

    def get_first_link(self) -> Optional[str]:
        """
        Get the URL of the first search result.

        Returns:
            Optional[str]: The URL of the first search result, or None if no results
                           are available or an error occurred.
        """
        if self.query_result and len(self.query_result) > 0:
            return self.query_result[0].get("href")
        return None

    def get_results(self) -> Optional[List[SearchResult]]:
        """
        Get all processed search results.

        Returns:
            Optional[List[SearchResult]]: A list of search result dictionaries,
                                          or None if no search has been performed.
        """
        return self.query_result


class BingSearch(BaseSearch):
    """
    Provides search functionality using the Bing Web Search API.
    """

    def __init__(self) -> None:
        """
        Initialize BingSearch with API key and endpoint from configuration.
        """
        super().__init__()
        self.config: Config = Config()
        self.bing_api_key: Optional[str] = self.config.get_bing_api_key()
        self.bing_api_endpoint: str = self.config.get_bing_api_endpoint()
        if not self.bing_api_key:
            logger.warning("Bing API key not found in configuration. BingSearch will not work.")


    def search(self, query: str) -> Optional[List[SearchResult]]:
        """
        Perform a search using the Bing Web Search API.

        Args:
            query (str): The search query string.

        Returns:
            Optional[List[SearchResult]]: A list of search results, or None if an error occurs
                                          or the API key is missing.
        """
        if not self.bing_api_key:
            logger.error("Bing API key is missing. Cannot perform search.")
            return None

        headers: Dict[str, str] = {
            "Ocp-Apim-Subscription-Key": self.bing_api_key,
            "User-Agent": DEFAULT_USER_AGENT,
        }
        params: Dict[str, str] = {"q": query, "mkt": "en-US", "count": "5"} # Fetch 5 results

        try:
            response = requests.get(
                self.bing_api_endpoint, headers=headers, params=params, timeout=DEFAULT_TIMEOUT
            )
            response.raise_for_status()  # Raises HTTPError for bad responses (4XX or 5XX)
            raw_results: Dict[str, Any] = response.json()
            
            processed_results: List[SearchResult] = []
            if raw_results.get("webPages") and raw_results["webPages"].get("value"):
                for item in raw_results["webPages"]["value"]:
                    processed_results.append({
                        "title": item.get("name", ""),
                        "href": item.get("url", ""),
                        "body": item.get("snippet", "")
                    })
            self.query_result = processed_results
            return self.query_result
        except requests.exceptions.RequestException as e:
            logger.error(f"Bing Search API request failed: {e}")
            return None
        except (KeyError, ValueError) as e: # Catches JSON parsing errors or missing keys
            logger.error(f"Failed to parse Bing Search API response: {e}")
            return None


class GoogleSearch(BaseSearch):
    """
    Provides search functionality using the Google Custom Search JSON API.
    """

    def __init__(self) -> None:
        """
        Initialize GoogleSearch with API key, engine ID, and endpoint from configuration.
        """
        super().__init__()
        self.config: Config = Config()
        self.api_key: Optional[str] = self.config.get_google_search_api_key()
        self.engine_id: Optional[str] = self.config.get_google_search_engine_id()
        self.api_endpoint: str = self.config.get_google_search_api_endpoint()
        if not self.api_key or not self.engine_id:
            logger.warning("Google Search API key or Engine ID not found. GoogleSearch will not work.")


    def search(self, query: str) -> Optional[List[SearchResult]]:
        """
        Perform a search using the Google Custom Search JSON API.

        Args:
            query (str): The search query string.

        Returns:
            Optional[List[SearchResult]]: A list of search results, or None if an error occurs
                                          or API credentials are missing.
        """
        if not self.api_key or not self.engine_id:
            logger.error("Google Search API key or Engine ID missing. Cannot perform search.")
            return None

        params: Dict[str, str] = {
            "key": self.api_key,
            "cx": self.engine_id,
            "q": query,
            "num": 5 # Fetch 5 results
        }
        headers: Dict[str, str] = {"User-Agent": DEFAULT_USER_AGENT}

        try:
            logger.info(f"Searching Google for: {query}")
            response = requests.get(self.api_endpoint, params=params, headers=headers, timeout=DEFAULT_TIMEOUT)
            response.raise_for_status()
            raw_results: Dict[str, Any] = response.json()

            processed_results: List[SearchResult] = []
            if raw_results.get("items"):
                for item in raw_results["items"]:
                    processed_results.append({
                        "title": item.get("title", ""),
                        "href": item.get("link", ""),
                        "body": item.get("snippet", "")
                    })
            self.query_result = processed_results
            return self.query_result
        except requests.exceptions.RequestException as e:
            logger.error(f"Google Search API request failed: {e}")
            return None
        except (KeyError, ValueError) as e:
            logger.error(f"Failed to parse Google Search API response: {e}")
            return None


class DuckDuckGoSearch(BaseSearch):
    """
    Provides search functionality using DuckDuckGo (via curl_cffi to simulate browser).

    Note: This class uses web scraping techniques that might be fragile if
    DuckDuckGo changes its website structure or protections.
    The `curl_cffi` library is used to bypass potential bot detection.
    """

    def __init__(self) -> None:
        """Initialize DuckDuckGoSearch with an async session."""
        super().__init__()
        self.asession: Optional[AsyncSession] = None
        try:
            self.asession = AsyncSession(
                impersonate="chrome", allow_redirects=False
            )
            if self.asession: # mypy check
                self.asession.headers["Referer"] = "https://duckduckgo.com/"
                self.asession.headers["User-Agent"] = DEFAULT_USER_AGENT
        except Exception as e:
            logger.error(f"Failed to initialize curl_cffi AsyncSession for DuckDuckGo: {e}")
            self.asession = None


    async def _get_url_content(self, method: str, url: str, data: Optional[Dict[str, str]] = None, params: Optional[Dict[str, str]] = None) -> Optional[bytes]:
        """
        Fetch URL content using the async session.

        Args:
            method (str): HTTP method (GET or POST).
            url (str): URL to fetch.
            data (Optional[Dict[str, str]]): POST data.
            params (Optional[Dict[str, str]]): URL parameters for GET.

        Returns:
            Optional[bytes]: Response content as bytes, or None on error.
        """
        if not self.asession:
            logger.error("DuckDuckGoSearch session not initialized.")
            return None
        try:
            resp = await self.asession.request(method, url, data=data, params=params, timeout=DEFAULT_TIMEOUT)
            if resp.status_code == 200:
                return resp.content
            # DDG might return 202, 301, 403 for rate limiting or other issues
            logger.warning(f"DuckDuckGo request to {url} returned status {resp.status_code}.")
            return None # Consider specific error handling for these codes if needed
        except CurlTimeoutError:
            logger.error("DuckDuckGo request timed out.")
            return None
        except RequestsError as e: # More generic curl_cffi error
            logger.error(f"DuckDuckGo request error: {e}")
            return None
        except Exception as e:
            logger.error(f"Unexpected error during DuckDuckGo request: {e}")
            return None


    async def _duckduckgo_search_async(self, query: str) -> Optional[List[SearchResult]]:
        """
        Perform the DuckDuckGo search asynchronously.

        Args:
            query (str): The search query.

        Returns:
            Optional[List[SearchResult]]: A list of search results, or None on error.
        """
        if not self.asession: return None

        # Get VQD token
        vqd_payload = {"q": query}
        resp_html_bytes = await self._get_url_content("POST", "https://duckduckgo.com/", data=vqd_payload)
        if not resp_html_bytes:
            logger.error("Failed to get VQD token from DuckDuckGo.")
            return None

        vqd = self._extract_vqd(resp_html_bytes)
        if not vqd:
            logger.error("Failed to extract VQD token from DuckDuckGo HTML.")
            return None

        # Perform search using VQD
        search_params = {
            "q": query, "kl": "en-us", "p": "1", "s": "0", "df": "", "vqd": vqd, "ex": ""
        }
        links_resp_bytes = await self._get_url_content("GET", "https://links.duckduckgo.com/d.js", params=search_params)
        if not links_resp_bytes:
            logger.error("Failed to get search links from DuckDuckGo.")
            return None

        page_data = self._text_extract_json(links_resp_bytes)
        if not page_data:
            logger.error("Failed to extract JSON data from DuckDuckGo links response.")
            return None

        results: List[SearchResult] = []
        for row in page_data:
            href = row.get("u")
            if href and href != f"http://www.google.com/search?q={query}": # Filter out Google search links
                body = self._normalize_text(row.get("a", ""))
                if body: # Ensure body is not empty after normalization
                    result: SearchResult = {
                        "title": self._normalize_text(row.get("t", "")),
                        "href": self._normalize_url(href),
                        "body": body,
                    }
                    results.append(result)
        return results

    def search(self, query: str) -> Optional[List[SearchResult]]:
        """
        Perform a search using DuckDuckGo.

        This method runs the asynchronous _duckduckgo_search_async method
        in a new event loop.

        Args:
            query (str): The search query string.

        Returns:
            Optional[List[SearchResult]]: A list of search results, or None if an error occurs.
        """
        if not self.asession:
            logger.error("DuckDuckGoSearch session not initialized. Cannot perform search.")
            return None
        try:
            # Running async code from a sync method requires an event loop
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            self.query_result = loop.run_until_complete(self._duckduckgo_search_async(query))
            loop.close()
        except Exception as e:
            logger.error(f"Error running DuckDuckGo async search: {e}")
            self.query_result = None
        return self.query_result


    @staticmethod
    def _extract_vqd(html_bytes: bytes) -> Optional[str]:
        """Extract VQD token from HTML content."""
        patterns: List[Tuple[bytes, int, bytes]] = [
            (b'vqd="', 5, b'"'), (b"vqd=", 4, b"&"), (b"vqd='", 5, b"'")
        ]
        for start_pattern, offset, end_pattern in patterns:
            try:
                start_index = html_bytes.index(start_pattern) + offset
                end_index = html_bytes.index(end_pattern, start_index)
                return html_bytes[start_index:end_index].decode()
            except ValueError:
                continue
        return None

    @staticmethod
    def _text_extract_json(html_bytes: bytes) -> Optional[List[Dict[str, Any]]]:
        """Extract JSON data from DDG links page."""
        try:
            # Pattern might need adjustment if DDG changes its JS structure
            start_index = html_bytes.index(b"DDG.pageLayout.load('d',") + 24
            end_index = html_bytes.index(b");DDG.duckbar.load(", start_index)
            json_data = orjson.loads(html_bytes[start_index:end_index])
            return json_data if isinstance(json_data, list) else None
        except (ValueError, orjson.JSONDecodeError) as e: # More specific exceptions
            logger.error(f"Error extracting or parsing JSON from DDG links: {e}")
            return None

    @staticmethod
    def _normalize_url(url: str) -> str:
        """Normalize and unquote a URL."""
        return unquote(url.replace(" ", "+")) if url else ""

    @staticmethod
    def _normalize_text(raw_html: str) -> str:
        """Remove HTML tags and unescape HTML entities."""
        return unescape(re.sub("<.*?>", "", raw_html)).strip() if raw_html else ""
