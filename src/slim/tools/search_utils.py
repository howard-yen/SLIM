import json
import requests
from tqdm.contrib.concurrent import thread_map


SEARCH_TOOL = {
    "type": "function",
    "function": {
        "name": "search",
        "description": "Search the web for information. This tool will return a list of urls with a snippet of the content in the url.",
        "parameters": {
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": "The search query."
                },
            },
            "required": [
                "query",
            ],
            "additionalProperties": False
        },
        "strict": True
    }
}

SEARCH_RESPONSE_TOOL = {
    "type": SEARCH_TOOL['type'],
    "name": SEARCH_TOOL['function']['name'],
    "description": SEARCH_TOOL['function']['description'],
    "parameters": SEARCH_TOOL['function']['parameters'],
    "strict": SEARCH_TOOL['function']['strict'],
}

VISIT_TOOL = {
    "type": "function",
    "function": {
        "name": "visit",
        "description": "Visit a url and optionally search for a specific query. If given an empty query, this tool will return the beginning of the page, but searching for a specific query will return the relevant part of the page that contains the query text.",
        "parameters": {
            "type": "object",
            "properties": {
                "url": {
                    "type": "string",
                    "description": "The url to open."
                },
                "query": {
                    "type": "string",
                    "description": "The query to search for in the url. The tool will perform fuzzy matching to find the part of the page that contains the highest textual similarity to the query."
                }
            },
            "required": [
                "url",
            ],
            "additionalProperties": False
        },
    }
}

VISIT_TOOL_NO_QUERY = {
    "type": "function",
    "function": {
        "name": "visit",
        "description": "Visit a url and return the page content.",
        "parameters": {
            "type": "object",
            "properties": {
                "url": {
                    "type": "string",
                    "description": "The url to open."
                }
            },
            "required": [
                "url",
            ],
            "additionalProperties": False
        },
        "strict": True
    }
}

VISIT_RESPONSE_TOOL = {
    "type": VISIT_TOOL['type'],
    "name": VISIT_TOOL['function']['name'],
    "description": VISIT_TOOL['function']['description'],
    "parameters": VISIT_TOOL['function']['parameters'],
}

VISIT_RESPONSE_TOOL_NO_QUERY = {
    "type": VISIT_TOOL_NO_QUERY['type'],
    "name": VISIT_TOOL_NO_QUERY['function']['name'],
    "description": VISIT_TOOL_NO_QUERY['function']['description'],
    "parameters": VISIT_TOOL_NO_QUERY['function']['parameters'],
}


class WebSearchTool():
    def __init__(self, port: int=8006):
        self.url = f"http://localhost:{port}"

    def search(self, query: str, topk: int = 10) -> str:
        """Search the web for information. This tool will return a list of urls that are relevant to the query."""
        if not query or not query.strip():
            return json.dumps({"error": "Please provide a query to search for."})

        payload = json.dumps({"query": query, "topk": topk})
        response = requests.post(self.url + "/search", data=payload)
        return response.json()['output']


    def batch_search(self, query_list: list[str], topk: int = 10) -> str:
        if len(query_list) == 0 or not isinstance(query_list, list) or not all(isinstance(query, str) and query.strip() for query in query_list):
            return json.dumps({"error": "Please provide a list of queries to search for."})

        # use threadmap and call the /search endpoint for each query
        def search_wrapper(query: str) -> dict:
            payload = json.dumps({"query": query, "topk": topk})
            response = requests.post(self.url + "/search", data=payload)
            output = response.json()['output']
            return {
                "query": query,
                "output": output
            }
        results = thread_map(search_wrapper, query_list)
        return json.dumps(results, indent=2)


    def open_url(self, url: str, query: str = "", content_length: int = 10000, scoring_func: str = "rouge", chunking_func: str = "newline") -> str:
        """Open a url and optionally search for a specific query. By default, this tool will return the beginning of the page, but searching for a specific query will return the relevant part of the page that contains the query text."""
        if not url or not isinstance(url, str) or not url.strip():
            return "Please provide a url to open."

        payload = {"url": url, "query": query, "content_length": content_length, "scoring_func": scoring_func, "chunking_func": chunking_func}
        payload = json.dumps(payload)
        response = requests.post(self.url + "/open_url", data=payload)
        try: 
            out = response.json()
            return out['output']
        except Exception as e:
            print("Open url error: " + str(e))
            print(response)
            print(response.text)
            return "Open url error: " + str(e)


    def batch_browse(self, url_list: list[str], query_list: list[str] | None = None, content_length: int = 10000, scoring_func: str = "rouge", chunking_func: str = "newline") -> str:
        if len(url_list) == 0 or not isinstance(url_list, list) or not all(isinstance(url, str) and url.strip() for url in url_list):
            return json.dumps({"error": "Please provide a non-empty list of urls to browse."})

        if query_list is None or len(query_list) == 0:
            query_list = [""] * len(url_list)

        # the query list can be None (treat as no query for all urls) and each query can be empty (treat as no query for the url)
        if not isinstance(query_list, list) or not all(isinstance(query, str) for query in query_list):
            return json.dumps({"error": "Please provide a list of queries to search for."})

        def browse_wrapper(url: str, query: str) -> dict:
            payload = {"url": url, "query": query, "content_length": content_length, "scoring_func": scoring_func, "chunking_func": chunking_func}
            payload = json.dumps(payload)
            response = requests.post(self.url + "/open_url", data=payload)
            output = response.json()['output']
            return {
                "url": url,
                "query": query,
                "output": output
            }
        results = thread_map(browse_wrapper, url_list, query_list)
        return json.dumps(results, indent=2)


    def search_open_url(self, query: str, topk: int = 10, content_length: int = 10000) -> str:
        """Search the web for information, and also open all the urls. Following search-open-url's format."""
        if not query or not query.strip():
            return "Search error: Please provide a query to search for."

        payload = json.dumps({"query": query, "topk": topk, "content_length": content_length})
        response = requests.post(self.url + "/search_open_url", data=payload)
        return response.json()['output']

    def search_o1(self, query: str, topk: int = 10) -> str:
        """Search the web for information. Following search-o1's format."""
        if not query or not query.strip():
            return json.dumps({"output": "Search error: Please provide a query to search for.", "search_results": []})

        payload = json.dumps({"query": query, "topk": topk})
        response = requests.post(self.url + "/search_o1", data=payload)
        try:
            out = response.json()
            return out
        except Exception as e:
            print("Search o1 error: " + str(e))
            print(response)
            print(response.text)
            return {"output": "Search error: " + str(e), "search_results": []}
