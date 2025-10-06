import os
import json
import requests
import argparse
from functools import lru_cache
import time
import asyncio
from typing import Dict, List, Tuple

import uvicorn
from fastapi import FastAPI
from pydantic import BaseModel
from urllib.parse import urlparse
from rouge_score import rouge_scorer
from diskcache import Cache
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor, as_completed
import bm25s
import Stemmer

from crawl4ai.processors.pdf import PDFCrawlerStrategy, PDFContentScrapingStrategy
from crawl4ai.markdown_generation_strategy import DefaultMarkdownGenerator
from crawl4ai.content_filter_strategy import BM25ContentFilter, PruningContentFilter
from crawl4ai import CrawlerRunConfig, AsyncWebCrawler, BrowserConfig

HEADERS = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/58.0.3029.110 Safari/537.3"
}
scorer = rouge_scorer.RougeScorer(['rougeL'], use_stemmer=True)
cache = Cache("./cache")

class SearchRequest(BaseModel):
    query: str
    topk: int = 10
    content_length: int = 10000

class OpenUrlRequest(BaseModel):
    url: str
    query: str
    content_length: int = 10000
    scoring_func: str = "rouge"
    chunking_func: str = "newline"


def detect_content_type(url: str) -> str:
    parsed_url = urlparse(url)
    if parsed_url.path.lower().endswith('.pdf'):
        return 'pdf'

    try:
        response = requests.head(url, headers=HEADERS, timeout=(3, 5))
        response.raise_for_status()
        content_type = response.headers.get('Content-Type', '')
        return "pdf" if "pdf" in content_type else "html"
    except Exception as e:
        return "html"


def find_snippet(texts: List[str], snippet: str, num_characters: int = 4000, scoring_func: str = "rouge"):
    """
    We iterate through the texts, break them into chunks of 1000 characters, and then use the scoring function to find the best chunk.
    The text is already split into arbitrary chunks.
    The scoring function can be "rouge" or "bm25".
    We also take the surrounding text of the snippet to fill up the num_characters.
    """
    positions = []
    start = 0
    best_recall = 0
    best_idx = 0

    if scoring_func == 'bm25':
        stemmer = Stemmer.Stemmer('english')
        corpus_tokens = bm25s.tokenize(texts, stopwords='en', stemmer=stemmer)
        retriever = bm25s.BM25()
        retriever.index(corpus_tokens)
        query_tokens = bm25s.tokenize(snippet, stopwords='en', stemmer=stemmer)
        results, scores = retriever.retrieve(query_tokens, k=1)
        best_idx = int(results[0, 0])

    for i, text in enumerate(texts):
        if scoring_func == "rouge":
            score = scorer.score(target=snippet, prediction=text)
            recall = score['rougeL'].recall
            if recall > best_recall:
                best_recall = recall
                best_idx = i
        positions.append((start, start + len(text)))
        start += len(text) + 1

    best_len = len(texts[best_idx])
    num_characters = num_characters - best_len
    final_text = []
    for i, pos in enumerate(positions):
        if (pos[0] >= positions[best_idx][0] - num_characters/2 and pos[1] <= positions[best_idx][1] + num_characters/2) or i == best_idx:
            final_text.append(texts[i])

    return "\n".join(final_text)


async def scrape_pdf(url: str) -> Tuple[bool, str, str]:
    import fitz
    response = requests.get(url, headers=HEADERS, timeout=(3, 5))  # (connect timeout, read timeout)
    response.raise_for_status()
    with fitz.open(stream=response.content, filetype="pdf") as doc:
        text = ""
        for page in doc:
            text += page.get_text()

    texts = text.split("\n")

    return True, text, text


async def scrape_html(url: str) -> Tuple[bool, str, str]:
    prune_filter = PruningContentFilter(threshold=0.4, threshold_type="dynamic", min_word_threshold=3)
    md_generator = DefaultMarkdownGenerator(content_filter=prune_filter, options={"ignore_links": False})
    browser_config = BrowserConfig(
        headless=True, verbose=False,
        extra_args=["--disable-gpu", "--disable-dev-shm-usage", "--no-sandbox", "--disable-extensions"]
    )
    crawler_config = CrawlerRunConfig(markdown_generator=md_generator, page_timeout=15000, verbose=False)

    async with AsyncWebCrawler(config=browser_config) as crawler:
        result = await asyncio.wait_for(crawler.arun(url=url, config=crawler_config), timeout=30)

    if not result.success:
        return False, f"Failed to scrape the page due to {result.error_message}", ""

    if len(result.markdown.raw_markdown.strip()) == 0:
        return False, f"Failed to scrape the page, returned empty content.", ""

    fit_markdown = result.markdown.fit_markdown
    raw_markdown = result.markdown.raw_markdown

    return True, fit_markdown, raw_markdown

MAX_RETRIES = 3
INITIAL_RETRY_DELAY = 0.2


@lru_cache(maxsize=8192)
@cache.memoize(typed=True, expire=1e7, tag="serper")
def serper_search(query: str, topk: int = 10) -> List[Dict]:
    url = "https://google.serper.dev/search"
    headers = {"X-API-KEY": os.environ["SERPER_API_KEY"], 'Content-Type': 'application/json'}
    payload = json.dumps({"q": query, "num": topk})

    for attempt in range(MAX_RETRIES):
        try:
            response = requests.post(url=url, headers=headers, data=payload)
            response.raise_for_status()
        except requests.exceptions.Timeout as e:
            if attempt < MAX_RETRIES - 1:
                delay = INITIAL_RETRY_DELAY * (attempt + 1)
                time.sleep(delay)
                continue
            else:
                raise e
        except Exception as e:
            raise e

        if response.status_code in [408, 500, 502, 503, 504]:
            if attempt < MAX_RETRIES - 1:
                delay = INITIAL_RETRY_DELAY * (attempt + 1)
                time.sleep(delay)
                continue
            else:
                raise Exception(response.text)

    response = response.json()
    return response


@lru_cache(maxsize=8192)
def _cached_search(query: str, topk: int = 10) -> str:
    """Cached search function that takes hashable parameters."""
    try:
        response = serper_search(query, topk=topk)
    except Exception as e:
        return f"Failed to search for query {query}.\nError: {str(e)}"

    keys = ["title", "link", "snippet"]
    template = "<Search Result {position}>\n<Title: {title}>\n<URL: {link}>\n{snippet}\n</Search Result {position}>"

    results = response["organic"]
    results = [r for r in results if all(k in r for k in keys) and all(not isinstance(r[k], str) or len(r[k]) > 0 for k in keys)][:10]
    results = [{**r, "position": i+1} for i, r in enumerate(results)]
    output = "\n\n".join([template.format(**r) for r in results])
    output = f"The search engine returned {len(results)} results:\n\n" + output
    return output


@lru_cache(maxsize=8192)
def _cached_search_o1(query: str, topk: int = 10) -> List[Dict]:
    # adapted from https://github.com/sunnynexus/Search-o1/blob/main/scripts/bing_search.py
    try:
        response = serper_search(query, topk=topk)
    except Exception as e:
        return f"Search error: {str(e)}"

    useful_info = []
    for i, result in enumerate(response['organic']):
        info = {
            'id': i + 1,  # Increment i for easier subsequent operations
            'title': result.get('title', ''),
            'url': result.get('link', ''),
            'site_name': result.get('source', ''),
            'date': result.get('date', ''),
            'snippet': result.get('snippet', ''),  # Remove HTML tags
            # Add context content to the information
            'context': ''  # Reserved field to be filled later
        }
        useful_info.append(info)

    return useful_info


@lru_cache(maxsize=8192)
@cache.memoize(typed=True, expire=1e7, tag="content")
def _cached_get_content(url: str, content_length: int = 10000) -> Tuple[bool, str, str]:
    """Cached function to get raw content from URL."""
    try:
        content_type = detect_content_type(url)
        if content_type == "pdf":
            result = asyncio.run(scrape_pdf(url))
        else:
            result = asyncio.run(scrape_html(url))
        return result
    except Exception as e:
        return False, str(e), ""


@lru_cache(maxsize=8192)
@cache.memoize(typed=True, expire=1e7, tag="snippet")
def _cached_find_snippet(content: str, query: str, num_characters: int = 10000, scoring_func: str = "rouge", chunking_func: str = "newline") -> str:
    """Cached function to find snippet in content."""
    if not query:
        return content[:num_characters]

    if chunking_func == "newline":
        content_lines = content.split("\n")
        content_lines = [line for line in content_lines if line.strip()]
    elif "words" in chunking_func:
        num_words = int(chunking_func.split("_")[1])
        content_lines = content.split(" ")
        content_lines = [line for line in content_lines if line.strip()]
        content_lines = [content_lines[i:i+num_words] for i in range(0, len(content_lines), num_words)]
        content_lines = [" ".join(line) for line in content_lines]
    else:
        raise ValueError(f"Invalid chunking function: {chunking_func}")

    if len(content_lines) == 0:
        return None

    return find_snippet(content_lines, query, num_characters, scoring_func)


def _cached_open_url(url: str, query: str, content_length: int = 10000, scoring_func: str = "rouge", chunking_func: str = "newline") -> str:
    """Main function that combines cached content retrieval and snippet finding."""
    # First get the raw content (cached by URL only)
    success, content_or_error, raw_content = _cached_get_content(url, content_length)
    if not success:
        return f"Failed to open the url {url}.\nAdditional information: {content_or_error}"

    final_content = _cached_find_snippet(content_or_error, query, content_length, scoring_func, chunking_func)
    if final_content is None:
        return f"Failed to open {url}"

    return f"Successfully opened the url {url}.\n<Content>\n{final_content}\n</Content>"


app = FastAPI()

@app.post("/search")
def search(request: SearchRequest):
    print(f"Search query: {request.query}")
    output = _cached_search(request.query, topk=request.topk)
    return {"output": output}


@app.post("/open_url")
def open_url(request: OpenUrlRequest):
    print(f"Open url: {request.url} with query: {request.query}")
    output = _cached_open_url(request.url, request.query, request.content_length, request.scoring_func, request.chunking_func)
    return {'output': output}


@app.post("/search_open_url")
def search_open_url(request: SearchRequest):
    print(f"Search query: {request.query}")
    try:
        search_results = serper_search(request.query, topk=request.topk)
    except Exception as e:
        return {"output": f"Search error: {str(e)}"}
    search_results = [r for r in search_results['organic'] if "link" in r]
    urls = [r['link'] for r in search_results]
    output = ""

    with ThreadPoolExecutor(max_workers=32) as executor:
        futures = {
            executor.submit(_cached_get_content, url, request.content_length): url
            for url in urls
        }
        for i, future in enumerate(tqdm(as_completed(futures), desc="Fetching URLs", total=len(urls))):
            url = futures[future]
            try:
                data = future.result()
                if not data[0]:
                    output += f"<URL {i}: {url}>\n<Error: {data[1]}>\n"
                else:
                    title = search_results[i].get('title', '') if i < len(search_results) else ''
                    output += f"<URL {i}: {url}>\n<Title: {title}>\n<Content>\n{data[1]}\n</Content>\n"
            except Exception as e:
                output += f"<URL {i}: {url}>\n<Error: {str(e)}>\n"

    return {"output": output}


@app.post("/search_o1")
def search_o1_search(request: SearchRequest):
    # returns all the formatted documents
    from search_o1_utils import fetch_page_content, extract_snippet_with_context
    print(f"Search query: {request.query}")
    output = _cached_search_o1(request.query, topk=request.topk)
    if isinstance(output, str):
        return {"output": output, "search_results": []}

    # after getting the output, search o1 always fetches all the content of the urls
    urls = [info['url'] for info in output]
    fetched_contents = fetch_page_content(urls, use_jina=False, jina_api_key=None)

    formatted_documents = ""
    for i, info in enumerate(output):
        url = info['url']
        raw_context = fetched_contents.get(url, '')
        info['snippet'] = info['snippet'].replace('<b>','').replace('</b>','')
        # default is 3000 chars in search o1
        success, filtered_context = extract_snippet_with_context(raw_context, info['snippet'], context_chars=3000)
        if success:
            context = filtered_context
        else:
            context = raw_context[:3000*2]

        info['context'] = context
        formatted_documents += f"**Web Page {i + 1}:**\n"
        formatted_documents += json.dumps(info, ensure_ascii=False, indent=2) + "\n"

    return {"output": formatted_documents, "search_results": output}


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--port", type=int, default=8006)
    args = parser.parse_args()
    assert os.environ["SERPER_API_KEY"] is not None, "SERPER_API_KEY is not set"

    uvicorn.run(app, host="0.0.0.0", port=args.port)
