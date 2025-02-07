import time
import requests
import re
import os
import sys

from  random import uniform
from collections import deque
from datetime import date
from bs4 import BeautifulSoup
from urllib.parse import urljoin, urlparse
from PyPDF2 import PdfReader

from utils import logger, create_directories_if_not_exist, save_to_file
from chroma_functions import save_to_chromadb

log = logger(__name__)

SAVE_TO = "chroma"  # either "chroma" or "locally"
WEBPAGE_DIRECTORY = "website_data"
EXTRACTED_PDF_DIRECTORY = "pdf_data"
PDF_DIRECTORY = "pdfs"
START_URL = "https://www.hdm-stuttgart.de"
ALLOWED_DOMAINS = {
    "hdm-stuttgart.de",
    "hdm-weiterbildung.de",
    "vs-hdm.de",
    "pmt.hdm-stuttgart.de",
    "omm.hdm-stuttgart.de",
}
DISALLOWED_PATHS = [
    "/studienfuehrer/vorlesungsverzeichnis/",
    "/studienfuehrer/Studiengaenge/",
    "/studienfuehrer/dozentenplaene/",
    "/studienfuehrer/raumbelegung/",
    "*/manage",
    "*/manage_main",
    "*/manage_workspace",
    "/pdm/pdm_deutsch/",
    "/pdm/pdm_englisch/",
    "/pdm/pdm_spanisch/",
    "*/html2pdf",
    "*/htmltopdf",
    "*printview=1",
    "/pmm/studiengang/team/mitarbeiter/lindig/",
    "/hochschule/neubau/webcams/tag*",
    "/ifak/startseite/redaktionzukunft/beitrag.html?beitrag_ID=1817&stars=2",
    "/*beitrag.html?beitrag_ID=1817",
    "*view_fotostrecke*",
    "*hdmnewsmail_simple*",
    "/vwif/",
    "splan.hdm-stuttgart.de"
]

def is_allowed(url: str) -> bool:
    """Check if a url is allowed based on the disallowed paths.

    Args:
        url (str): thue url to check

    Returns:
        bool: True if allowed, False if disallowed
    """
    for path in DISALLOWED_PATHS:
        if "*" in path:
            # Match wildcard patterns
            regex_path = path.replace("*", ".*")
            if re.search(regex_path, url):
                return False
        elif path in url:
            return False
    return True

def extract_domain_part(url: str) -> str:
    """Takes in an url and returns a string that is based on the urls domain, path and query.

    Args:
        url (str): The URL of the page being parsed.

    Returns:
        str: The url in a form thats usable as a path.
    """
    try:
        parsed_url = urlparse(url)
        # Extract the base domain (e.g., hdm-stuttgart from www.hdm-stuttgart.de)
        domain_match = re.search(r"(?:www\.)?(.*?)\.(de|com|org|net|pdf)", parsed_url.netloc)
        base_domain = domain_match.group(1) if domain_match else parsed_url.netloc
        # Extract the url path after .de
        path = parsed_url.path.strip("/").replace("/", "_")
        # Extract url query parameters after ?
        query = parsed_url.query.replace("&", "_").replace("=", "_") if parsed_url.query else ""

        # Combine components
        filename = f"{base_domain}"
        if path:
            filename += f"_{path}"
        if query:
            filename += f"_{query}"

        # Ensure filename is safe
        filename = re.sub(r'[<>:"/\\|?*]', "_", filename)
        return filename or "default"  # Fallback if filename is empty
    except Exception as e:
        log.error(f"Error generating filename from URL {url}: {e}")
        return "default"

def extract_relevant_content(soup: BeautifulSoup) -> str:
    """Extract relevant content from the main sections of the page.

    Args:
        soup (BeautifulSoup): The BeautifulSoup object of the page content.

    Returns:
        str: The cleaned and relevant content.
    """
    try:
        # List of potential tags to search for in priority order
        potential_main_tags = [
            {"name": "main", "attrs": {"id": "content_wrapper"}},
            {"name": "main", "attrs": {"id": "site-content"}},
            {"name": "div", "attrs": {"id": "main-body"}},
            {"name": "section", "attrs": {"id": "sp-main-body"}},
            {"name": "div", "attrs": {"id": "main"}},
            {"name": "div", "attrs": {"class": "sc-gsTCUz bhdLno"}},
        ]

        # Iterate through potential tags until one is found
        main_content = None
        for tag in potential_main_tags:
            main_content = soup.find(tag["name"], tag["attrs"])
            if main_content:
                break  # Stop searching once a valid tag is found

        if not main_content:
            log.warning("No relevant content section found.")
            return ""

        # Remove unwanted tags
        for tag in main_content.find_all(["nav", "aside", "script", "style"]):
            tag.decompose()  # Remove the tag and its content

        relevant_text = main_content.get_text(separator=" \n ", strip=True)
        # relevant_text = main_content.get_text(separator=" ", strip=True)

        return relevant_text
    except Exception as e:
        log.error(f"Error extracting relevant content: {e}")
        return ""

def process_webpage_content(soup: BeautifulSoup, url: str) -> None:
    """Saves specific parts of a webpage into a file.

    Args:
        soup (BeautifulSoup): The BeautifulSoup object of the pag content.
        url (str): The URL of the page being parsed.
    """
    try:
        relevant_content = extract_relevant_content(soup)
        if relevant_content == "":
            log.warning("No relevant content found.")
            return
        relevant_content = relevant_content.lower()

        title = soup.title.get_text(strip=True) if soup.title else "No Title"

        sanitized_url = extract_domain_part(url)
        filename = os.path.join(WEBPAGE_DIRECTORY, sanitized_url + ".json")

        if SAVE_TO == "chroma":
            save_to_chromadb(url, title, relevant_content, "webpage", log)
        elif SAVE_TO == "locally":
            save_to_file(url=url, title=title, content=relevant_content, file_path=filename, log=log)

    except Exception as e:
        log.error(f"Failed to save content from {url}: {e}")

def process_pdf_content(response, url: str) -> None:
    """Process and save PDF content directly from the response.

    Args:
        response: The HTTP response containing the PDF content.
        url (str): The URL of the PDF.
    """
    try:
        # Save the PDF locally
        sanitized_url = extract_domain_part(url)
        if not sanitized_url.endswith(".pdf"):
            sanitized_url += ".pdf"
        pdf_filename = os.path.join(PDF_DIRECTORY, sanitized_url)
        json_filename = os.path.join(EXTRACTED_PDF_DIRECTORY, sanitized_url.replace(".pdf", ".json"))

        with open(pdf_filename, "wb") as file:
            file.write(response.content)

        # Extract text from the PDF
        reader = PdfReader(pdf_filename)
        pdf_text = ""
        for page in reader.pages:
            pdf_text += page.extract_text()
        pdf_text = pdf_text.replace("\n", " \n ")
        pdf_text = pdf_text.lower()

        # Extract title from metadata, fallback to filename
        metadata_title = reader.metadata.get("/Title", None) if reader.metadata else None
        if not isinstance(metadata_title, str):
            metadata_title = ""
        parsed_url = urlparse(url)
        filename_title = os.path.basename(parsed_url.path).replace(".pdf", "")
        title = metadata_title or filename_title or "Untitled PDF"

        if SAVE_TO == "chroma":
            save_to_chromadb(url, title, pdf_text, "pdf", log)
        elif SAVE_TO == "locally":
            save_to_file(url=url, title=title, content=pdf_text, file_path=json_filename, log=log)

        log.info(f"Saved PDF and extracted content: {pdf_filename}, {json_filename}")
    except Exception as e:
        log.error(f"Failed to process PDF {url}: {e}")

def extract_links(soup: BeautifulSoup, url: str, visited: set[str], to_visit: set[str]) -> set[str]:
    """Extracts all valid links from a webpage.

    Args:
        soup (BeautifulSoup): The BeautifulSoup object of the pag content.
        url (str): The URL of the page being parsed.

    Returns:
        Set[str]: A set of valid links extracted from the page.
    """
    try:
        # Extract all <a> tags with href attributes
        filtered_links = set()
        for a in soup.find_all("a", href=True):
            href = a["href"].strip()
            full_url = urljoin(url, href)
            
            if (
                "#" in full_url
                or "hdm" not in full_url
                or full_url.startswith("mailto:")
                or not full_url.lower().endswith((".pdf", ".html", ".htm", ""))
                or not full_url.startswith(("http://", "https://"))
            ):
                continue

            parsed_url = urlparse(full_url)
            domain = parsed_url.netloc.lower()
            if not any(domain.endswith(allowed) for allowed in ALLOWED_DOMAINS):
                continue

            if full_url not in visited and full_url not in to_visit:
                filtered_links.add(full_url)

        return filtered_links

    except requests.RequestException as e:
        log.error(f"Failed to fetch links from {url}: {e}")
        return set()

def save_crawl_state(to_visit, visited, filename_to_visit: str = "to_visit.json", filename_visited: str = "visited.json") -> None:
    """Saves the `to_visit` deque or set and `visited` set into separate JSON files.

    Args:
        to_visit (deque or set): The collection of URLs still to visit.
        visited (set): The collection of URLs already visited.
        filename_to_visit (str): The file where the URLs to visit will be stored.
        filename_visited (str): The file where the visited URLs will be stored.
    """
    try:

        if isinstance(to_visit, deque):
            to_visit = list(to_visit)
        to_visit_data = {"to_visit": to_visit}
        visited_data = {"visited": list(visited)}
        
        with open(filename_to_visit, "w", encoding="utf-8") as file:
            json.dump(to_visit_data, file, indent=4, ensure_ascii=False)
        
        with open(filename_visited, "w", encoding="utf-8") as file:
            json.dump(visited_data, file, indent=4, ensure_ascii=False)
        

        log.info(f"Saved {len(to_visit)} URLs to {filename_to_visit}.")
        log.info(f"Saved {len(visited)} URLs to {filename_visited}.")
    except Exception as e:
        log.error(f"Error saving to_visit to {filename_to_visit}: {e}")
        log.error(f"Error saving visited to {filename_visited}: {e}")

def load_crawl_state(filename_to_visit: str = "to_visit.json", filename_visited: str = "visited.json"):
    """Loads the saved state of `to_visit` deque and `visited` set from JSON files if they exist.
    
    Args:
        filename_to_visit (str): The file where the URLs to visit are stored.
        filename_visited (str): The file where the visited URLs are stored.
    
    Returns:
        tuple: A deque of URLs to visit and a set of visited URLs.
    """
    to_visit = deque()
    visited = set()
    
    if os.path.exists(filename_to_visit):
        try:
            with open(filename_to_visit, "r", encoding="utf-8") as file:
                data = json.load(file)
                to_visit = deque(data.get("to_visit", []))
                logging.info(f"Loaded {len(to_visit)} URLs from {filename_to_visit}.")
        except Exception as e:
            logging.error(f"Error loading {filename_to_visit}: {e}")
    
    if os.path.exists(filename_visited):
        try:
            with open(filename_visited, "r", encoding="utf-8") as file:
                data = json.load(file)
                visited = set(data.get("visited", []))
                logging.info(f"Loaded {len(visited)} URLs from {filename_visited}.")
        except Exception as e:
            logging.error(f"Error loading {filename_visited}: {e}")
    
    return to_visit, visited

def crawl_website(start_url: str, continue_from_last=False, page_depth: int = None) -> None:
    """Crawl a website starting from a given URL, timing each iteration.

    Args:
        start_url (str): The URL to begin crawling from.
    """
    session = requests.Session()
    i = 0

    create_directories_if_not_exist(WEBPAGE_DIRECTORY, EXTRACTED_PDF_DIRECTORY, PDF_DIRECTORY)

    if continue_from_last:
        to_visit, visited = load_crawl_state()
        if not to_visit:
            to_visit = deque([start_url])
    else:
        visited = set()
        to_visit = deque([start_url])
    
    to_visit_set = set(to_visit)
    try:
        while to_visit:
            iteration_start_time = time.time()  # Start timing the iteration
            current_url = to_visit.popleft()
            to_visit_set.remove(current_url)

            if current_url in visited:
                continue
            
            if not is_allowed(current_url):
                visited.add(current_url)
                log.warning(f"Skipping disallowed URL: {current_url}")
                continue
            
            visited.add(current_url)
            
            log.info(f"Links visited: {len(visited)}, Links left to visit: {len(to_visit)}")
            log.info(f"Visiting: {current_url}")
            try:
                response = session.get(current_url, timeout=10)
                response.raise_for_status()

                content_type = response.headers.get("Content-Type", "").lower()
                log.info(f"Content type: {content_type}")
                
                # Process the content based on the content type
                if "text/html" in content_type:
                    log.info(f"Detected HTML: {current_url}")
                    soup = BeautifulSoup(response.content, "html.parser")
                    process_webpage_content(soup, current_url)

                    # extracts all links from webpage, and adds them to the list/set if they are not 
                    # already visited or in the to_visit set
                    new_links = extract_links(
                        soup=soup, url=current_url, visited=visited, to_visit=to_visit_set
                    )
                    for link in new_links:
                        to_visit.append(link)
                        to_visit_set.add(link)

                elif "application/pdf" in content_type:
                    log.info(f"Detected PDF: {current_url}")
                    process_pdf_content(response, current_url)

                else:
                    log.error(
                        f"Current URL {current_url} is of unsupported content type {content_type}."
                    )
                    continue

            except Exception as e:
                log.error(f"Failed to process URL {current_url}: {e}")

            if page_depth is not None:
                i += 1
                if i >= page_depth:
                    log.info("Reached maximum page depth, stopping.")
                    break

            iteration_end_time = time.time()
            elapsed_time = iteration_end_time - iteration_start_time
            log.info(f"Iteration {i} completed in {elapsed_time:.4f} seconds")

            delay = uniform(0.5, 2)
            log.info(f"Sleeping for {delay:.2f} seconds")
            time.sleep(delay)
    except KeyboardInterrupt:
        log.warning("KeyboardInterrupt detected! Saving crawl state before exiting...")
        save_crawl_state(to_visit, visited)
        log.info("Crawl state saved. Exiting gracefully.")
        sys.exit(0)

    except Exception as e:
        log.error(f"Unexpected error: {e}. Saving crawl state...")
        save_crawl_state(to_visit, visited)
        log.info("Crawl state saved. Exiting gracefully.")
        sys.exit(1)

    finally:
        save_crawl_state(to_visit, visited)
        log.info("Crawl completed successfully.")

crawl_website(START_URL, continue_from_last=False, page_depth=100)
