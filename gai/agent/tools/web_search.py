import urllib.parse
import urllib.request
import re

HAS_DUCKDUCKGO = False
for mod_name in ("ddgs", "duckduckgo_search"):
    try:
        mod = __import__(mod_name, fromlist=["DDGS"])
        DDGS = mod.DDGS
        HAS_DUCKDUCKGO = True
        break
    except ImportError:
        continue

HAS_BS4 = False
try:
    from bs4 import BeautifulSoup
    HAS_BS4 = True
except ImportError:
    pass

HAS_REQUESTS = False
try:
    import requests
    HAS_REQUESTS = True
except ImportError:
    pass


class WebSearch:
    def __init__(self):
        self.name = "web_search"
        self.description = "Search the internet or fetch webpage content"
        self._session = None

    @property
    def session(self):
        if self._session is None and HAS_REQUESTS:
            self._session = requests.Session()
            self._session.headers.update({
                "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"
            })
        return self._session

    def search(self, query, max_results=5):
        if HAS_DUCKDUCKGO:
            try:
                with DDGS() as ddgs:
                    results = list(ddgs.text(query, max_results=max_results))
                if not results:
                    return "No results found."
                return "\n\n".join(
                    f"{i+1}. {r['title']}\n   URL: {r['href']}\n   {r.get('body', '')[:500]}"
                    for i, r in enumerate(results)
                )
            except Exception as e:
                return f"Search error: {e}"

        encoded = urllib.parse.quote(query)
        if self.session:
            try:
                resp = self.session.get(
                    f"https://html.duckduckgo.com/html/?q={encoded}",
                    timeout=10,
                )
                html = resp.text
            except Exception as e:
                return f"Search error: {e}"
        else:
            req = urllib.request.Request(
                f"https://html.duckduckgo.com/html/?q={encoded}",
                headers={"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"}
            )
            try:
                with urllib.request.urlopen(req, timeout=10) as resp:
                    html = resp.read().decode("utf-8", errors="replace")
            except Exception as e:
                return f"Search error: {e}"

        results = []
        if HAS_BS4:
            soup = BeautifulSoup(html, "html.parser")
            for result in soup.select(".result"):
                link = result.select_one(".result__a")
                snippet = result.select_one(".result__snippet")
                if link:
                    url = link.get("href", "")
                    title = link.get_text(strip=True)
                    snippet_text = snippet.get_text(strip=True) if snippet else ""
                    results.append({"title": title, "url": url, "snippet": snippet_text})
                    if len(results) >= max_results:
                        break
        else:
            for m in re.finditer(
                r'class="result__a"[^>]*href="(.*?)".*?>(.*?)</a>.*?class="result__snippet">(.*?)(?:</(?:a|div)>)',
                html, re.DOTALL
            ):
                url = m.group(1)
                title = re.sub(r"<[^>]+>", "", m.group(2)).strip()
                snippet = re.sub(r"<[^>]+>", "", m.group(3)).strip()
                results.append({"title": title, "url": url, "snippet": snippet})
                if len(results) >= max_results:
                    break

        if not results:
            return "No results found."
        return "\n\n".join(
            f"{i+1}. {r['title']}\n   URL: {r['url']}\n   {r['snippet']}"
            for i, r in enumerate(results)
        )

    def fetch_page(self, url):
        if self.session:
            try:
                resp = self.session.get(url, timeout=10)
                html = resp.text
            except Exception as e:
                return f"Fetch error: {e}"
        else:
            req = urllib.request.Request(url, headers={
                "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"
            })
            try:
                with urllib.request.urlopen(req, timeout=10) as resp:
                    html = resp.read().decode("utf-8", errors="replace")
            except Exception as e:
                return f"Fetch error: {e}"

        if HAS_BS4:
            soup = BeautifulSoup(html, "html.parser")
            for tag in soup(["script", "style", "nav", "footer", "header"]):
                tag.decompose()
            text = soup.get_text(separator=" ", strip=True)
        else:
            from html.parser import HTMLParser
            class Parser(HTMLParser):
                def __init__(self):
                    super().__init__()
                    self.text = []
                def handle_data(self, d):
                    self.text.append(d)
            p = Parser()
            p.feed(html)
            text = " ".join(p.text)

        text = re.sub(r"\s+", " ", text).strip()
        return text[:8000] + ("..." if len(text) > 8000 else "")

    def run(self, action, **kwargs):
        if action == "search":
            return self.search(kwargs.get("query", ""), kwargs.get("max_results", 5))
        elif action == "fetch":
            return self.fetch_page(kwargs.get("url", ""))
        return f"Unknown action: {action}. Available: search, fetch"
