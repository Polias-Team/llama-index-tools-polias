"""Polias tool spec."""

import os
from typing import Any, Dict, List, Optional

import requests
from llama_index.core.schema import Document
from llama_index.core.tools.tool_spec.base import BaseToolSpec

DEFAULT_BASE_URL = "https://api.polias.eu/v1"


class PoliasToolSpec(BaseToolSpec):
    """
    Polias tool spec.

    Search an EU-hosted index for web pages, news, official sources and law.
    Get an API key at https://api.polias.eu/account.
    """

    spec_functions = ["search", "ai_search", "legal_search"]

    def __init__(
        self,
        api_key: Optional[str] = None,
        lang: str = "de",
        country: Optional[str] = None,
        base_url: str = DEFAULT_BASE_URL,
        timeout: float = 20.0,
    ) -> None:
        """
        Initialize with parameters.

        Args:
            api_key: Polias API key (starts with "pol_"). Falls back to the
                environment variable POLIAS_API_KEY.
            lang: Index language, "de" (default) or "en" (focus on technical topics).
            country: Prefer results from a country, e.g. "de", "at" or "ch".
            base_url: API base URL.
            timeout: Request timeout in seconds.

        """
        self.api_key = api_key or os.environ.get("POLIAS_API_KEY")
        if not self.api_key:
            raise ValueError(
                "A Polias API key is required: pass api_key or set POLIAS_API_KEY."
            )
        self.lang = lang
        self.country = country
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout

    def _get(self, path: str, params: Dict[str, Any]) -> Dict[str, Any]:
        params = {k: v for k, v in params.items() if v not in (None, "", [])}
        response = requests.get(
            f"{self.base_url}{path}",
            params=params,
            headers={
                "Authorization": f"Bearer {self.api_key}",
                "Accept": "application/json",
            },
            timeout=self.timeout,
        )
        if response.status_code >= 400:
            try:
                message = response.json().get("error", response.text)
            except ValueError:
                message = response.text
            raise ValueError(f"Polias API error {response.status_code}: {message}")
        return response.json()

    def search(
        self,
        query: str,
        areas: Optional[List[str]] = None,
        count: int = 10,
        offset: int = 0,
        town: Optional[str] = None,
    ) -> List[Document]:
        """
        Search the web, news, law or official sources.

        Args:
            query: The search query (up to 200 characters). Case numbers and
                ECLI work too.
            areas: Optional list of areas: "web", "news" (last 30 days),
                "legal" (court decisions and legislation) and "official"
                (websites of public authorities; not together with "web").
                Without areas, a web search is run.
            count: Number of results, 1 to 10. With "web" and other areas,
                the results are shared: each other area gets count divided by
                the number of areas (rounded down), web gets the rest. For
                example count=5 with ["web", "legal"] gives 3 web and 2 legal
                results.
            offset: Start at this result, 0 to 90 (page 2 = 10).
            town: Optional municipality in Germany, Austria or Switzerland to
                focus local results, e.g. "Köln".

        Returns:
            A list of documents with the snippet as text and title, url, area
            and, for news, the publication time as metadata.

        """
        areas = [_AREA_PARAM.get(a, a) for a in (areas or [])]
        data = self._get(
            "/search",
            {
                "q": query,
                "result_filter": ",".join(areas),
                "count": count,
                "offset": offset,
                "lang": self.lang,
                "land": self.country,
                "ort": town,
                **_area_counts(areas, count),
            },
        )
        documents: List[Document] = []
        if "results" in data:
            blocks = {"web": data["results"]}
        else:
            blocks = {
                area: data.get(area, {}).get("results", [])
                for area in data.get("result_filter", [])
            }
        for area, results in blocks.items():
            for r in results:
                documents.append(
                    Document(
                        text=r.get("snippet", ""),
                        metadata=_clean(
                            {
                                "title": r.get("title"),
                                "url": r.get("url"),
                                "area": _AREA_NAME.get(area, area),
                                "published": r.get("page_age"),
                                "notice": r.get("hinweis"),
                            }
                        ),
                    )
                )
        return documents

    def ai_search(
        self,
        query: str,
        areas: Optional[List[str]] = None,
        count: int = 5,
    ) -> List[Document]:
        """
        Search for use in AI answers: each result carries licence details and,
        where the licence allows, a longer excerpt or the full legal text.

        Args:
            query: The search query (up to 200 characters).
            areas: Optional list of areas: "web", "news" and "legal". Without
                areas, web and news are searched.
            count: Number of results per area, 1 to 10.

        Returns:
            A list of documents with the excerpt (or full text) as text and
            title, url, area, licence and source notice as metadata.

        """
        data = self._get(
            "/ki",
            {
                "q": query,
                "scope": ",".join(areas or []),
                "count": count,
                "lang": self.lang,
                "land": self.country,
            },
        )
        documents: List[Document] = []
        for area in ("web", "news", "official", "legal"):
            for r in data.get(area) or []:
                text = r.get("volltext") or r.get("excerpt") or r.get("snippet") or ""
                documents.append(
                    Document(
                        text=text,
                        metadata=_clean(
                            {
                                "title": r.get("title") or r.get("titel"),
                                "url": r.get("url"),
                                "area": area,
                                "as_of": r.get("as_of") or r.get("date"),
                                "license": r.get("license") or r.get("lizenz"),
                                "ai_training": r.get("ai_training"),
                                "notice": r.get("notice") or r.get("hinweis"),
                                "court": r.get("court"),
                                "case_number": r.get("case_number"),
                                "ecli": r.get("ecli"),
                                "full_text": bool(r.get("volltext")),
                            }
                        ),
                    )
                )
        return documents

    def legal_search(
        self,
        query: str,
        country: Optional[str] = None,
        court: Optional[str] = None,
        count: int = 10,
    ) -> List[Document]:
        """
        Search court decisions and legislation from the EU and European countries.

        Args:
            query: The search query, e.g. a topic, case number, ECLI or citation
                such as "§ 536 BGB".
            country: Optional country codes, comma-separated: "de", "at", "ch",
                "eu", "fr", "nl", "es".
            court: Optional court, e.g. "BGH", "CJEU" or an ECLI court code.
            count: Number of results, 1 to 10.

        Returns:
            A list of documents with an excerpt as text and title, url, court,
            case number, date, ECLI, licence and source notice as metadata.
            Please show the source notice with the result.

        """
        data = self._get(
            "/legal",
            {"q": query, "country": country, "court": court, "count": count},
        )
        return [
            Document(
                text=r.get("excerpt") or r.get("ausschnitt") or "",
                metadata=_clean(
                    {
                        "title": r.get("title"),
                        "url": r.get("url"),
                        "court": r.get("court"),
                        "case_number": r.get("case_number"),
                        "date": r.get("date"),
                        "ecli": r.get("ecli"),
                        "country": r.get("country"),
                        "license": r.get("license"),
                        "notice": r.get("notice"),
                    }
                ),
            )
            for r in data.get("results", [])
        ]


_AREA_PARAM = {"legal": "recht", "official": "amtlich"}
_AREA_NAME = {"recht": "legal", "amtlich": "official"}
_COUNT_PARAM = {"recht": "recht_count", "news": "news_count"}


def _area_counts(areas: List[str], count: int) -> Dict[str, int]:
    """
    Share the results between web and the other areas.

    Without explicit shares, the API adds news and legal results to a web
    search only when they match the query well. Each other area gets
    count // len(areas) results, web gets the rest.
    """
    others = [a for a in areas if a in _COUNT_PARAM]
    if "web" not in areas or not others:
        return {}
    share = count // len(areas)
    if share < 1:
        return {}
    return {_COUNT_PARAM[a]: share for a in others}


def _clean(metadata: Dict[str, Any]) -> Dict[str, Any]:
    return {k: v for k, v in metadata.items() if v not in (None, "")}
