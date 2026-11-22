from unittest.mock import Mock, patch

import pytest
from llama_index.core.schema import Document
from llama_index.core.tools.tool_spec.base import BaseToolSpec
from llama_index.tools.polias import PoliasToolSpec


def _response(payload, status=200):
    r = Mock()
    r.status_code = status
    r.json.return_value = payload
    r.text = str(payload)
    return r


def test_class():
    names_of_base_classes = [b.__name__ for b in PoliasToolSpec.__mro__]
    assert BaseToolSpec.__name__ in names_of_base_classes


def test_spec_functions():
    assert PoliasToolSpec.spec_functions == ["search", "ai_search", "legal_search"]
    tools = PoliasToolSpec(api_key="pol_test").to_tool_list()
    assert [t.metadata.name for t in tools] == ["search", "ai_search", "legal_search"]


def test_requires_key(monkeypatch):
    monkeypatch.delenv("POLIAS_API_KEY", raising=False)
    with pytest.raises(ValueError):
        PoliasToolSpec()


def test_key_from_env(monkeypatch):
    monkeypatch.setenv("POLIAS_API_KEY", "pol_env")
    assert PoliasToolSpec().api_key == "pol_env"


@patch("llama_index.tools.polias.base.requests.get")
def test_search_web(mock_get):
    mock_get.return_value = _response(
        {"results": [{"title": "T", "url": "https://a.example", "snippet": "S"}]}
    )
    docs = PoliasToolSpec(api_key="pol_test", country="at").search("wetter", count=3)
    assert isinstance(docs[0], Document)
    assert docs[0].text == "S"
    assert docs[0].metadata == {"title": "T", "url": "https://a.example", "area": "web"}
    args, kwargs = mock_get.call_args
    assert args[0] == "https://api.polias.eu/v1/search"
    assert kwargs["params"] == {"q": "wetter", "count": 3, "offset": 0, "lang": "de", "land": "at"}
    assert kwargs["headers"]["Authorization"] == "Bearer pol_test"


@patch("llama_index.tools.polias.base.requests.get")
def test_search_areas(mock_get):
    mock_get.return_value = _response(
        {
            "result_filter": ["news", "legal"],
            "news": {"results": [{"title": "N", "url": "https://n.example", "snippet": "n", "page_age": "2026-11-20T09:00:00"}]},
            "legal": {"results": [{"title": "L", "url": "https://l.example", "snippet": "l", "hinweis": "Quelle: X"}]},
        }
    )
    docs = PoliasToolSpec(api_key="pol_test").search("q", areas=["news", "legal"])
    assert mock_get.call_args.kwargs["params"]["result_filter"] == "news,recht"
    assert [d.metadata["area"] for d in docs] == ["news", "legal"]
    assert docs[0].metadata["published"] == "2026-11-20T09:00:00"
    assert docs[1].metadata["notice"] == "Quelle: X"


@patch("llama_index.tools.polias.base.requests.get")
def test_ai_search_full_text(mock_get):
    mock_get.return_value = _response(
        {
            "web": [{"title": "W", "url": "https://w.example", "snippet": "s", "license": "unknown"}],
            "legal": [{"titel": "Urteil", "url": "https://r.example", "volltext": "Volltext", "lizenz": "CC0", "ecli": "ECLI:X"}],
        }
    )
    docs = PoliasToolSpec(api_key="pol_test").ai_search("q", areas=["web", "legal"], count=2)
    assert mock_get.call_args.kwargs["params"]["scope"] == "web,legal"
    assert docs[0].text == "s" and docs[0].metadata["full_text"] is False
    assert docs[1].text == "Volltext"
    assert docs[1].metadata["license"] == "CC0" and docs[1].metadata["full_text"] is True


@patch("llama_index.tools.polias.base.requests.get")
def test_legal_search(mock_get):
    mock_get.return_value = _response(
        {"results": [{"title": "BGH", "url": "https://b.example", "excerpt": "e", "court": "BGH", "case_number": "VIII ZR 1/23", "notice": "Quelle"}]}
    )
    docs = PoliasToolSpec(api_key="pol_test").legal_search("Mietminderung", country="de", court="BGH")
    assert mock_get.call_args.args[0].endswith("/legal")
    assert docs[0].metadata["case_number"] == "VIII ZR 1/23"


@patch("llama_index.tools.polias.base.requests.get")
def test_error(mock_get):
    mock_get.return_value = _response({"error": "API key missing or unknown"}, status=401)
    with pytest.raises(ValueError, match="401"):
        PoliasToolSpec(api_key="pol_bad").search("q")


@pytest.mark.parametrize(
    ("areas", "count", "expected"),
    [
        (["web", "legal"], 5, {"recht_count": 2}),
        (["web", "news"], 10, {"news_count": 5}),
        (["web", "news", "legal"], 5, {"news_count": 1, "recht_count": 1}),
        (["legal", "web"], 4, {"recht_count": 2}),
        (["web", "legal"], 1, {}),
        (["news", "legal"], 5, {}),
        (["web"], 5, {}),
        (None, 5, {}),
    ],
)
@patch("llama_index.tools.polias.base.requests.get")
def test_search_shares_results_with_web(mock_get, areas, count, expected):
    mock_get.return_value = _response({"result_filter": [], "web": {"results": []}})
    PoliasToolSpec(api_key="pol_test").search("q", areas=areas, count=count)
    params = mock_get.call_args.kwargs["params"]
    shares = {k: v for k, v in params.items() if k in ("recht_count", "news_count")}
    assert shares == expected


@patch("llama_index.tools.polias.base.requests.get")
def test_search_web_and_legal_returns_both(mock_get):
    mock_get.return_value = _response(
        {
            "result_filter": ["web", "legal"],
            "web": {"results": [{"title": "W", "url": "https://w.example", "snippet": "w"}] * 3},
            "recht": {"results": [{"title": "L", "url": "https://l.example", "snippet": "l"}] * 2},
            "legal": {"results": [{"title": "L", "url": "https://l.example", "snippet": "l"}] * 2},
        }
    )
    docs = PoliasToolSpec(api_key="pol_test").search("q", areas=["web", "legal"], count=5)
    params = mock_get.call_args.kwargs["params"]
    assert params["result_filter"] == "web,recht"
    assert params["recht_count"] == 2
    assert [d.metadata["area"] for d in docs] == ["web"] * 3 + ["legal"] * 2
