# LlamaIndex Tools Integration: Polias

This tool connects to [Polias](https://api.polias.eu/), a search engine with its own index, hosted in the EU.
Agents can use it to search the web, news, official sources and law.

## Installation

```bash
pip install llama-index-tools-polias
```

You need a Polias API key. Create an account at [api.polias.eu/account](https://api.polias.eu/account), as a company or as a
private person (private accounts: residents of Germany, Austria or Switzerland, 18 or older, non-commercial use only).
Each month, 1,000 searches are free for companies and 100 for private persons. More is available as packages, see
[pricing](https://api.polias.eu/pricing).

```bash
export POLIAS_API_KEY="pol_..."
```

## Usage

```python
from llama_index.tools.polias import PoliasToolSpec

polias = PoliasToolSpec()  # reads the key from POLIAS_API_KEY

docs = polias.search("Kündigung Eigenbedarf", areas=["web", "legal"], count=5)
for doc in docs:
    print(doc.metadata["area"], doc.metadata["title"], doc.metadata["url"])
```

With an agent:

```python
from llama_index.core.agent.workflow import FunctionAgent

llm = ...  # any LlamaIndex LLM with function calling

agent = FunctionAgent(tools=PoliasToolSpec().to_tool_list(), llm=llm)
print(await agent.run("Which court decisions deal with rent reduction because of mould?"))
```

## Tools

- `search(query, areas=None, count=10, offset=0, town=None)`: results with title, URL and a short snippet. The areas
  are `web`, `news` (last 30 days), `legal` (court decisions and legislation) and `official` (websites of public
  authorities in Germany, Austria, Switzerland and the EU). `town` focuses local results, e.g. `"Köln"`. With `web`
  and other areas, the results are shared: each other area gets `count` divided by the number of areas (rounded down),
  web gets the rest. `areas=["web", "legal"], count=5` gives 3 web and 2 legal results.
- `ai_search(query, areas=None, count=5)`: results for AI answers. Each result carries licence details and, where the
  licence allows, a longer excerpt. For `legal` results, the full text is included where the licence allows it.
- `legal_search(query, country=None, court=None, count=10)`: court decisions and legislation from the EU and European
  countries, with court, case number, date, ECLI and source. Please show the source notice (`notice`) with the result.

Options for the whole tool: `lang="de"` (default) or `"en"` (a focus on technical topics such as developer
documentation, software and IT), and `country="de"`, `"at"` or `"ch"` to prefer results from a country.

Documentation: [api.polias.eu](https://api.polias.eu/) · Questions: hallo@polias.eu
