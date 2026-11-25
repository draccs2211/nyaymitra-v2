from fastmcp import FastMCP
from backend.config import INDIAN_KANOON_API_KEY, INDIAN_KANOON_BASE_URL, KANOON_CACHE_TTL
from backend.utils.cache import cache_get_json, cache_set_json, make_key
import httpx

mcp = FastMCP("IndianKanoon MCP Server")


@mcp.tool()
async def search_case_laws(
    query: str,
    court: str = None,
    year_from: int = None,
    year_to: int = None,
    max_results: int = 5
) -> dict:
    """
    Search Indian case laws from Indian Kanoon.
    Returns discovery results only — verify from official sources before citing in court.

    Args:
        query: Legal query or keywords
        court: Optional Indian Kanoon doctypes filter, comma-separated
               (e.g. "supremecourt", "allahabad", "highcourts")
        year_from: Optional start year filter
        year_to: Optional end year filter
        max_results: Number of results to return (max 10)
    """
    cache_key = make_key("kanoon_search", query, court, year_from, year_to, max_results)

    cached = cache_get_json(cache_key)
    if cached:
        return cached

    if not INDIAN_KANOON_API_KEY:
        return {
            "status": "error",
            "code": "not_configured",
            "message": "Indian Kanoon API key not configured",
            "results": [],
            "disclaimer": "Configure INDIAN_KANOON_API_KEY in .env file"
        }

    try:
        params = {"formInput": query, "pagenum": 0}
        if court:
            params["formInput"] += f" doctypes:{court}"      # documented filter operator
        if year_from:
            params["formInput"] += f" fromdate:01-01-{year_from}"
        if year_to:
            params["formInput"] += f" todate:31-12-{year_to}"

        headers = {"Authorization": f"Token {INDIAN_KANOON_API_KEY}", "Accept": "application/json"}

        # API docs show POST with the query in the URL; billed per result page returned
        async with httpx.AsyncClient(timeout=15.0) as client:
            response = await client.post(
                f"{INDIAN_KANOON_BASE_URL}/search/",
                params=params,
                headers=headers
            )

            if response.status_code == 200:
                data = response.json()
                docs = data.get("docs", [])[:max_results]

                results = []
                for doc in docs:
                    results.append({
                        "title": doc.get("title", "Unknown"),
                        "citation": doc.get("citation", ""),
                        "court": doc.get("docsource", ""),
                        "date": doc.get("publishdate", ""),
                        "snippet": doc.get("headline", ""),
                        "url": f"https://indiankanoon.org/doc/{doc.get('tid', '')}",
                        "tid": doc.get("tid", ""),
                        "author": doc.get("author", "")
                    })

                output = {
                    "status": "success",
                    "query": query,
                    "total_found": data.get("total", 0),
                    "results": results,
                    "disclaimer": "⚠️ Indian Kanoon is for discovery only. Verify from official court websites before citing in court."
                }

                cache_set_json(cache_key, output, KANOON_CACHE_TTL)
                return output

            return {
                "status": "error",
                "code": "upstream_error",
                "message": f"API returned {response.status_code}",
                "results": []
            }

    except Exception as e:
        return {
            "status": "error",
            "code": "upstream_error",
            "message": str(e),
            "results": [],
            "disclaimer": "Failed to fetch from Indian Kanoon. Please try Indian Kanoon website directly."
        }


@mcp.tool()
async def get_case_details(tid: str) -> dict:
    """
    Get full details of a specific case from Indian Kanoon.

    Args:
        tid: The document ID from Indian Kanoon
    """
    cache_key = make_key("kanoon_case", tid)

    cached = cache_get_json(cache_key)
    if cached:
        return cached

    if not INDIAN_KANOON_API_KEY:
        return {"status": "error", "code": "not_configured", "message": "API key not configured"}

    try:
        headers = {"Authorization": f"Token {INDIAN_KANOON_API_KEY}", "Accept": "application/json"}
        async with httpx.AsyncClient(timeout=15.0) as client:
            response = await client.post(
                f"{INDIAN_KANOON_BASE_URL}/doc/{tid}/",
                headers=headers
            )
            if response.status_code == 200:
                data = response.json()
                output = {
                    "status": "success",
                    "tid": tid,
                    "title": data.get("title", ""),
                    "court": data.get("docsource", ""),
                    "date": data.get("publishdate", ""),
                    "citation": data.get("citation", ""),
                    "doc": data.get("doc", "")[:3000],
                    "url": f"https://indiankanoon.org/doc/{tid}/",
                    "disclaimer": "⚠️ Verify from official source before citing in court."
                }
                cache_set_json(cache_key, output, KANOON_CACHE_TTL * 7)
                return output

            return {"status": "error", "message": f"Case not found: {tid}"}

    except Exception as e:
        return {"status": "error", "message": str(e)}


@mcp.tool()
async def check_case_validity(citation: str) -> dict:
    """
    Check if a case citation is valid and not overruled.

    Args:
        citation: Case citation string e.g. "AIR 2020 SC 123"
    """
    cache_key = make_key("kanoon_validity", citation)
    cached = cache_get_json(cache_key)
    if cached:
        return cached

    try:
        search_result = await search_case_laws(
            query=citation,
            max_results=3
        )

        if search_result.get("results"):
            output = {
                "status": "found",
                "citation": citation,
                "cases": search_result["results"],
                "validity": "Please verify on official court website",
                "official_sources": {
                    "Supreme Court": "main.sci.gov.in",
                    "Allahabad HC": "allahabadhighcourt.in",
                    "eCourts": "ecourts.gov.in"
                },
                "disclaimer": "⚠️ Indian Kanoon cannot confirm if a case is overruled. Always verify from official sources."
            }
        else:
            output = {
                "status": "not_found",
                "citation": citation,
                "message": "Citation not found in Indian Kanoon database",
                "disclaimer": "Please verify directly on official court websites."
            }

        cache_set_json(cache_key, output, KANOON_CACHE_TTL)
        return output

    except Exception as e:
        return {"status": "error", "message": str(e)}


if __name__ == "__main__":
    mcp.run(transport="stdio")
