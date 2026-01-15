from langchain_core.tools import tool
import requests


# Example 1: get user ticket
@tool
def get_user_ticket(user_id: str):
    """Get support ticket details for a user"""
    # Example: replace with DB call
    return {
        "user_id": user_id,
        "ticket_id": "TCK123",
        "status": "Open",
        "priority": "High",
    }


# Example 2: create ticket
@tool
def create_ticket(issue: str):
    """Create a new support ticket"""
    return {
        "ticket_id": "TCK999",
        "status": "Created",
        "issue": issue,
    }


@tool
def web_search(query: str) -> str:
    """Search the web for up‑to‑date information about a query.

    Uses DuckDuckGo's instant answer API (no API key required) and
    returns a short summary plus a few related titles/links.
    """
    try:
        resp = requests.get(
            "https://api.duckduckgo.com/",
            params={
                "q": query,
                "format": "json",
                "no_html": 1,
                "skip_disambig": 1,
            },
            timeout=10,
        )
        resp.raise_for_status()
        data = resp.json()

        abstract = data.get("AbstractText") or "No direct summary available."

        # Collect a few related topics with URLs
        related_items = []
        for item in data.get("RelatedTopics", [])[:3]:
            if isinstance(item, dict) and item.get("Text") and item.get("FirstURL"):
                related_items.append(f"- {item['Text']} ({item['FirstURL']})")

        related_block = "\n".join(related_items) if related_items else "No related links found."

        return f"Summary: {abstract}\n\nTop results:\n{related_block}"

    except Exception as e:
        return f"Web search failed: {e}"
