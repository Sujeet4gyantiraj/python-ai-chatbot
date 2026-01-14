from langchain_core.tools import tool

# Example 1: get user ticket
@tool
def get_user_ticket(user_id: str):
    """Get support ticket details for a user"""
    # Example: replace with DB call
    return {
        "user_id": user_id,
        "ticket_id": "TCK123",
        "status": "Open",
        "priority": "High"
    }

# Example 2: create ticket
@tool
def create_ticket(issue: str):
    """Create a new support ticket"""
    return {
        "ticket_id": "TCK999",
        "status": "Created",
        "issue": issue
    }
