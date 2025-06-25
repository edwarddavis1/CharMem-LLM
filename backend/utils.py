import json


def parse_websocket_message(data: str):
    message_data = json.loads(data)
    message_type = message_data.get("type", "unknown")

    user_message = message_data.get("content", "").strip()
    current_page = message_data.get("current_page", 1)
    total_pages = message_data.get("total_pages", 1)

    return {
        "type": message_type,
        "content": user_message,
        "current_page": current_page,
        "total_pages": total_pages,
    }
