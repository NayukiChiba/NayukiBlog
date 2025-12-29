import json
from typing import List, Any

def extract_unique_tags(items: List[Any], field_name: str = "tags") -> List[str]:
    """
    Extracts unique tags from a list of objects where tags are stored as a JSON string or list.
    """
    all_tags = set()
    for item in items:
        val = getattr(item, field_name, None)
        if val:
            try:
                # If it's already a list
                if isinstance(val, list):
                    tags_list = val
                else:
                    # Assume JSON string
                    tags_list = json.loads(val)
                
                if isinstance(tags_list, list):
                    for tag in tags_list:
                        all_tags.add(tag)
            except (json.JSONDecodeError, TypeError):
                pass
    return list(sorted(all_tags))
