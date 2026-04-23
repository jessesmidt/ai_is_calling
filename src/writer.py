import json
import os
from src.models import OutputDefinition


def write_json_output(filename: str, output: list[OutputDefinition]) -> None:
    """Write OutputDefinition objects to a JSON file.

    Args:
        filename: Path to the output JSON file.
        output: List of OutputDefinition objects to serialize.
    """
    os.makedirs(os.path.dirname(filename), exist_ok=True)
    with open(filename, 'w') as f:
        json.dump([item.model_dump() for item in output], f, indent=4)
