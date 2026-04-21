import json
import os
from typing import List
from src.models import OutputDefinition

def write_json_output(filename: str, output: List[OutputDefinition]) -> None:
    os.makedirs(os.path.dirname(filename), exist_ok=True)
    with open(filename, 'w') as f:
        json.dump([item.model_dump() for item in output], f, indent=4)

