import os
import json
from src.models import FunctionDefinition, InputPrompts

def function_loader(filename: str) -> list[FunctionDefinition]:
    if not os.path.exists(filename):
            raise FileNotFoundError(
                f"Functions file '{filename}' was not found"
            )
    with open(filename, 'r') as f:
        try:
            data: list[dict] = json.load(f)
        except json.JSONDecodeError as e:
            raise ValueError(f"Function file '{filename}' contains invalid JSON: {e}")

    return [FunctionDefinition(**item) for item in data]


def prompt_loader(filename: str) -> list[InputPrompts]:
    if not os.path.exists(filename):
            raise FileNotFoundError(
                f"Prompts file '{filename}' was not found"
            )
    with open(filename, 'r') as f:
        try:
            data: list[dict] = json.load(f)
        except json.JSONDecodeError as e:
            raise ValueError(f"Prompts file '{filename}' contains invalid JSON: {e}")

    return [InputPrompts(**item) for item in data]
