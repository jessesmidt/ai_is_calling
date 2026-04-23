import os
import json
from typing import Any
from src.models import FunctionDefinition, InputPrompts


def function_loader(filename: str) -> list[FunctionDefinition]:
    """Load function definitions from a JSON file.

    Args:
        filename: Path to the JSON file containing function definitions.

    Returns:
        List of FunctionDefinition objects parsed from the file.

    Raises:
        FileNotFoundError: If the file does not exist.
        ValueError: If the file contains invalid JSON.
    """
    if not os.path.exists(filename):
        raise FileNotFoundError(
            f"Functions file '{filename}' was not found"
        )
    with open(filename, 'r') as f:
        try:
            data: list[Any] = json.load(f)
        except json.JSONDecodeError as e:
            raise ValueError(
                f"Function file '{filename}' contains invalid JSON: {e}"
            )
    return [FunctionDefinition(**item) for item in data]


def prompt_loader(filename: str) -> list[InputPrompts]:
    """Load input prompts from a JSON file.

    Args:
        filename: Path to the JSON file containing prompts.

    Returns:
        List of InputPrompts objects parsed from the file.

    Raises:
        FileNotFoundError: If the file does not exist.
        ValueError: If the file contains invalid JSON.
    """
    if not os.path.exists(filename):
        raise FileNotFoundError(
            f"Prompts file '{filename}' was not found"
        )
    with open(filename, 'r') as f:
        try:
            data: list[Any] = json.load(f)
        except json.JSONDecodeError as e:
            raise ValueError(
                f"Prompts file '{filename}' contains invalid JSON: {e}"
            )
    return [InputPrompts(**item) for item in data]
