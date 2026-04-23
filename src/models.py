from pydantic import BaseModel
from enum import Enum


class ValidParameter(Enum):
    """Enum of supported parameter types for function definitions."""

    NUMBER = "number"
    STRING = "string"
    BOOLEAN = "boolean"


class InputPrompts(BaseModel):
    """A single user input prompt."""

    prompt: str


class ParameterType(BaseModel):
    """Type descriptor for a single function parameter."""

    type: ValidParameter


class FunctionDefinition(BaseModel):
    """Full specification of a callable function."""

    name: str
    description: str
    parameters: dict[str, ParameterType]
    returns: ParameterType


class OutputDefinition(BaseModel):
    """Result of a decoded function call."""

    prompt: str
    name: str
    parameters: dict[str, float | str | bool]
