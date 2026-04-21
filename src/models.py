from pydantic import BaseModel
from enum import Enum
from typing import Dict

class ValidParameter(Enum):
    NUMBER = "number"
    STRING = "string"
    BOOLEAN = "boolean"

class InputPrompts(BaseModel):
    prompt: str

class ParameterType(BaseModel):
    type: ValidParameter

class FunctionDefinition(BaseModel):
    name: str
    description: str
    parameters: Dict[str, ParameterType]
    returns: ParameterType

class OutputDefinition(BaseModel):
    prompt: str
    name: str
    parameters: Dict[str, float | str | bool]

