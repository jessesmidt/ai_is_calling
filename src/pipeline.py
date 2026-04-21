from src.models import FunctionDefinition, OutputDefinition, InputPrompts, ValidParameter
from src.loader import function_loader, prompt_loader
from src.decoder import Decoder
from src.writer import write_json_output
import traceback
import json
import sys

def pipeline(prompts: list[InputPrompts], functions: list[FunctionDefinition], decoder: Decoder) -> list[OutputDefinition]:
    results: list[OutputDefinition] = []
    for prompt in prompts:
        try:
            result = decoder.decode_prompt(prompt, functions)
            print(f"DEBUG result: {result}")
            data = json.loads(result)
            data['prompt'] = prompt.prompt
            results.append(OutputDefinition(**data))
        except Exception as e:
            print(f"Error processing prompt '{prompt.prompt}': {e}", file=sys.stderr)
            traceback.print_exc()
    return results