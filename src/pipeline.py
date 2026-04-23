import json
import sys
import traceback
from src.models import FunctionDefinition, OutputDefinition, InputPrompts
from src.decoder import Decoder


def pipeline(
    prompts: list[InputPrompts],
    functions: list[FunctionDefinition],
    decoder: Decoder,
) -> list[OutputDefinition]:
    """Process a list of prompts through the constrained decoder.

    Args:
        prompts: Input prompts to decode into function calls.
        functions: Available function definitions for the decoder.
        decoder: Decoder instance for generating function call JSON.

    Returns:
        List of OutputDefinition objects for successfully decoded prompts.
    """
    results: list[OutputDefinition] = []
    for prompt in prompts:
        try:
            result = decoder.decode_prompt(prompt, functions)
            print(f"DEBUG result: {result}")
            result = result.replace('Ġ', ' ')
            data = json.loads(result)
            data['prompt'] = prompt.prompt
            results.append(OutputDefinition(**data))
        except Exception as e:
            print(
                f"Error processing prompt '{prompt.prompt}': {e}",
                file=sys.stderr,
            )
            traceback.print_exc()
    return results
