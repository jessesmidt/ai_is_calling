import sys
import json
from src.pipeline import pipeline
from src.loader import function_loader, prompt_loader
from src.decoder import Decoder
from src.writer import write_json_output


def parse_args(argv: list[str]) -> dict:
    args = {
        "functions_definition": None,
        "input": None,
        "output": None,
        "visual": 0,
    }
    i = 1
    while i < len(argv):
        arg = argv[i]
        if arg == "--functions_definition":
            if i + 1 >= len(argv):
                raise ValueError("Missing value for --functions_definition")
            args["functions_definition"] = argv[i + 1]
            i += 2
        elif arg == "--input":
            if i + 1 >= len(argv):
                raise ValueError("Missing value for --input")
            args["input"] = argv[i + 1]
            i += 2
        elif arg == "--output":
            if i + 1 >= len(argv):
                raise ValueError("Missing value for --output")
            args["output"] = argv[i + 1]
            i += 2
        elif arg == "--visual":
            args["visual"] = 1
            i += 1
        else:
            raise ValueError(f"Unknown argument: {arg}")
    return args


def main():

    try:
        from llm_sdk import Small_LLM_Model
    except ImportError:
        print("Error: llm_sdk not found. Please copy the llm_sdk folder into the project root.", file=sys.stderr)
        sys.exit(1)
    try:
        flags = parse_args(sys.argv)
    except ValueError as e:
        print(f"Error: {e}")
        print("Usage:")
        print("uv run python -m src "
              "[--functions_definition <file>] "
              "[--input <file>] "
              "[--output <file>] "
              "[--visual]")
        sys.exit(1)
    if flags["functions_definition"] is None:
        flags["functions_definition"] = "data/input/functions_definition.json"
    if flags["input"] is None:
        flags["input"] = "data/input/"
    if flags["output"] is None:
        flags["output"] = "data/output/"

    prompts: list[InputPrompts] = prompt_loader(flags["input"])
    functions: list[FunctionDefinition] = function_loader(flags["functions_definition"])

    model = Small_LLM_Model()
    path = model.get_path_to_vocab_file()
    with open(path) as f:
        vocab = json.load(f)

    decoder: "Decoder" = Decoder(model)

    output = pipeline(prompts, functions, decoder)
    write_json_output(flags["output"], output)
    


if __name__ == "__main__":
    main()