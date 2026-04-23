import sys
from src.pipeline import pipeline
from src.loader import function_loader, prompt_loader
from src.decoder import Decoder
from src.writer import write_json_output
from src.models import InputPrompts, FunctionDefinition


def parse_args(argv: list[str]) -> dict[str, str | int | None]:
    """Parse command-line arguments for the callme pipeline.

    Args:
        argv: Argument list, typically sys.argv.

    Returns:
        Dictionary mapping argument names to their parsed values.

    Raises:
        ValueError: If a required value is missing or an unknown argument
            is given.
    """
    args: dict[str, str | int | None] = {
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


def main() -> None:
    """Run the callme function-calling pipeline."""
    try:
        from llm_sdk import Small_LLM_Model
    except ImportError:
        print(
            "Error: llm_sdk not found. Please copy the llm_sdk folder "
            "into the project root.",
            file=sys.stderr,
        )
        sys.exit(1)
    try:
        flags = parse_args(sys.argv)
    except ValueError as e:
        print(f"Error: {e}")
        print("Usage:")
        print(
            "uv run python -m src "
            "[--functions_definition <file>] "
            "[--input <file>] "
            "[--output <file>] "
            "[--visual]"
        )
        sys.exit(1)
    if flags["functions_definition"] is None:
        flags["functions_definition"] = "data/input/functions_definition.json"
    if flags["input"] is None:
        flags["input"] = "data/input/"
    if flags["output"] is None:
        flags["output"] = "data/output/"

    funcs_def = str(flags["functions_definition"])
    input_path = str(flags["input"])
    output_path = str(flags["output"])

    prompts: list[InputPrompts] = prompt_loader(input_path)
    functions: list[FunctionDefinition] = function_loader(funcs_def)

    model = Small_LLM_Model()
    decoder: Decoder = Decoder(model)

    output = pipeline(prompts, functions, decoder)
    write_json_output(output_path, output)


if __name__ == "__main__":
    main()
