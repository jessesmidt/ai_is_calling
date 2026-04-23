*This project has been created as part of the 42 curriculum by jsmidt.*

# call me maybe

## Description

**call me maybe** is a function calling system that translates natural language prompts into structured JSON function calls using a small LLM (Qwen3-0.6B) with constrained decoding.

Given a prompt like `"What is the sum of 2 and 3?"`, the system does not answer the question. Instead it produces:

```json
{
    "prompt": "What is the sum of 2 and 3?",
    "name": "fn_add_numbers",
    "parameters": {"a": 2.0, "b": 3.0}
}
```

The core challenge is reliability: small language models only produce valid JSON about 30% of the time when prompted naively. This project solves that using constrained decoding — a technique that guides the model's output token-by-token to guarantee 100% structurally valid JSON output.

---

## Instructions

### Requirements

- Python 3.13+
- `uv` package manager
- The `llm_sdk` package (copy into the project root manually)
- 2 GB of free space for installation

### Installation

```bash
git clone <your-repo-url>
cd callme
cp -r /path/to/llm_sdk ./llm_sdk
uv sync
```

### Running

```bash
uv run python -m src \
    --functions_definition data/input/functions_definition.json \
    --input data/input/function_calling_tests.json \
    --output data/output/results.json
```

All arguments are optional — by default the program reads from `data/input/` and writes to `data/output/`.

### Makefile commands

```bash
make install   # install dependencies
make run       # run with default arguments
make debug     # run with pdb debugger
make lint      # run flake8 and mypy
make clean     # remove __pycache__ and .mypy_cache
```

---

## Algorithm Explanation

The constrained decoding pipeline works as follows:

### 1. Context construction
Before encoding the prompt, a context string is built that lists all available functions with their parameter names and types. This gives the model the information it needs to select the right function.

### 2. Token-by-token generation
The model generates output one token at a time. At each step:
- The model produces logits — a probability score for every token in the vocabulary (~50,000 tokens)
- Our decoder intervenes before token selection, masking invalid tokens to `-inf`
- Only valid tokens can be selected

### 3. State machine
The decoder tracks which part of the JSON it is currently generating through a set of states:

| State | What gets forced |
|---|---|
| Opening | `{"` → `name` → `":` → `Ġ"` |
| Function name | LLM picks from valid function name continuations |
| Pre-parameters | `","` → `Ġ"parameters":` → `Ġ{` |
| Parameter name | LLM picks token-by-token until name matches a known parameter |
| Parameter value | Constrained by type (number/string/boolean) |
| Closing | `}` → `}` |

### 4. Type-specific value generation

- **NUMBER** — only tokens containing digits, `.`, or `-` are allowed. The closer (`}` or `,`) is added to valid tokens once a digit has been generated, letting the model decide when the number is complete.
- **STRING** — near-free generation using raw logits, but structural tokens (`}`, newlines, combined `",` tokens) are blocked to prevent the model from accidentally closing the JSON mid-string. A repetition detector closes the string if the model gets stuck in a loop.
- **BOOLEAN** — only `true` and `false` tokens are allowed.

### 5. Stack-based termination
A stack tracks opening braces. When the stack empties after the final `}`, the generation loop breaks.

---

## Design Decisions

**Plain functions over classes** — `loader.py`, `pipeline.py`, and `writer.py` are plain functions since they hold no state. The `Decoder` class is the one exception because it loads the model and vocabulary once at initialization and reuses them across all prompts.

**Pydantic for validation** — all input and output data is validated through pydantic models (`FunctionDefinition`, `InputPrompts`, `OutputDefinition`). This catches malformed JSON files early with clear error messages.

**Vocabulary inversion** — the SDK provides a `token → id` mapping. We invert it to `id → token` at initialization so we can efficiently look up what string any generated token ID represents.

**Context prompt** — passing function names, parameter names, and descriptions to the model before the user prompt significantly improves function selection accuracy, especially for small models.

**Raw logits for strings** — for string values, using raw (unmasked) logits produces better results than heavily filtered logits. The model's natural language understanding is trusted for content generation; only structural tokens are blocked.

---

## Performance Analysis

Tested on the provided 11-prompt test suite and an extended 84-prompt suite:

- **Function selection accuracy**: ~90%+ on clear prompts. The model occasionally picks the wrong function for ambiguous prompts (e.g. selecting `fn_get_square_root` for unrelated prompts).
- **JSON validity**: 100% — every output is parseable and schema-compliant due to constrained decoding.
- **Integer parameters**: ~95%+ accuracy. The model correctly extracts most integer values from prompts.
- **Decimal parameters**: ~60% accuracy. The 0.6B model struggles to continue past the first digit of a decimal number (e.g. `0.5` → `0.0`). This is a known limitation of the model size.
- **String parameters**: ~85%+ accuracy. Values are correctly extracted in most cases. Complex regex patterns may be truncated by the repetition detector.
- **Speed**: ~5-15 seconds per prompt on a modern multi-core CPU. Full 11-prompt suite completes in under 3 minutes.

---

## Challenges Faced

**Multi-character tokens** — the tokenizer frequently combines multiple characters into a single token (e.g. `')"'`, `'",`', `'}}Ċ'`). These tokens contain structural characters that can bypass the state machine if not handled explicitly. The solution was to block any token containing both `"` and `,` from string generation, and to handle multi-character closing tokens explicitly in the state machine.

**Vocabulary structure** — the vocabulary JSON maps token strings to IDs, not IDs to strings. This required building an inverted map at initialization. Additionally, token IDs in the vocabulary are stored as Python `int` values but can sometimes behave as tensors, requiring explicit `int()` casting throughout.

**Tensor vs list** — the SDK's `encode()` method returns a 2D tensor (`[[id1, id2, ...]]`). It needed `.squeeze().tolist()` to convert to a plain Python list before appending new tokens.

**Model repetition** — the 0.6B model gets stuck generating the same tokens in a loop, particularly for regex patterns and string values. A sliding window repetition detector (checking if the last 3 tokens have appeared anywhere earlier in the value) forces the string to close when a loop is detected.

**String closing detection** — detecting the end of a string value is non-trivial because the closing `"` may appear as a standalone token, as part of `Ġ"`, or as part of a combined token like `')"'`. Each case required explicit handling.

---

## Testing Strategy

The implementation was validated by:

1. **Debug token tracing** — printing each generated token and its index during development to verify the state machine transitions correctly.
2. **Incremental testing** — testing one prompt at a time during development before running the full suite.
3. **Edge case prompts** — testing with prompts that have no clear numeric arguments (to verify the model doesn't hallucinate numbers), prompts with decimal values, prompts with complex regex patterns, and prompts with multi-word string values.
4. **Extended test suite** — running against 84 prompts covering all 7 functions including edge cases like `0.5`, negative numbers, long strings, and repeated substrings.

---

## Example Usage

Run with default paths:
```bash
uv run python -m src
```

Run with custom paths:
```bash
uv run python -m src \
    --functions_definition data/input/functions_definition.json \
    --input data/input/function_calling_tests.json \
    --output data/output/results.json
```

Example output for `"Greet shrek"`:
```json
{
    "prompt": "Greet shrek",
    "name": "fn_greet",
    "parameters": {
        "name": "shrek"
    }
}
```

Example output for `"What is the sum of 2 and 3?"`:
```json
{
    "prompt": "What is the sum of 2 and 3?",
    "name": "fn_add_numbers",
    "parameters": {
        "a": 2.0,
        "b": 3.0
    }
}
```

---

## Resources

### References
- [Qwen3 model on HuggingFace](https://huggingface.co/Qwen/Qwen3-0.6B)
- [Outlines — structured text generation](https://github.com/outlines-dev/outlines) *(forbidden in this project but useful for understanding the concept)*
- [lm-format-enforcer](https://github.com/noamgat/lm-format-enforcer) *(similar approach)*
- [Pydantic documentation](https://docs.pydantic.dev/)
- [Byte Pair Encoding tokenization](https://huggingface.co/learn/nlp-course/chapter6/5)
- [JSON specification](https://www.json.org/json-en.html)

### AI Usage
Claude (Anthropic) was used throughout this project for:
- **Architecture decisions** — discussing module structure, OOP vs functional design, and where classes add value.
- **Debugging** — interpreting error messages, tracing token generation issues, and identifying edge cases in the state machine.
- **Understanding concepts** — clarifying how tokenizers work, what logits represent, and how constrained decoding differs from naive prompting.
- **Code review** — catching type errors, import issues, and logic bugs during development.

All code was written and understood incrementally. No code was copied without understanding its purpose.