from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import json

from src.models import FunctionDefinition, InputPrompts, ValidParameter

if TYPE_CHECKING:
    from llm_sdk import Small_LLM_Model


class Decoder:
    """Constrains LLM output to valid function call JSON via logit masking."""

    def __init__(self, model: Small_LLM_Model) -> None:
        """Initialize the decoder with an LLM model.

        Args:
            model: A Small_LLM_Model instance providing token encoding and
                logit generation.
        """
        self.model: Small_LLM_Model = model
        self.vocab_path: str = self.model.get_path_to_vocab_file()
        with open(self.vocab_path) as fd:
            self.vocab: dict[str, int] = json.load(fd)
        self.max_tokens: int = 85
        self.id_to_token: dict[int, str] = {
            v: k for k, v in self.vocab.items()
        }

    def get_token_id(self, token_str: str) -> int:
        """Return the vocabulary token ID for a given token string.

        Args:
            token_str: The token string to look up.

        Returns:
            The integer token ID.
        """
        return int(self.vocab[token_str])

    def _token_for_struct(self, generated: list[str]) -> int:
        """Return the forced token for one of the first 4 structural positions.

        The JSON output always starts with the fixed sequence:
        '{"', 'name', '":', 'Ġ"'

        Args:
            generated: Token strings generated so far (length must be < 4).

        Returns:
            The token ID for the current structural position.
        """
        fixed = ['{"', 'name', '":', 'Ġ"']
        return self.get_token_id(fixed[len(generated)])

    def _token_for_name(
        self,
        logits: list[float],
        generated: list[str],
        functions: list[FunctionDefinition],
    ) -> int:
        """Constrain output to valid function name prefixes.

        Args:
            logits: Raw logit scores from the model.
            generated: Token strings generated so far.
            functions: Available function definitions.

        Returns:
            The token ID that extends or completes a valid function name,
            or the '","' token ID when the name is complete.
        """
        name_so_far = "".join(generated[4:])
        valid_functions = [
            f for f in functions if f.name.startswith(name_so_far)
        ]
        if any(f.name == name_so_far for f in functions):
            return self.get_token_id('",')
        valid_token_ids = []
        for func in valid_functions:
            remaining = func.name[len(name_so_far):]
            for token_str, token_id in self.vocab.items():
                if remaining.startswith(token_str):
                    valid_token_ids.append(token_id)
        masked_logits = np.full(len(logits), -np.inf)
        for token_id in valid_token_ids:
            masked_logits[token_id] = logits[token_id]
        return int(np.argmax(masked_logits))

    def _token_for_pre_param(self, generated: list[str]) -> int:
        """Emit the fixed ' "parameters": {' sequence after the function name.

        Args:
            generated: Token strings generated so far.

        Returns:
            The next fixed token ID in the pre-parameter sequence.

        Raises:
            ValueError: If an unexpected number of tokens follows '",'.
        """
        comma_idx = generated.index('",')
        n = len(generated) - comma_idx - 1
        fixed: dict[int, str] = {0: 'Ġ"', 1: 'parameters', 2: '":', 3: 'Ġ{'}
        if n in fixed:
            return self.get_token_id(fixed[n])
        raise ValueError(
            f"Unexpected pre-parameter state: {n} tokens after comma"
        )

    def _token_for_param(
        self,
        logits: list[float],
        generated: list[str],
        functions: list[FunctionDefinition],
    ) -> int:
        """Select the next token within the parameters object.

        Handles parameter name building and value generation for NUMBER,
        STRING, and BOOLEAN types.

        Args:
            logits: Raw logit scores from the model.
            generated: Token strings generated so far.
            functions: Available function definitions.

        Returns:
            The next valid token ID for the current parameter.

        Raises:
            ValueError: If an unsupported parameter type is encountered.
        """
        func_name = "".join(generated[4:generated.index('",')])
        selected_func = next(
            f for f in functions if f.name == func_name
        )

        brace_idx = generated.index('Ġ{')
        tokens_after_brace = generated[brace_idx + 1:]

        param_names = list(selected_func.parameters.keys())
        current_param_idx = len(
            [t for t in tokens_after_brace if t == ',']
        )

        if (
            '}' in tokens_after_brace
            or current_param_idx >= len(param_names)
        ):
            return self.get_token_id('}')

        current_param_name = param_names[current_param_idx]
        current_param_type = (
            selected_func.parameters[current_param_name].type
        )
        is_last = current_param_idx == len(param_names) - 1
        closer_str = '}' if is_last else ','
        closer = self.get_token_id(closer_str)

        commas = [
            i for i, t in enumerate(tokens_after_brace) if t == ','
        ]
        param_start = (commas[-1] + 1) if commas else 0
        current_param_tokens = tokens_after_brace[param_start:]

        print(f"current param tokens = {current_param_tokens}")
        if len(current_param_tokens) == 0:
            return self.get_token_id('Ġ"')
        elif '":' not in current_param_tokens:
            name_so_far = "".join(current_param_tokens[1:])
            if name_so_far == current_param_name:
                return self.get_token_id('":')
            remaining = current_param_name[len(name_so_far):]
            valid_token_ids = [
                int(tid) for token_str, tid in self.vocab.items()
                if remaining.startswith(token_str)
            ]
            masked_logits = np.full(len(logits), -np.inf)
            for tid in valid_token_ids:
                masked_logits[tid] = logits[tid]
            return int(np.argmax(masked_logits))
        else:
            colon_idx = current_param_tokens.index('":')
            value_tokens = current_param_tokens[colon_idx + 1:]

            if current_param_type == ValidParameter.NUMBER:
                # print("param = number")
                if len(value_tokens) >= 10:
                    return closer
                valid_token_ids = []
                for token_str, tid in self.vocab.items():
                    stripped = token_str.replace('Ġ', '')
                    if (
                        all(c in '0123456789.-' for c in stripped)
                        and stripped
                        and any(c.isdigit() for c in stripped)
                    ):
                        valid_token_ids.append(int(tid))
                if value_tokens:
                    valid_token_ids.append(closer)
                masked_logits = np.full(len(logits), -np.inf)
                for tid in valid_token_ids:
                    masked_logits[tid] = logits[tid]
                return int(np.argmax(masked_logits))

            elif current_param_type == ValidParameter.STRING:
                content_tokens = value_tokens[1:] if value_tokens else []
                if len(content_tokens) >= 6:
                    last = content_tokens[-3:]
                    earlier = content_tokens[:-3]
                    for i in range(len(earlier) - 2):
                        if earlier[i:i+3] == last:
                            return self.get_token_id('"')
                endings = ('])', ']', ']+', ')"')
                if content_tokens and content_tokens[-1] in endings:
                    if content_tokens[-1] in (')', ')"'):
                        return closer
                    return self.get_token_id('"')
                if not value_tokens:
                    return self.get_token_id('Ġ"')
                elif '"' in content_tokens or 'Ġ"' in content_tokens:
                    last_tok = content_tokens[-1] if content_tokens else ''
                    if ',' in last_tok:
                        return self.get_token_id('Ġ"')
                    return closer
                else:
                    valid_token_ids = []
                    for token_str, tid in self.vocab.items():
                        if (
                            '}' not in token_str
                            and 'Ċ' not in token_str
                            and not ('"' in token_str and ',' in token_str)
                            and token_str not in ("',", "'}", ',', 'Ġ,')
                        ):
                            valid_token_ids.append(int(tid))
                    masked_logits = np.full(len(logits), -np.inf)
                    for tid in valid_token_ids:
                        masked_logits[tid] = logits[tid]
                    return int(np.argmax(masked_logits))

            elif current_param_type == ValidParameter.BOOLEAN:
                # print("param = bool")
                if not value_tokens:
                    valid_token_ids = [
                        int(tid) for token_str, tid in self.vocab.items()
                        if token_str.replace('Ġ', '') in ('true', 'false')
                    ]
                    masked_logits = np.full(len(logits), -np.inf)
                    for tid in valid_token_ids:
                        masked_logits[tid] = logits[tid]
                    return int(np.argmax(masked_logits))
                return closer

            raise ValueError(
                f"Unexpected state in decoder: generated={generated}"
            )

    def next_valid_token(
        self,
        logits: list[float],
        generated: list[str],
        functions: list[FunctionDefinition],
    ) -> int:
        """Select the next valid token ID using constrained decoding.

        Dispatches to a phase-specific helper based on how many tokens
        have been generated and which structural markers are present.

        Args:
            logits: Raw logit scores from the model for each vocab token.
            generated: Token strings generated so far.
            functions: Available function definitions.

        Returns:
            The token ID of the next valid token.
        """
        if len(generated) < 4:
            return self._token_for_struct(generated)
        elif '",' not in generated:
            return self._token_for_name(logits, generated, functions)
        elif 'Ġ{' not in generated:
            return self._token_for_pre_param(generated)
        else:
            return self._token_for_param(logits, generated, functions)

    def decode_prompt(
        self,
        input: InputPrompts,
        functions: list[FunctionDefinition],
    ) -> str:
        """Decode a prompt into a function call JSON string.

        Args:
            input: The input prompt to decode.
            functions: Available function definitions.

        Returns:
            A JSON string representing the decoded function call.
        """
        stack: list[str] = []
        generated: list[str] = []
        should_break: bool = False

        context = "Available functions:\n"
        for func in functions:
            params = ", ".join(
                f"{k}: {v.type.value}"
                for k, v in func.parameters.items()
            )
            context += f"- {func.name}({params}): {func.description}\n"
        context += (
            "\nChoose the most appropriate function for this request "
            f"and call the appropiate function: {input.prompt}"
            "\n\nResponse:"
        )

        token_ids: list[int] = self.model.encode(context).squeeze().tolist()
        while len(generated) < self.max_tokens and not should_break:
            last = generated[-1] if generated else 'start'
            # print(f"DEBUG token {len(generated)}: {last}")
            logits = self.model.get_logits_from_input_ids(token_ids)
            next_token: int = self.next_valid_token(
                logits, generated, functions
            )
            if self.id_to_token[next_token] in ["Ġ{", '{"']:
                stack.append('{')
            elif self.id_to_token[next_token] == "}":
                for char in self.id_to_token[next_token]:
                    if char == '{':
                        stack.append('{')
                    elif char == '}':
                        if stack:
                            stack.pop()
                            if not stack:
                                should_break = True
                                break
            generated.append(self.id_to_token[next_token])
            token_ids.append(next_token)
            if should_break:
                break
        return "".join(generated)
