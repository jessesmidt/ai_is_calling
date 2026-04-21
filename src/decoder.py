from __future__ import annotations
from typing import List
from src.models import FunctionDefinition, InputPrompts, ValidParameter
import numpy as np
import torch
import json
import traceback

class Decoder:
    def __init__(self, model: "Small_LLM_Model"):
        self.model: "Small_LLM_Model" = model
        self.vocab_path: str = self.model.get_path_to_vocab_file()
        with open(self.vocab_path) as fd:
            self.vocab: dict = json.load(fd)
        self.max_tokens: int = 500
        self.id_to_token = {v: k for k, v in self.vocab.items()}

    def get_token_id(self, token_str: str) -> int:
        return int(self.vocab[token_str])

    def next_valid_token(
        self, logits: list[float], generated: list[str],
        functions: list[FunctionDefinition]) -> int:
        if not generated:
            token_id = self.get_token_id('{"')
            logits = np.full(len(logits), -np.inf)
            logits[token_id] = 0
            return token_id
        elif len(generated) == 1 and generated[0] == '{"':
            token_id = self.get_token_id('name')
            logits = np.full(len(logits), -np.inf)
            logits[token_id] = 0
            return token_id
        elif len(generated) == 2 and generated[1] == 'name':
            token_id = self.get_token_id('":')
            logits = np.full(len(logits), -np.inf)
            logits[token_id] = 0
            return token_id
        elif len(generated) == 3 and generated[2] == '":':
            token_id = self.get_token_id('Ġ"')
            logits = np.full(len(logits), -np.inf)
            logits[token_id] = 0
            return token_id
        elif '",' not in generated:
            # function name section
            name_so_far = "".join(generated[4:])
            valid_functions = [f for f in functions if f.name.startswith(name_so_far)]
            valid_token_ids = []
            for func in valid_functions:
                remaining = func.name[len(name_so_far):]
                for token_str, token_id in self.vocab.items():
                    if remaining.startswith(token_str):
                        valid_token_ids.append(token_id)
            exact_match = [f for f in functions if f.name == name_so_far]
            if exact_match:
                token_id = self.get_token_id('",')
                logits = np.full(len(logits), -np.inf)
                logits[token_id] = 0
                return token_id
            masked_logits = np.full(len(logits), -np.inf)
            for token_id in valid_token_ids:
                masked_logits[token_id] = logits[token_id]
            return int(np.argmax(masked_logits))
        elif '",' in generated and 'Ġ{' not in generated:
            # pre parameter section
            comma_idx = generated.index('",')
            tokens_after_comma = generated[comma_idx + 1:]
            if len(tokens_after_comma) == 0:
                token_id = self.get_token_id('Ġ"')
                logits = np.full(len(logits), -np.inf)
                logits[token_id] = 0
                return token_id
            elif len(tokens_after_comma) == 1:
                token_id = self.get_token_id('parameters')
                logits = np.full(len(logits), -np.inf)
                logits[token_id] = 0
                return token_id
            elif len(tokens_after_comma) == 2:
                token_id = self.get_token_id('":')
                logits = np.full(len(logits), -np.inf)
                logits[token_id] = 0
                return token_id
            elif len(tokens_after_comma) == 3:
                token_id = self.get_token_id('Ġ{')
                logits = np.full(len(logits), -np.inf)
                logits[token_id] = 0
                return token_id
        else:
            # parameter section
            # print(f"DEBUG generated: {generated}")
            func_name = "".join(generated[4:generated.index('",')])
            selected_func = next(f for f in functions if f.name == func_name)
            
            brace_idx = generated.index('Ġ{')
            tokens_after_brace = generated[brace_idx + 1:]
            
            param_names = list(selected_func.parameters.keys())
            current_param_idx = len([t for t in tokens_after_brace if t == ','])
            current_param_name = param_names[current_param_idx]
            current_param_type = selected_func.parameters[current_param_name].type
            
            commas = [i for i, t in enumerate(tokens_after_brace) if t == ',']
            param_start = (commas[-1] + 1) if commas else 0
            current_param_tokens = tokens_after_brace[param_start:]

            if len(current_param_tokens) == 0:
                token_id = self.get_token_id('Ġ"')
                logits = np.full(len(logits), -np.inf)
                logits[token_id] = 0
                return token_id
            elif len(current_param_tokens) == 1:
                token_id = self.get_token_id(current_param_name)
                logits = np.full(len(logits), -np.inf)
                logits[token_id] = 0
                return token_id
            elif len(current_param_tokens) == 2:
                token_id = self.get_token_id('":')
                logits = np.full(len(logits), -np.inf)
                logits[token_id] = 0
                return token_id
            else:
                if current_param_type == ValidParameter.NUMBER:
                    value_tokens = current_param_tokens[3:]
                    if not value_tokens:
                        valid_token_ids = []
                        for token_str, tid in self.vocab.items():
                            stripped = token_str.replace('Ġ', '')
                            if all(c in '0123456789.-' for c in stripped) and stripped:
                                valid_token_ids.append(tid)
                        masked_logits = np.full(len(logits), -np.inf)
                        for tid in valid_token_ids:
                            masked_logits[tid] = logits[tid]
                        return int(np.argmax(masked_logits))
                    else:
                        if current_param_idx == len(param_names) - 1:
                            token_id = self.get_token_id('}')
                        else:
                            token_id = self.get_token_id(',')
                        logits = np.full(len(logits), -np.inf)
                        logits[token_id] = 0
                        return token_id
                elif current_param_type == ValidParameter.STRING:
                    value_tokens = current_param_tokens[3:]
                    if not value_tokens:
                        token_id = self.get_token_id('Ġ"')
                        logits = np.full(len(logits), -np.inf)
                        logits[token_id] = 0
                        return token_id
                    elif '"' in value_tokens:
                        if current_param_idx == len(param_names) - 1:
                            token_id = self.get_token_id('}')
                            logits = np.full(len(logits), -np.inf)
                            logits[token_id] = 0
                            return token_id
                        else:
                            return int(np.argmax(logits))
                    else:
                        valid_token_ids = list(self.vocab.values())
                        masked_logits = np.full(len(logits), -np.inf)
                        for tid in valid_token_ids:
                            masked_logits[tid] = logits[tid]
                        return int(np.argmax(masked_logits))
                elif current_param_type == ValidParameter.BOOLEAN:
                    value_tokens = current_param_tokens[3:]
                    if not value_tokens:
                        valid_token_ids = []
                        for token_str, tid in self.vocab.items():
                            stripped = token_str.replace('Ġ', '')
                            if stripped in ['true', 'false']:
                                valid_token_ids.append(tid)
                        masked_logits = np.full(len(logits), -np.inf)
                        for tid in valid_token_ids:
                            masked_logits[tid] = logits[tid]
                        return int(np.argmax(masked_logits))
                    else:
                        if current_param_idx == len(param_names) - 1:
                            token_id = self.get_token_id('}')
                        else:
                            token_id = self.get_token_id(',')
                        logits = np.full(len(logits), -np.inf)
                        logits[token_id] = 0
                        return token_id  
            raise ValueError(f"Unexpected state in decoder: generated={generated}")


    def decode_prompt(self, input: InputPrompts, functions: list[FunctionDefinition]) -> str:
        stack: list[str] = []
        generated: list[str] = []

        token_ids: list[int] = self.model.encode(input.prompt).squeeze().tolist()
        while len(generated) < self.max_tokens:
            logits: list[float] = self.model.get_logits_from_input_ids(token_ids)
            next_token: int = self.next_valid_token(logits, generated, functions)
            if self.id_to_token[next_token] == "{":
                stack.append('{')
            elif self.id_to_token[next_token] == "}":
                if stack:
                    stack.pop()
                    if not stack:
                        break
            generated.append(self.id_to_token[next_token])
            token_ids.append(next_token)
        return "".join(generated)
