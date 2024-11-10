# -*- coding: utf-8 -*-


import re
import pandas as pd

def split_identifier(identifier):
    """
    Split camelCase or snake_case identifier into constituent parts.
    """
    # Split snake_case
    parts = identifier.split('_')
    # Further split camelCase for each part
    parts = [re.sub(r'([a-z])([A-Z])', r'\1 \2', part).split() for part in parts]
    # Flatten the list and return as individual tokens
    return [token for sublist in parts for token in sublist]

def tokenize_function_code(code):
    """
    Tokenize Solidity function code into a sequence of tokens, with support for splitting
    camelCase and snake_case identifiers.
    """
    if not isinstance(code, str):
        return None  # Skip non-string entries

    # Tokenize code using regular expressions
    tokens = re.findall(r'[A-Za-z_][A-Za-z0-9_]*|\d+|==|!=|<=|>=|&&|\|\||[+\-*/%<>()=;{},.]', code)

    # Process each token to further split camelCase and snake_case identifiers
    processed_tokens = []
    for token in tokens:
        if re.match(r'^[A-Za-z_][A-Za-z0-9_]*$', token):  # Check if it's an identifier
            processed_tokens.extend(split_identifier(token))  # Split and add parts
        else:
            processed_tokens.append(token)  # Add as is if not an identifier

    return processed_tokens