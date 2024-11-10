import re

def extract_api_sequence(code):
    """
    Extract a comprehensive, ordered API sequence from Solidity code by identifying built-in
    and common contract functions, as well as global variables, preserving the order of appearance.
    """
    if not isinstance(code, str):
        return None  # Skip non-string entries

    # Extended regex to capture built-in functions, common contract methods, events, and global variables
    api_calls = re.findall(r'\b(require|assert|revert|transfer|send|call|delegatecall|balance|address|'
                           r'gasleft|block\.number|block\.timestamp|block\.difficulty|block\.gaslimit|'
                           r'now|keccak256|sha256|ecrecover|addmod|mulmod|emit\s+\w+|approve|'
                           r'balanceOf|transferFrom|safeTransferFrom|safeMint|mint|burn|totalSupply|'
                           r'allowance|safeApprove|increaseAllowance|decreaseAllowance|'
                           r'name|symbol|decimals|permit|msg\.sender|msg\.value|tx\.origin|'
                           r'block\.hash|block\.coinbase)\b', code)

    # Return the sequence as is, preserving the order
    return api_calls

