DIALOGUE_END_TOKEN = "<end_of_dialogue>" # Token indicating the end of a dialogue session
DEFAULT_SEPARATOR_TOKEN = "</s>" # Default token used to separate different parts of the dialogue. This is the separator of BART and T5 models.

def update_end_token(new_token: str) -> None:
    """
    Updates the default dialogue end token.

    Args:
        new_token (str): The new dialogue end token to be set.
    """
    global DIALOGUE_END_TOKEN
    DIALOGUE_END_TOKEN = new_token

def update_sep_token(new_token: str) -> None:
    """
    Updates the default separator token.

    Args:
        new_token (str): The new separator token to be set.
    """
    global DEFAULT_SEPARATOR_TOKEN
    DEFAULT_SEPARATOR_TOKEN = new_token
