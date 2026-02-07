"""
llama_config.py

This module contains the configuration settings for the LLaMA model used in the story-to-dialogue conversion project.
"""

def get_chat_template(
        narrative: str,
        dialogue: str | None = None,
        keep_assistance: bool = True
    ) -> list:
    """
    Retrieves the chat template based on the specified type.

    Args:
        narrative (str): The narrative text.
        dialogue (str | None): The dialogue text. Optional, defaults to None.
        keep_assistance (bool): Whether to keep the assistance role in the template.

    Returns:
        list: The chat template as a list of dictionaries.
    """
    template = [
        {"role": "system", "content": "Convert this story into a dialogue."},
        {"role": "user", "content": f"Narrative: {narrative}"},
        {"role": "assistant", "content": dialogue}
        ]
    if not keep_assistance:
        template = template[:-1]
    return template
