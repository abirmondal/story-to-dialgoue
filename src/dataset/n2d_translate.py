"""
n2d_translate.py

This module provides functionality to translate narrative-to-story data.
"""

import torch
from transformers import AutoModelForImageTextToText, AutoProcessor, BitsAndBytesConfig

class N2DTranslate:
    """
    Class to handle translation of narrative-to-story data.
    """
    def __init__(
            self,
            translation_model_name: str = "google/translategemma-4b-it",
            device: str = "cuda" if torch.cuda.is_available() else "cpu",
            load_in_4bit: bool = False,
            load_in_8bit: bool = False,
            source_lang: str = "en",
            target_lang: str = "bn"
        ) -> None:
        """
        Initializes the N2DTranslate class.

        Args:
            translation_model_name (str): The name of the translation model to use.
            device (str): The device to run the model on (e.g., "cuda" or "cpu").
            load_in_4bit (bool): Whether to load the model in 4-bit precision.
            load_in_8bit (bool): Whether to load the model in 8-bit precision.
            source_lang (str): The source language code (e.g., "en").
            target_lang (str): The target language code (e.g., "bn").
        """
        self.device = device
        self.source_lang = source_lang
        self.target_lang = target_lang

        # Load the translation model and processor
        if load_in_4bit and device == "cuda":
            # For 4-bit quantization with advanced settings for better performance and quality
            bnb_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_quant_type="nf4",             # Use NF4 for better quality than standard 4-bit
                bnb_4bit_use_double_quant=True,        # Saves even more memory (adds a second quantization step)
                bnb_4bit_compute_dtype=torch.bfloat16  # Recommended for newer models like Gemma
            )
        elif load_in_8bit and device == "cuda":
            # For 8-bit quantization
            bnb_config = BitsAndBytesConfig(load_in_8bit=True)

        # Load the processor
        self.processor = AutoProcessor.from_pretrained(translation_model_name)
        
        # Load the model with appropriate quantization settings
        if device == "cuda" and (load_in_4bit or load_in_8bit):
            self.model = AutoModelForImageTextToText.from_pretrained(
                translation_model_name,
                device_map="auto",
                quantization_config=bnb_config
            )
        elif device == "cuda":
            self.model = AutoModelForImageTextToText.from_pretrained(
                translation_model_name,
                device_map="auto"
            )
        else:
            self.model = AutoModelForImageTextToText.from_pretrained(
                translation_model_name,
                torch_dtype=torch.float32,
                device_map={"": "cpu"},
                low_cpu_mem_usage=True
            )

    def _format_prompt_for_n2d(self, narrative: str, dialogue: str, name_map: dict) -> str:
        """
        Formats the narrative and dialogue into a prompt for the translation model.

        Args:
            narrative (str): The narrative text to be translated.
            dialogue (str): The dialogue text to be translated.
            name_map (dict): A mapping of character names for consistent translation.

        Returns:
            str: The formatted prompt for the translation model.
        """
        instruction = (
            "You are a professional translator. I will provide a Story Narrative and a Dialogue. "
            "Your task is to translate BOTH into Bengali. "
            "\nRules:"
            "\n1. First, translate the 'Narrative' section."
            "\n2. Second, translate the 'Dialogue' section."
            f"\n3. Use these character names: {name_map}."
            "\n4. Use 'Apni' for formal and 'Tumi' for informal tones based on the story."
            "\n5. Keep the format 'Name: Text' for the dialogue."
        )

        content_text = (
            f"{instruction}\n\n"
            f"### English Narrative:\n{narrative}\n\n"
            f"### English Dialogue:\n{dialogue}\n\n"
            "### Bengali Translation (Narrative followed by Dialogue):"
        )

        return [{
            "role": "user",
            "content": [{
                "type": "text",
                "source_lang_code": self.source_lang,
                "target_lang_code": self.target_lang,
                "text": content_text
            }]
        }]

    def translate_names(self, names: list) -> dict:
        """
        Translates character names based.

        Args:
            names (list): A list of character names to be translated.

        Returns:
            dict: A mapping of original names to translated names.
        """
        names_str = ", ".join(names)
        instruction = "Translate the following list of names into Bengali. Output only the Bengali names separated by commas."

        mapping_prompt = f"{instruction}\nNames: {names_str}"

        messages = [{
            "role": "user",
            "content": [{
                "type": "text",
                "source_lang_code": self.source_lang,
                "target_lang_code": self.target_lang,
                "text": mapping_prompt
            }]
        }]

        inputs = self.processor.apply_chat_template(
            messages,
            tokenize=True,
            add_generation_prompt=True,
            return_dict=True,
            return_tensors="pt"
        ).to(self.model.device)
        input_len = len(inputs['input_ids'][0])

        with torch.inference_mode():
            generation = self.model.generate(**inputs, do_sample=False)

        generation = generation[0][input_len:]
        decoded = self.processor.decode(generation, skip_special_tokens=True)
        translated_names = [name.strip() for name in decoded.split(",")]
        
        return dict(zip(names, translated_names))
