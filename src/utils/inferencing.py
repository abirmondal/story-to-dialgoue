"""
inferencing.py

Utility functions for model inferencing.
"""

import torch
import warnings
from transformers import (
    AutoModelForSeq2SeqLM,
    AutoTokenizer,
    StoppingCriteria,
    StoppingCriteriaList
)
from peft import PeftModel
from config.dialogue_special_tokens import update_sep_token, DIALOGUE_END_TOKEN, DEFAULT_SEPARATOR_TOKEN

class HFModelForInferencing:
    """
    Class to load and manage HuggingFace models for inferencing.
    """
    def __init__(
            self,
            hf_model_repo_name: str,
            is_lora: bool = False,
            peft_model_repo_name: str | None = None,
            hf_commit_hash: str | None = None,
    ) -> None:
        """
        Initializes the HFModelForInferencing class.

        Args:
            hf_model_repo_name (str): HuggingFace model repository name.
            is_lora (bool): Flag indicating if a LoRA model is used. Defaults to False.
            peft_model_repo_name (str | None): PEFT model repository name. Defaults to None.
            hf_commit_hash (str | None): Specific commit hash for the HuggingFace model. Defaults to None.
        """
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model_repo_name = hf_model_repo_name
        self.is_lora = is_lora
        if peft_model_repo_name and peft_model_repo_name.endswith("-lora") and not is_lora:
            warnings.warn(
                f"The model repository name '{peft_model_repo_name}' suggests it is a LoRA model. "
                "Consider setting 'is_lora=True' to ensure proper loading."
            )
        self.peft_model_repo_name = peft_model_repo_name
        self.commit_hash = hf_commit_hash

        self.tokenizer = AutoTokenizer.from_pretrained(self.peft_model_repo_name, use_fast=True, revision=hf_commit_hash)
        if self.is_lora and self.peft_model_repo_name:
            base_model = AutoModelForSeq2SeqLM.from_pretrained(
                hf_model_repo_name,
                device_map="auto" if self.device == "cuda" else None,
            )
            base_model.resize_token_embeddings(len(self.tokenizer))
            peft_model = PeftModel.from_pretrained(
                base_model,
                self.peft_model_repo_name,
                revision=hf_commit_hash,
                device_map="auto" if self.device == "cuda" else None,
            )
            self.model = peft_model.merge_and_unload()
        else:
            self.model = AutoModelForSeq2SeqLM.from_pretrained(
                hf_model_repo_name,
                revision=hf_commit_hash,
                device_map="auto" if self.device == "cuda" else None,
            )
        self.model.to(self.device)
        self.model.eval()

    def generate_dialogue(
            self,
            input_text: str,
            tokenizer_max_length: int = 1024,
            prefix_prompt: str | None = None,
            gen_turn_by_turn: bool = False,
            max_turns: int | None = None,
            characters: list[str] | None = None,
            separator_token: str = DEFAULT_SEPARATOR_TOKEN,
            generation_kwargs: dict | None = None,
            skip_special_tokens: bool = True,
    ) -> str:
        """
        Generates dialogue text based on the input.

        Args:
            input_text (str): The input text for generation. It can be just the narrative or narrative with dialogue history.
            tokenizer_max_length (int): The maximum length for the tokenizer. Defaults to 1024.
            prefix_prompt (str | None): An optional prefix prompt to prepend to the input text. Defaults to None.
            gen_turn_by_turn (bool): Whether to generate dialogue turn-by-turn. Defaults to False.
            max_turns (int | None): Maximum number of turns to generate when gen_turn_by_turn is True. Defaults to None.
            characters (list[str] | None): List of character names for turn-by-turn generation. Defaults to None.
            separator_token (str): Token used to separate dialogue turns. Defaults to DEFAULT_SEPARATOR_TOKEN.
            generation_kwargs (dict | None): Additional generation keyword arguments. Defaults to None.
                    Possible keys include:
                    - max_new_tokens (int)
                    - no_repeat_ngram_size (int)
                    - repetition_penalty (float)
                    - early_stopping (bool)
                    - pad_token_id (int)
                    - eos_token_id (int)
                    - forced_eos_token_id (int)
            skip_special_tokens (bool): Whether to skip special tokens in the output. Defaults to True.

        Returns:
            str: The generated dialogue text.
        """
        if gen_turn_by_turn and max_turns is None:
            raise ValueError("max_turns must be specified when gen_turn_by_turn is True.")
        if gen_turn_by_turn and (characters is None or len(characters) < 2):
            raise ValueError("At least two character names must be provided when generating dialogue turn-by-turn.")
        if generation_kwargs is None:
            generation_kwargs = {
                "max_new_tokens": 512,
                "no_repeat_ngram_size": 3,
                "repetition_penalty": 1.2,
                "early_stopping": True,
                "pad_token_id": self.tokenizer.pad_token_id,
                "eos_token_id": self.tokenizer.eos_token_id,
            }
        if "num_beams" in generation_kwargs:
            warnings.warn(
                "Beam search is not recommended for dialogue generation as it may lead to generic responses. "
                "Consider using sampling methods instead."
            )
            if "do_sample" in generation_kwargs and generation_kwargs["do_sample"]:
                warnings.warn(
                    "Both 'num_beams' and 'do_sample' are set in generation_kwargs. "
                    "Beam search will take precedence over sampling."
                )
        
        input_text = prefix_prompt + "\n" + input_text if prefix_prompt else input_text
        full_dialogue_output = ""
        
        with torch.no_grad():            
            if gen_turn_by_turn:
                dialogue_history = ""
                for turn_index in range(max_turns):
                    current_character = characters[turn_index % len(characters)]
                    if dialogue_history:
                        prompt = input_text + separator_token + dialogue_history + current_character + ":"
                    else:
                        prompt = input_text + separator_token + current_character + ":"

                    input_ids = self.tokenizer(prompt, return_tensors="pt", truncation=True, max_length=tokenizer_max_length).input_ids.to(self.device)
                    
                    output_ids = self.model.generate(input_ids, **generation_kwargs)

                    generated_text = self.tokenizer.decode(output_ids[0], skip_special_tokens=skip_special_tokens)

                    if DIALOGUE_END_TOKEN in generated_text:
                        # Model wants to end the dialogue
                        cleaned_turn = generated_text.split(DIALOGUE_END_TOKEN)[0].strip()
                        full_dialogue_output += f"{current_character}: " + \
                            cleaned_turn + "\n"
                        break
                    elif self.tokenizer.eos_token and self.tokenizer.eos_token in generated_text:
                        # Model finished a normal turn
                        cleaned_turn = generated_text.split(self.tokenizer.eos_token)[0].strip()
                    else:
                        # Model hit max tokens without stop token
                        cleaned_turn = generated_text.strip()

                    full_dialogue_output += f"{current_character}: " + \
                        cleaned_turn + "\n"
                    
                    dialogue_history += f"{current_character}: " + cleaned_turn + separator_token
            else:
                prompt = input_text
                input_ids = self.tokenizer(prompt, return_tensors="pt", truncation=True, max_length=tokenizer_max_length).input_ids.to(self.device)
                
                output_ids = self.model.generate(input_ids, **generation_kwargs)

                generated_text = self.tokenizer.decode(output_ids[0], skip_special_tokens=skip_special_tokens)
                full_dialogue_output = generated_text
        return full_dialogue_output.strip()