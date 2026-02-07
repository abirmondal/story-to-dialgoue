"""
inferencing.py

Utility functions for model inferencing.
"""

import torch
import warnings
from transformers import (
    AutoModelForSeq2SeqLM,
    AutoModelForCausalLM,
    AutoTokenizer,
    AutoConfig
)
from peft import PeftModel
from config.dialogue_special_tokens import DIALOGUE_END_TOKEN, DEFAULT_SEPARATOR_TOKEN
from config.llama_config import get_chat_template

class HFModelForInferencing:
    """
    Class to load and manage HuggingFace models for inferencing.
    """
    def __init__(
            self,
            hf_model_repo_name: str,
            is_lora: bool = False,
            peft_model_repo_name: str | None = None,
            merge_lora: bool = True,
            hf_commit_hash: str | None = None,
            device: str = "auto",
            access_token: str | None = None
    ) -> None:
        """
        Initializes the HFModelForInferencing class.

        Args:
            hf_model_repo_name (str): HuggingFace model repository name.
            is_lora (bool): Flag indicating if a LoRA model is used. Defaults to False.
            peft_model_repo_name (str | None): PEFT model repository name. Defaults to None.
            merge_lora (bool): Whether to merge the LoRA weights into the base model for faster inference, but it can exceed memory capacity of the device (like GPU VRAM). Defaults to True.
            hf_commit_hash (str | None): Specific commit hash for the HuggingFace model. Defaults to None.
            device (str): Device to load the model on. Can be "auto", "cuda", or "cpu". Defaults to "auto".
            access_token (str | None): HuggingFace access token for private models. Defaults to None.
        """
        if device == "auto":
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        elif device in ["cuda", "cpu"]:
            self.device = device
        else:
            raise ValueError("Invalid device specified. Choose 'auto', 'cuda', or 'cpu'.")
        
        self.model_repo_name = hf_model_repo_name
        self.is_lora = is_lora
        if peft_model_repo_name and peft_model_repo_name.endswith("-lora") and not is_lora:
            warnings.warn(
                f"The model repository name '{peft_model_repo_name}' suggests it is a LoRA model. "
                "Consider setting 'is_lora=True' to ensure proper loading."
            )
        self.peft_model_repo_name = peft_model_repo_name
        if hf_commit_hash is None:
            hf_commit_hash = "main"
        self.commit_hash = hf_commit_hash

        self.tokenizer = AutoTokenizer.from_pretrained(
            self.peft_model_repo_name if (
                is_lora and self.peft_model_repo_name) else hf_model_repo_name,
            use_fast=True,
            revision=hf_commit_hash,
            token=access_token
        )

        # Get the base model config
        config = AutoConfig.from_pretrained(hf_model_repo_name, token=access_token)
        model_arch = config.architectures[0].lower(
            ) if config.architectures else ""

        if "llama" in model_arch or "bloom" in model_arch or "gpt" in model_arch or "opt" in model_arch or "causal" in model_arch:
            model_class = AutoModelForCausalLM
            self.__is_casual = True
            self.tokenizer.padding_side = "left"  # Causal models typically require left padding
            if not self.tokenizer.pad_token:
                # If the tokenizer doesn't have a pad token, we set it to the eos token
                self.tokenizer.pad_token = self.tokenizer.eos_token
        else:
            model_class = AutoModelForSeq2SeqLM
            self.__is_casual = False
        
        self.model = model_class.from_pretrained(
            hf_model_repo_name,
            device_map="auto" if self.device == "cuda" else None,
            token=access_token
        )
        self.model.resize_token_embeddings(len(self.tokenizer))
        
        if self.is_lora and self.peft_model_repo_name:
            self.model = PeftModel.from_pretrained(
                self.model,
                self.peft_model_repo_name,
                revision=hf_commit_hash,
                device_map="auto" if self.device == "cuda" else None,
                token=access_token
            )
            if merge_lora:
                self.model = self.model.merge_and_unload()

        self.model.to(self.device)
        self.model.eval()

    def generate_dialogue(
            self,
            narrative: str,
            tokenizer_max_length: int = 1024,
            max_new_tokens: int = 512,
            prefix_prompt: str | None = None,
            gen_turn_by_turn: bool = False,
            max_turns: int | None = None,
            characters: list[str] | None = None,
            separator_token: str = DEFAULT_SEPARATOR_TOKEN,
            skip_special_tokens: bool = True,
            **generation_kwargs
    ) -> str:
        """
        Generates dialogue text based on the input.

        Args:
            narrative (str): The input narrative for generation.
            tokenizer_max_length (int): The maximum length for the tokenizer. Defaults to 1024.
            max_new_tokens (int): The maximum number of new tokens to generate. Defaults to 512.
            prefix_prompt (str | None): An optional prefix prompt to prepend to the input text. Defaults to None.
            gen_turn_by_turn (bool): Whether to generate dialogue turn-by-turn. Defaults to False.
            max_turns (int | None): Maximum number of turns to generate when gen_turn_by_turn is True. Defaults to None.
            characters (list[str] | None): List of character names for turn-by-turn generation. Defaults to None.
            separator_token (str): Token used to separate dialogue turns. Defaults to DEFAULT_SEPARATOR_TOKEN.
            skip_special_tokens (bool): Whether to skip special tokens in the output. Defaults to True.
            generation_kwargs (dict): Additional keyword arguments to pass to the model's generate function.
                    
                    Possible keys include -
                    - num_beams (int)
                    - do_sample (bool)
                    - max_new_tokens (int)
                    - no_repeat_ngram_size (int)
                    - repetition_penalty (float)
                    - early_stopping (bool)
                    - pad_token_id (int)
                    - eos_token_id (int)
                    - forced_eos_token_id (int)

        Returns:
            str: The generated dialogue text.
        """
        if gen_turn_by_turn and max_turns is None:
            raise ValueError("max_turns must be specified when gen_turn_by_turn is True.")
        if gen_turn_by_turn and (characters is None or len(characters) < 2):
            raise ValueError("At least two character names must be provided when generating dialogue turn-by-turn.")
        
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

        if self.__is_casual and gen_turn_by_turn:
            raise ValueError("Turn-by-turn generation is not currently supported for causal language models due to the way they handle input and output sequences. Consider using full dialogue generation instead or switch to a seq2seq model.")
        
        if self.__is_casual and "decoder_start_token_id" in generation_kwargs:
            warnings.warn(
                "The 'decoder_start_token_id' parameter is not applicable for causal language models and will be ignored."
            )

        if self.__is_casual and ("forced_eos_token_id" in generation_kwargs or "eos_token_id" in generation_kwargs or "pad_token_id" in generation_kwargs):
            warnings.warn(
                "The 'forced_eos_token_id', 'eos_token_id', and 'pad_token_id' parameters may not work as expected for causal language models. "
                "Ensure that the specified token IDDs are appropriate for your model's tokenizer."
            )

        prompt = narrative

        if self.__is_casual:
            # Use chat templates for causal models to better structure the dialogue
            messages = get_chat_template(narrative=narrative, keep_assistance=False)
            prompt = self.tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True
            )
        
        prompt = prefix_prompt + "\n" + prompt if prefix_prompt else prompt
        
        full_dialogue_output = ""

        def gen_one_output(prompt):
            input_ids = self.tokenizer(
                prompt,
                return_tensors="pt",
                truncation=True,
                max_length=tokenizer_max_length,
            ).to(self.device)

            input_len = input_ids.input_ids.shape[1]
            
            output_ids = self.model.generate(
                **input_ids,
                max_new_tokens=max_new_tokens,
                **generation_kwargs
            )

            generated_text = self.tokenizer.decode(
                output_ids[0][input_len if self.__is_casual else 0:],
                skip_special_tokens=skip_special_tokens
            )
            return generated_text
        
        with torch.no_grad():            
            if gen_turn_by_turn:
                dialogue_history = ""
                for turn_index in range(max_turns):
                    current_character = characters[turn_index % len(characters)]
                    if dialogue_history:
                        prompt = narrative + separator_token + dialogue_history + current_character + ":"
                    else:
                        prompt = narrative + separator_token + current_character + ":"

                    generated_text = gen_one_output(prompt)

                    if skip_special_tokens == False:
                        # When not skipping special tokens, we need to handle the dialogue end token
                        cleaned_turn = generated_text
                    else:
                        # When skipping special tokens, we need to check for dialogue end token manually
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
                full_dialogue_output = gen_one_output(prompt)                

        return full_dialogue_output.strip()
