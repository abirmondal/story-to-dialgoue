"""
inferencing.py

Utility functions for model inferencing.
"""

import torch
import warnings
import threading
from typing import List, Optional, Callable
from tqdm.auto import tqdm
from transformers import (
    AutoModelForSeq2SeqLM,
    AutoModelForCausalLM,
    AutoTokenizer,
    AutoConfig,
    TextIteratorStreamer
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
            merge_lora (bool): Whether to merge the LoRA weights into the base model for faster inference. Defaults to True.
            hf_commit_hash (str | None): Specific commit hash for the HuggingFace model. Defaults to None.
            device (str): Device to load the model on. Can be "auto", "cuda", or "cpu". Defaults to "auto".
            access_token (str | None): HuggingFace access token for private models. Defaults to None.
        """
        if device == "auto":
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        elif device in ["cuda", "cpu"]:
            self.device = device
        else:
            raise ValueError(
                "Invalid device specified. Choose 'auto', 'cuda', or 'cpu'.")

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

        config = AutoConfig.from_pretrained(
            hf_model_repo_name, token=access_token)
        model_arch = config.architectures[0].lower(
        ) if config.architectures else ""

        if "llama" in model_arch or "bloom" in model_arch or "gpt" in model_arch or "opt" in model_arch or "causal" in model_arch:
            model_class = AutoModelForCausalLM
            self.__is_casual = True
            self.tokenizer.padding_side = "left"
            if not self.tokenizer.pad_token:
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

    def _validate_generation_kwargs(self, gen_turn_by_turn: bool, generation_kwargs: dict):
        """
        Validates generation arguments and issues warnings for potential conflicts or unsupported configurations.

        Args:
            gen_turn_by_turn (bool): Whether turn-by-turn generation is enabled.
            generation_kwargs (dict): Dictionary containing arguments passed to the model's generate function.
        """
        if "num_beams" in generation_kwargs:
            warnings.warn(
                "Beam search is not recommended for dialogue generation. Consider using sampling methods instead."
            )
            if "do_sample" in generation_kwargs and generation_kwargs["do_sample"]:
                warnings.warn(
                    "Both 'num_beams' and 'do_sample' are set. Beam search will take precedence."
                )

        if self.__is_casual and gen_turn_by_turn:
            raise ValueError(
                "Turn-by-turn generation is not currently supported for causal language models.")

        if self.__is_casual and "decoder_start_token_id" in generation_kwargs:
            warnings.warn(
                "The 'decoder_start_token_id' parameter is not applicable for causal language models."
            )

        if self.__is_casual and ("forced_eos_token_id" in generation_kwargs or "eos_token_id" in generation_kwargs or "pad_token_id" in generation_kwargs):
            warnings.warn(
                "The 'forced_eos_token_id', 'eos_token_id', and 'pad_token_id' parameters may not work as expected for causal language models."
            )

    def _prepare_prompts(self, narratives: List[str], prefix_prompt: Optional[str]) -> List[str]:
        """
        Prepares raw narratives into prompts suitable for the model, applying chat templates or prefixes if necessary.

        Args:
            narratives (List[str]): A list of input narrative strings.
            prefix_prompt (Optional[str]): An optional prefix string to prepend to each narrative.

        Returns:
            List[str]: A list of processed prompt strings ready for tokenization.
        """
        prompts = []
        for narrative in narratives:
            prompt = narrative
            if self.__is_casual:
                messages = get_chat_template(
                    narrative=narrative, keep_assistance=False)
                prompt = self.tokenizer.apply_chat_template(
                    messages,
                    tokenize=False,
                    add_generation_prompt=True
                )

            if prefix_prompt:
                prompt = prefix_prompt + "\n" + prompt

            prompts.append(prompt)
        return prompts

    def generate_batch_dialogue(
            self,
            narratives: List[str],
            tokenizer_max_length: int = 1024,
            max_new_tokens: int = 512,
            prefix_prompt: str | None = None,
            skip_special_tokens: bool = True,
            preprocess_prompts: bool = True,
            batch_size: int = 8,
            show_progress: bool = True,
            **generation_kwargs
    ) -> List[str]:
        """
        Generates dialogue text for a batch of narratives.

        Args:
            narratives (List[str]): List of input narratives or prompts.
            tokenizer_max_length (int): Maximum length for the tokenizer's input truncation. Defaults to 1024.
            max_new_tokens (int): Maximum number of new tokens to generate. Defaults to 512.
            prefix_prompt (str | None): Optional prefix to prepend to each input text. Defaults to None.
            skip_special_tokens (bool): Whether to remove special tokens (like PAD, EOS) from the output. Defaults to True.
            preprocess_prompts (bool): Whether to apply chat templates and prefixes. Set to False if inputs are already formatted. Defaults to True.
            batch_size (int): Number of inputs to process in a single batch. Defaults to 8.
            show_progress (bool): Whether to display a progress bar (tqdm) during generation. Defaults to True.
            **generation_kwargs: Additional keyword arguments passed to the model's `generate` method (e.g., `do_sample`, `temperature`).

        Returns:
            List[str]: A list of generated dialogue strings corresponding to the input narratives.
        """
        self._validate_generation_kwargs(
            gen_turn_by_turn=False, generation_kwargs=generation_kwargs)

        if preprocess_prompts:
            prompts = self._prepare_prompts(narratives, prefix_prompt)
        else:
            prompts = narratives

        generated_texts = []

        iterator = range(0, len(prompts), batch_size)
        if show_progress and len(prompts) > batch_size:
            iterator = tqdm(iterator, desc="Generating Batches", unit="batch")

        for i in iterator:
            batch_prompts = prompts[i:i + batch_size]

            input_ids = self.tokenizer(
                batch_prompts,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=tokenizer_max_length,
            ).to(self.device)

            with torch.no_grad():
                output_ids = self.model.generate(
                    **input_ids,
                    max_new_tokens=max_new_tokens,
                    **generation_kwargs
                )

            input_len = input_ids.input_ids.shape[1]

            if self.__is_casual:
                generated_tokens = output_ids[:, input_len:]
            else:
                generated_tokens = output_ids

            batch_decoded = self.tokenizer.batch_decode(
                generated_tokens,
                skip_special_tokens=skip_special_tokens
            )
            generated_texts.extend(batch_decoded)

        return generated_texts

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
            use_streaming: bool = False,
            stream_callback: Optional[Callable[[str], None]] = None,
            **generation_kwargs
    ) -> str:
        """
        Generates dialogue text based on a single input narrative, supporting streaming and turn-by-turn generation.

        Args:
            narrative (str): The input narrative text.
            tokenizer_max_length (int): Maximum length for the tokenizer's input truncation. Defaults to 1024.
            max_new_tokens (int): Maximum number of new tokens to generate. Defaults to 512.
            prefix_prompt (str | None): Optional prefix to prepend to the input text. Defaults to None.
            gen_turn_by_turn (bool): Whether to generate dialogue interactively turn-by-turn (Seq2Seq models only). Defaults to False.
            max_turns (int | None): Maximum number of turns to generate if `gen_turn_by_turn` is True. Defaults to None.
            characters (list[str] | None): List of character names for turn-by-turn generation. Required if `gen_turn_by_turn` is True. Defaults to None.
            separator_token (str): Token used to separate dialogue turns in history. Defaults to `DEFAULT_SEPARATOR_TOKEN`.
            skip_special_tokens (bool): Whether to remove special tokens from the output. Defaults to True.
            use_streaming (bool): Whether to yield tokens as they are generated using a streamer. Defaults to False.
            stream_callback (Optional[Callable[[str], None]]): Function called with each generated token string if `use_streaming` is True. Defaults to None.
            **generation_kwargs: Additional keyword arguments passed to the model's `generate` method.

        Returns:
            str: The complete generated dialogue text.
        """
        if gen_turn_by_turn and max_turns is None:
            raise ValueError(
                "max_turns must be specified when gen_turn_by_turn is True.")
        if gen_turn_by_turn and (characters is None or len(characters) < 2):
            raise ValueError(
                "At least two character names must be provided when generating dialogue turn-by-turn.")

        self._validate_generation_kwargs(gen_turn_by_turn, generation_kwargs)

        prompts = self._prepare_prompts([narrative], prefix_prompt)
        final_prompt = prompts[0]

        if use_streaming:
            input_ids = self.tokenizer(
                final_prompt,
                return_tensors="pt",
                truncation=True,
                max_length=tokenizer_max_length,
            ).to(self.device)

            streamer = TextIteratorStreamer(
                self.tokenizer,
                skip_special_tokens=skip_special_tokens,
                skip_prompt=True
            )
            generation_kwargs_with_streamer = {
                **generation_kwargs, "streamer": streamer}

            thread = threading.Thread(
                target=self.model.generate,
                kwargs={
                    **input_ids,
                    "max_new_tokens": max_new_tokens,
                    **generation_kwargs_with_streamer
                }
            )
            thread.start()

            generated_text = ""
            for text in streamer:
                generated_text += text
                if stream_callback:
                    stream_callback(text)

            thread.join()
            return generated_text.strip()

        if gen_turn_by_turn:
            dialogue_history = ""
            full_dialogue_output = ""

            iterator = range(max_turns)
            iterator = tqdm(iterator, desc="Generating Turns", unit="turn")

            for turn_index in iterator:
                current_character = characters[turn_index % len(characters)]
                if dialogue_history:
                    prompt = narrative + separator_token + \
                        dialogue_history + current_character + ":"
                else:
                    prompt = narrative + separator_token + current_character + ":"

                generated_texts = self.generate_batch_dialogue(
                    narratives=[prompt],
                    tokenizer_max_length=tokenizer_max_length,
                    max_new_tokens=max_new_tokens,
                    prefix_prompt=None,
                    skip_special_tokens=skip_special_tokens,
                    preprocess_prompts=False,
                    show_progress=False,
                    **generation_kwargs
                )
                generated_text = generated_texts[0]

                if skip_special_tokens == False:
                    cleaned_turn = generated_text
                else:
                    if DIALOGUE_END_TOKEN in generated_text:
                        cleaned_turn = generated_text.split(
                            DIALOGUE_END_TOKEN)[0].strip()
                        full_dialogue_output += f"{current_character}: " + \
                            cleaned_turn + "\n"
                        break
                    elif self.tokenizer.eos_token and self.tokenizer.eos_token in generated_text:
                        cleaned_turn = generated_text.split(
                            self.tokenizer.eos_token)[0].strip()
                    else:
                        cleaned_turn = generated_text.strip()

                full_dialogue_output += f"{current_character}: " + \
                    cleaned_turn + "\n"
                dialogue_history += f"{current_character}: " + \
                    cleaned_turn + separator_token

            return full_dialogue_output.strip()

        results = self.generate_batch_dialogue(
            narratives=[narrative],
            tokenizer_max_length=tokenizer_max_length,
            max_new_tokens=max_new_tokens,
            prefix_prompt=prefix_prompt,
            skip_special_tokens=skip_special_tokens,
            preprocess_prompts=True,
            show_progress=False,
            **generation_kwargs
        )
        return results[0].strip()
