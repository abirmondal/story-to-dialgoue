"""
load_data_soda.py

This module contains functions to load and preprocess the SODA dataset for story-to-dialogue conversion tasks.
The dataset is created by Kim et al. (2023) and is available at https://huggingface.co/datasets/allenai/soda.
"""

import json
from datasets import load_dataset, DatasetDict
from config.dir import SODA_HF_REPO

class SODADataLoader:
    """
    Class to load and preprocess the SODA dataset.
    """
    def __init__(
            self,
            data_types: list[str] = ['train', 'test', 'validation'],
            use_features: list[str] = ['narrative', 'dialogue', 'speakers'],
            percent_of_all_splits: int | None = None,
            samples_per_split: int | None = None,
            random_state: int = 42,
            join_narrative_and_speakers: bool = False,
            join_with: str | None = None,
            join_dialogue_and_speakers: bool = False,
            add_characters_in_narrative: bool = False,
            add_turns_count_in_narrative: bool = False,
            min_story_length: int | None = None,
            max_story_length: int | None = None,
            show_dataset_info_after_load: bool = True
        ) -> None:
        """
        Initializes the SODADataLoader with specified parameters.

        Args:
            data_types (list): List of dataset splits to load. Options are `train`, `test`, `validation`.
            use_features (list): List of features to retain from the dataset. For all features, use `['all']`.
            percent_of_all_splits (int): Percentage of each split to load (between 0 and 100). Default is `None`, which loads the full splits.
            samples_per_split (int): Number of samples to load per split. If specified, overrides `percent_of_all_splits`. Default is `None`.
            random_state (int): Random seed for reproducibility.
            join_narrative_and_speakers (bool): If `True`, joins the `narrative` and `speakers` features into a single feature.
            join_with (str | None): String to use for joining `narrative` and `speakers` if `join_narrative_and_speakers` is `True`.
            join_dialogue_and_speakers (bool): If `True`, joins the `dialogue` and `speakers` features into a single feature.
            add_characters_in_narrative (bool): If `True`, adds character information to the `narrative` feature.
            add_turns_count_in_narrative (bool): If `True`, adds turn count to the `narrative` feature.
            min_story_length (int | None): Minimum number of words in the `narrative` feature to retain an example. If `None`, no minimum is applied.
            max_story_length (int | None): Maximum number of words in the `narrative` feature to retain an example. If `None`, no maximum is applied.
            show_dataset_info_after_load (bool): If `True`, displays dataset information including feature details after loading. Default is `True`.
        """
        if data_types is None or len(data_types) == 0:
            raise ValueError("data_types must be a non-empty list containing any of 'train', 'test', 'validation'.")
        # Check if data_types value is valid
        valid_splits = {'train', 'test', 'validation'}
        for split in data_types:
            if split not in valid_splits:
                raise ValueError(f"Invalid data_type '{split}'. Valid options are 'train', 'test', 'validation'.")
        if use_features is None or len(use_features) == 0:
            raise ValueError("use_features must be a non-empty list of feature names or ['all'].")
        if 'all' in use_features and len(use_features) > 1:
            raise ValueError("If 'all' is specified in use_features, it must be the only entry.")
        if percent_of_all_splits is not None:
            if percent_of_all_splits <= 0 or percent_of_all_splits > 100:
                raise ValueError("percent_of_all_splits must be between 1 and 100.")
        if join_narrative_and_speakers and ('speakers' not in use_features or 'narrative' not in use_features or 'all' in use_features):
            raise ValueError(
                "To join narrative and speakers, both 'narrative' and 'speakers' must be in use_features or use_features must be ['all'].")
        if join_narrative_and_speakers and (join_with is None):
            raise ValueError("join_with must be a non-empty string when join_narrative_and_speakers is True.")
        if join_dialogue_and_speakers and ('speakers' not in use_features or 'dialogue' not in use_features or 'all' in use_features):
            raise ValueError(
                "To join dialogue and speakers, both 'dialogue' and 'speakers' must be in use_features or use_features must be ['all'].")
        if join_dialogue_and_speakers and join_narrative_and_speakers:
            raise ValueError("Only one of join_narrative_and_speakers or join_dialogue_and_speakers can be True.")
             
        # validate story length bounds
        if min_story_length is not None and min_story_length <= 0:
            raise ValueError("min_story_length must be a positive integer or None")
        if max_story_length is not None and max_story_length <= 0:
            raise ValueError("max_story_length must be a positive integer or None")
        if min_story_length is not None and max_story_length is not None and min_story_length > max_story_length:
            raise ValueError("min_story_length cannot be greater than max_story_length")
        # store small config params centrally (avoid setting as many self.* attributes)
        self.dataset_info: dict = {
            'params': {
                'data_types': data_types,
                'use_features': use_features,
                'percent_of_all_splits': percent_of_all_splits,
                'samples_per_split': samples_per_split,
                'join_narrative_and_speakers': join_narrative_and_speakers,
                'join_dialogue_and_speakers': join_dialogue_and_speakers,
                'add_characters_in_narrative': add_characters_in_narrative,
                'add_turns_count_in_narrative': add_turns_count_in_narrative,
                'min_story_length': min_story_length,
                'max_story_length': max_story_length,
                'join_with': join_with,
                'random_state': random_state,
            },
            'splits': {}
        }

        # load, filter and preprocess using local params and dataset_info
        dataset = self.__load_data(splits=data_types, features=use_features, percent_of_all_splits=percent_of_all_splits, samples_per_split=samples_per_split, random_state=random_state)
        dataset = self.__filter_by_story_length(dataset, min_story_length=min_story_length, max_story_length=max_story_length)
        self.dataset = self.__preprocess_data(
            dataset=dataset,
            join_narrative_and_speakers=join_narrative_and_speakers,
            join_with=join_with,
            join_dialogue_and_speakers=join_dialogue_and_speakers,
            add_characters_in_narrative=add_characters_in_narrative,
            add_turns_count_in_narrative=add_turns_count_in_narrative
        )

        # populate minimal metadata (counts, columns). heavy stats are computed lazily on demand
        self._populate_dataset_info()
        if show_dataset_info_after_load:
            self.show_dataset_info(show_features=True)

    def __load_data(self, splits: list[str], features: list[str], percent_of_all_splits: int = 100, samples_per_split: int | None = None) -> DatasetDict:
        """
        Loads the SODA dataset from the Hugging Face repository.

        Args:
            splits (list): List of dataset splits to load. Options are `train`, `test`, `validation`.
            features (list): List of features to retain from the dataset. For all features, use `['all']`.
            percent_of_all_splits (int): Percentage of each split to load (between 0 and 100). Default is 100 (load full splits).
            samples_per_split (int | None): Number of samples to load per split. If specified, overrides `percent_of_all_splits`. Default is None.

        Returns:
            DatasetDict: A dictionary containing the specified splits of the dataset.
        """
        dataset = {}
        for split in splits:
            if samples_per_split is not None:
                split_str = f"[:{samples_per_split}]"
            elif percent_of_all_splits is not None:
                split_str = f"[:{percent_of_all_splits}%]"
            else:
                split_str = ""
            dataset[split] = load_dataset(SODA_HF_REPO, split=f"{split}{split_str}")
        dataset = DatasetDict(dataset)
        ds_keys = list(dataset.keys())

        for split in ds_keys:
            if 'all' not in features:
                dataset[split] = dataset[split].remove_columns([col for col in dataset[split].column_names if col not in features])

        return dataset

    def __preprocess_data(
            self,
            dataset: DatasetDict,
            join_narrative_and_speakers: bool = False,
            join_with: str | None = None,
            join_dialogue_and_speakers: bool = False,
            add_characters_in_narrative: bool = False,
            add_turns_count_in_narrative: bool = False
        ) -> DatasetDict:
        """
        Preprocesses the SODA dataset by selecting specified splits and features, and optionally joining features.

        Returns:
            DatasetDict: A dictionary containing the specified splits of the dataset.
        """
        processed_splits = {}

        for split in self.dataset_info['params']['data_types']:
            if split in dataset:
                split_data = dataset[split]
                
                if join_narrative_and_speakers:
                    def join_narrative_speakers(example):
                        example['narrative'] = f"{example['narrative']}{join_with}{example['speakers']}"
                        return example
                    split_data = split_data.map(join_narrative_speakers, remove_columns=['speakers'], desc=f"Joining narrative and speakers for {split} split")
                
                if add_characters_in_narrative:
                    def add_characters(example):
                        characters = set(example['speakers'])
                        characters_str = "Characters: " + ", ".join(characters) + ". "
                        example['narrative'] = example['narrative'] + "\n" + characters_str
                        return example
                    split_data = split_data.map(add_characters, desc=f"Adding characters to narrative for {split} split")

                if add_turns_count_in_narrative:
                    def add_turns_count(example):
                        num_turns = len(example['dialogue'])
                        turns_str = f"Dialogue turns: {num_turns}. "
                        example['narrative'] = example['narrative'] + "\n" + turns_str
                        return example
                    split_data = split_data.map(add_turns_count, desc=f"Adding turn count to narrative for {split} split")

                if join_dialogue_and_speakers:
                    def join_dialogue_speakers(example):
                        # create a single string where each utterance is prefixed by its speaker
                        joined_lines = []
                        for utterance, speaker in zip(example['dialogue'], example['speakers']):
                            joined_lines.append(f"{speaker}: {utterance}")
                        # convert to a single string (separated by newlines) so it can be passed to models
                        example['dialogue'] = "\n".join(joined_lines)
                        return example
                    split_data = split_data.map(join_dialogue_speakers, remove_columns=['speakers'], desc=f"Joining dialogue and speakers for {split} split")

                processed_splits[split] = split_data

        return DatasetDict(processed_splits)

    def __get_num_words_in_story_batch(self, batch: list) -> list:
        """
        Computes the number of words in stories in a given batch.

        Args:
            batch (list): A batch of stories.
        
        Returns:
            list: A list of word counts for each story in the batch.
        """
        if not batch:
            return {"story_word_count": []}

        return {"story_word_count": [len(entry.split()) for entry in batch['narrative']]}

    def __get_num_words_in_dialogue_batch(self, batch: list) -> list:
        """
        Computes the number of words in dialogues in a given batch.

        Args:
            batch (list): A batch of dialogues.

        Returns:
            list: A list of word counts for each dialogue in the batch.
        """
        if not batch:
            return {"dialogue_word_count": []}

        counts = []
        for entry in batch['dialogue']:
            # if dialogues were joined with speakers they will be a single string
            if isinstance(entry, str):
                counts.append(len(entry.split()))
            else:
                # otherwise expect a list of utterances
                counts.append(len(" ".join(entry).split()))

        return {"dialogue_word_count": counts}

    def __filter_by_story_length(self, dataset: DatasetDict, min_story_length: int | None = None, max_story_length: int | None = None) -> DatasetDict:
        """
        Filters examples in the dataset splits based on the number of words in the `narrative` feature.

        Args:
            dataset (DatasetDict): The dataset to filter.
            min_story_length (int | None): Minimum number of words (inclusive). If None, no minimum applied.
            max_story_length (int | None): Maximum number of words (inclusive). If None, no maximum applied.

        Returns:
            DatasetDict: A new DatasetDict containing only examples within the specified bounds.
        """
        # If neither bound is provided, return dataset unchanged
        if min_story_length is None and max_story_length is None:
            return dataset

        processed = {}

        def _within_bounds(example):
            # If narrative isn't present, exclude the example
            if 'narrative' not in example or example['narrative'] is None:
                return False
            count = len(example['narrative'].split())
            if min_story_length is not None and count < min_story_length:
                return False
            if max_story_length is not None and count > max_story_length:
                return False
            return True

        for split, data in dataset.items():
            # Only attempt to filter splits that have 'narrative'
            if 'narrative' in data.column_names:
                filtered = data.filter(lambda example: _within_bounds(example), desc=f"Filtering {split} split by story length")
                processed[split] = filtered
            else:
                # keep as-is when no narrative field to evaluate
                processed[split] = data

        return DatasetDict(processed)

    def _populate_dataset_info(self) -> None:
        """
        Populate small, cheap-to-compute metadata for each split and store it in `self.dataset_info['splits']`.

        This function deliberately avoids computing heavy statistics (like per-example word counts).
        Those are computed lazily by `_get_story_stats` / `_get_dialogue_stats` when requested.
        """
        first_columns_set = False
        for split, data in self.dataset.items():
            # store per-split sample counts only; columns are expected to be identical across splits
            self.dataset_info['splits'][split] = {'num_samples': len(data)}
            if not first_columns_set:
                self.dataset_info['columns'] = list(data.column_names)
                first_columns_set = True

        # initialize caches for lazy stats
        self._story_stats_cache: dict[str, dict] = {}
        self._dialogue_stats_cache: dict[str, dict] = {}

    def _get_story_stats(self, split: str) -> dict | None:
        """Compute (and cache) min/max story word counts for a split. Returns dict or None if not applicable."""
        if split in self._story_stats_cache:
            return self._story_stats_cache[split]
        if split not in self.dataset:
            return None
        data = self.dataset[split]
        if 'narrative' not in data.column_names:
            return None

        min_c = None
        max_c = None
        # stream batches to avoid building a large list in memory
        for batch in data.iter(batch_size=1000):
            narratives = batch.get('narrative', [])
            counts = [len(n.split()) if n else 0 for n in narratives]
            if counts:
                bmin, bmax = min(counts), max(counts)
                min_c = bmin if min_c is None else min(min_c, bmin)
                max_c = bmax if max_c is None else max(max_c, bmax)

        stats = {'min': min_c, 'max': max_c}
        self._story_stats_cache[split] = stats
        return stats

    def _get_dialogue_stats(self, split: str) -> dict | None:
        """Compute (and cache) min/max dialogue word counts for a split. Returns dict or None if not applicable."""
        if split in self._dialogue_stats_cache:
            return self._dialogue_stats_cache[split]
        if split not in self.dataset:
            return None
        data = self.dataset[split]
        if 'dialogue' not in data.column_names:
            return None

        min_c = None
        max_c = None
        for batch in data.iter(batch_size=1000):
            dialogues = batch.get('dialogue', [])
            counts = []
            for entry in dialogues:
                if isinstance(entry, str):
                    counts.append(len(entry.split()))
                else:
                    counts.append(len(" ".join(entry).split()))
            if counts:
                bmin, bmax = min(counts), max(counts)
                min_c = bmin if min_c is None else min(min_c, bmin)
                max_c = bmax if max_c is None else max(max_c, bmax)

        stats = {'min': min_c, 'max': max_c}
        self._dialogue_stats_cache[split] = stats
        return stats

    def get_dataset_info(self, flat: bool = True, include_word_counts: bool = False) -> dict:
        """
        Return dataset_info as a single-level (flat) dictionary suitable for logging.

        Args:
            flat (bool): If True, return a flattened dict where nested keys are joined with dots.
                         If False, return the original nested `self.dataset_info` dict.

        Returns:
            dict: The dataset info dictionary (flattened if requested).
        """
        if not flat:
            return self.dataset_info

        flat_info: dict = {}
        # params: place keys at top level (no 'params.' prefix)
        params = self.dataset_info.get('params', {})
        for k, v in params.items():
            flat_info[k] = v

        # columns (single entry) - return as a list of strings
        cols = self.dataset_info.get('columns')
        if cols is not None:
            flat_info['columns'] = list(map(str, cols))

        # per-split sample counts (keys: num_samples.<split>)
        splits = self.dataset_info.get('splits', {})
        for split_name, split_info in splits.items():
            # use prefix 'split_name/' for keys, e.g. 'train/num_samples'
            if isinstance(split_info, dict) and 'num_samples' in split_info:
                flat_info[f"{split_name}/num_samples"] = split_info['num_samples']
            else:
                flat_info[f"{split_name}/num_samples"] = split_info

        # optionally include min/max word counts (computed lazily)
        if include_word_counts:
            for split_name in splits.keys():
                # story
                s = self._get_story_stats(split_name)
                if s is not None:
                    flat_info[f"{split_name}/story_min"] = s.get('min')
                    flat_info[f"{split_name}/story_max"] = s.get('max')
                # dialogue
                d = self._get_dialogue_stats(split_name)
                if d is not None:
                    flat_info[f"{split_name}/dialogue_min"] = d.get('min')
                    flat_info[f"{split_name}/dialogue_max"] = d.get('max')

        return flat_info

    def show_dataset_info(self, show_features: bool = False, show_word_counts: bool = False, show_dataset_info_details: bool = False) -> None:
        """
        Displays information about the loaded dataset.

        Args:
            show_features (bool): If `True`, displays the features of each split. Default is `False`.
            show_word_counts (bool): If `True`, computes and displays the minimum and maximum
                word counts for `narrative` and `dialogue` features in each split. Default is `False`.
        """
        if show_dataset_info_details:
            flat = self.get_dataset_info(flat=True)
            print("Dataset info (flattened):")
            print(json.dumps(flat, indent=2))
            print("-" * 40)

        for split, data in self.dataset.items():
            print(f"Split: {split}")
            # print per-split sample count using 'split_name/num_samples' format
            print(f"{split}/num_samples: {len(data)}")
            if show_features:
                print(f"Features: {data.column_names}")
            if show_word_counts:
                # use lazy cached stats to avoid remapping the dataset repeatedly
                if 'narrative' in data.column_names:
                    s_stats = self._get_story_stats(split)
                    if s_stats is not None:
                        print(f"Minimum story word count: {s_stats['min']}")
                        print(f"Maximum story word count: {s_stats['max']}")
                if 'dialogue' in data.column_names:
                    d_stats = self._get_dialogue_stats(split)
                    if d_stats is not None:
                        print(f"Minimum dialogue word count: {d_stats['min']}")
                        print(f"Maximum dialogue word count: {d_stats['max']}")
            print("-" * 40)
