"""
load_data_soda.py

This module contains functions to load and preprocess the SODA dataset for story-to-dialogue conversion tasks.
The dataset is created by Kim et al. (2023) and is available at https://huggingface.co/datasets/allenai/soda.
"""

from datasets import load_dataset, DatasetDict
from config.dir import SODA_HF_REPO

class SODADataLoader:
    """
    Class to load and preprocess the SODA dataset.
    """
    def __init__(
            self,
            data_types: list = ['train', 'test', 'validation'],
            use_features: list = ['narrative', 'dialogue', 'speakers'],
            percent_of_all_splits: float = 1.0,
            random_state: int = 42,
            join_narrative_and_speakers: bool = False,
            join_with: str | None = None,
            show_dataset_info_after_load: bool = True
        ) -> None:
        """
        Initializes the SODADataLoader with specified parameters.

        Args:
            data_types (list): List of dataset splits to load. Options are `train`, `test`, `validation`.
            use_features (list): List of features to retain from the dataset. For all features, use `['all']`.
            percent_of_all_splits (float): Percentage of each split to load (between 0 and 1).
            random_state (int): Random seed for reproducibility.
            join_narrative_and_speakers (bool): If `True`, joins the `narrative` and `speakers` features into a single feature.
            join_with (str | None): String to use for joining `narrative` and `speakers` if `join_narrative_and_speakers` is `True`.
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
        if percent_of_all_splits <= 0 or percent_of_all_splits > 1:
            raise ValueError(
                "percent_of_all_splits must be a float between 0 (exclusive) and 1 (inclusive).")
        if join_narrative_and_speakers and ('speakers' not in use_features or 'narrative' not in use_features or 'all' in use_features):
            raise ValueError(
                "To join narrative and speakers, both 'narrative' and 'speakers' must be in use_features or use_features must be ['all'].")
        if join_narrative_and_speakers and (join_with is None):
            raise ValueError("join_with must be a non-empty string when join_narrative_and_speakers is True.")
             
        self.data_types = data_types
        self.use_features = use_features
        self.percent_of_all_splits = percent_of_all_splits
        self.dataset = self.__load_and_preprocess_data(random_state, join_narrative_and_speakers, join_with)
        if show_dataset_info_after_load:
            self.show_dataset_info(show_features=True)

    def __load_and_preprocess_data(
            self,
            random_state: int = 42,
            join_narrative_and_speakers: bool = False,
            join_with: str | None = None
        ) -> DatasetDict:
        """
        Loads and preprocesses a specific split of the SODA dataset.

        Returns:
            DatasetDict: A dictionary containing the specified splits of the dataset.
        """
        dataset = load_dataset(SODA_HF_REPO)
        processed_splits = {}

        for split in self.data_types:
            if split in dataset:
                split_data = dataset[split]
                if self.percent_of_all_splits < 1.0:
                    split_data = split_data.train_test_split(
                        test_size=self.percent_of_all_splits,
                        seed=random_state
                    )['test']

                if 'all' not in self.use_features:
                    split_data = split_data.remove_columns([col for col in split_data.column_names if col not in self.use_features])
                
                if join_narrative_and_speakers:
                    def join_narrative_speakers(example):
                        example['narrative'] = f"{example['narrative']}{join_with}{example['speakers']}"
                        return example
                    split_data = split_data.map(join_narrative_speakers, remove_columns=['speakers'])

                processed_splits[split] = split_data

        return DatasetDict(processed_splits)

    def show_dataset_info(self, show_features: bool = False) -> None:
        """
        Displays information about the loaded dataset.

        Args:
            show_features (bool): If `True`, displays the features of each split. Default is `False`.
        """
        for split, data in self.dataset.items():
            print(f"Split: {split}")
            print(f"Number of samples: {len(data)}")
            if show_features:
                print(f"Features: {data.column_names}")
            print("-" * 40)
