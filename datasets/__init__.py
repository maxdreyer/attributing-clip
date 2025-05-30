from typing import Callable

from datasets.imagenet import get_imagenet

DATASETS = {
    "imagenet": get_imagenet,
}


def get_dataset(dataset_name: str) -> Callable:
    """
    Get dataset by name.
    :param dataset_name: Name of the dataset.
    :return: Dataset.

    """
    if dataset_name in DATASETS:
        def decorated_getter(*args, **kwargs):
            dataset = DATASETS[dataset_name](*args, **kwargs)
            setattr(dataset, "dataset_name", dataset_name)
            return dataset

        dataset_getter = decorated_getter
        print(f"Loading {dataset_name}")
        return dataset_getter  # type: ignore
    else:
        raise KeyError(f"DATASET {dataset_name} not defined.")
