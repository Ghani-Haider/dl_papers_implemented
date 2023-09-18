from torchvision.datasets import ImageFolder
from torchvision import transforms
from torch.utils.data import DataLoader
from typing import Tuple, List
"""
Contains functionality for creating PyTorch Dataloader for image classification data.
The images should be arranged in this way by default:

    root/dog/xxx.png
    root/dog/xxy.png
    root/dog/[...]/xxz.png

    root/cat/123.png
    root/cat/nsdf3.png
    root/cat/[...]/asd932_.png
"""

def create_dataloaders(train_dir: str,
                       test_dir:str,
                       transform: transforms.Compose,
                       batch_size: int,
                       num_workers: int) -> Tuple[DataLoader, DataLoader, List[str]]:
    """Creates training and testing DataLoaders.

    Takes in a training directory and testing directory path and turns
    them into PyTorch Datasets and then into PyTorch DataLoaders.

    Args:
        train_dir: Path to training directory.
        test_dir: Path to testing directory.
        transform: torchvision transforms to perform on training and testing data.
        batch_size: Number of samples per batch in each of the DataLoaders.
        num_workers: An integer for number of workers per DataLoader.

    Returns:
        A tuple of (train_dataloader, test_dataloader, class_names).
        Where class_names is a list of the target classes.
    
    Example usage:

    train_dataloader, test_dataloader, class_names = \
        = create_dataloaders(train_dir=path/to/train_dir,
                            test_dir=path/to/test_dir,
                            transform=some_transform,
                            batch_size=32,
                            num_workers=4)
    """
    # create datasets
    train_data = ImageFolder(train_dir, transform=transform)
    test_data = ImageFolder(test_dir, transform=transform)

    # get classes
    class_names = train_data.classes

    # create dataloaders
    train_dataloader = DataLoader(train_data,
                                  shuffle=True,
                                  batch_size=batch_size,
                                  num_workers=num_workers,
                                  pin_memory=True)
    test_dataloader = DataLoader(test_data,
                                  shuffle=True,
                                  batch_size=batch_size,
                                  num_workers=num_workers,
                                  pin_memory=True)
    
    return (train_dataloader, test_dataloader, class_names)