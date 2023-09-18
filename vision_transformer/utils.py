import torch
from torchvision import transforms
from pathlib import Path
import matplotlib.pyplot as plt
from typing import List, Dict
"""
Contains various utility functions for PyTorch model training and saving.
"""

def accuracy_fn(y_pred: torch.tensor, y_true: torch.tensor) -> float:
    """Calculates the accuracy of a model on given predictions

    Args:
        y_pred: predicted labels
        y_true: true labels
    
    Returns:
        A float value which is the calculated accuracy.
    """
    return ((torch.eq(y_pred, y_true).sum().item() / len(y_true)) * 100)

def save_model(model: torch.nn.Module,
               target_dir: str,
               model_name: str):
  """Saves a PyTorch model to a target directory.

  Args:
    model: A target PyTorch model to save.
    target_dir: A directory for saving the model to.
    model_name: A filename for the saved model. Should include
      either ".pth" or ".pt" as the file extension.

  Example usage:
    save_model(model=model_0,
               target_dir="models",
               model_name="05_going_modular_tingvgg_model.pth")
  """
  # create target directory
  target_dir_path = Path(target_dir)
  target_dir_path.mkdir(parents=True,
                        exist_ok=True)

  # create model save path
  assert model_name.endswith(".pth") or model_name.endswith(".pt"), "model_name should end with '.pt' or '.pth'"
  model_save_path = target_dir_path / model_name

  # save the model state_dict()
  print(f"[INFO] Saving model to: {model_save_path}")
  torch.save(obj=model.state_dict(),
             f=model_save_path)

def load_model(target_dir: str,
               model_name: str,):
  """Loads a state dictionary of PyTorch model from a target directory.

  Args:
    target_dir: A directory for loading the model state_dict from.
    model_name: A filename for the saved model's state_dict. Should include
      either ".pth" or ".pt" as the file extension.

  Example usage:
    load_model(target_dir="models",
               model_name="05_going_modular_tingvgg_model.pth")
  """
  # target directory
  target_dir_path = Path(target_dir)
  if not target_dir_path.is_dir():
     raise FileNotFoundError("Directory does not exist!")

  # model save path
  assert model_name.endswith(".pth") or model_name.endswith(".pt"), "model_name should end with '.pt' or '.pth'"
  model_save_path = target_dir_path / model_name

  if not model_save_path.is_file():
     raise FileNotFoundError("Model does not exist!")

  # load the model state_dict()
  print(f"[INFO] Loading the model's state_dict from: {model_save_path}")

  return torch.load(f=model_save_path)


# simple image transformation
def simple_transform() -> transforms.Compose:
   data_transform = transforms.Compose([
      transforms.Resize(size=(64, 64)),
      transforms.ToTensor()])
   return data_transform

# plot curves
def plot_curves(result) -> None:
    plt.figure(figsize=(10, 6))
    plt.subplot(1, 2, 1)
    plt.plot(result['Train_Loss'], label='train loss')
    plt.plot(result['Test_Loss'], label='test loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Loss Curves')
    plt.legend()
    # plot model acc
    plt.subplot(1, 2, 2)
    plt.plot(result['Train_Acc'], label='train acc')
    plt.plot(result['Test_Acc'], label='test acc')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy (%)')
    plt.title('Accuracy Curves')
    plt.legend()
    plt.tight_layout()
    plt.show()