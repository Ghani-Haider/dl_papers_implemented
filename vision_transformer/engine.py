from data_setup import create_dataloaders
from utils import accuracy_fn
import torch
from typing import Tuple, Dict, List
from tqdm.auto import tqdm
"""
Contains functions for training, testing and evaluting a model.
"""
# fit the model on training data
def train_step(model: torch.nn.Module,
               data_loader: torch.utils.data.DataLoader,
               loss_fn: torch.nn.Module,
               optimizer: torch.optim.Optimizer,
               device: torch.device) -> Tuple[float, float]:
    """Trains a PyTorch model for a single epoch.

    Turns a target PyTorch model to training mode and then
    runs through all of the required training steps (forward
    pass, loss calculation, optimizer step).

    Args:
        model: A PyTorch model to be trained.
        dataloader: A DataLoader instance for the model to be trained on.
        loss_fn: A PyTorch loss function to minimize.
        optimizer: A PyTorch optimizer to help minimize the loss function.
        device: A target device to compute on (e.g. "cuda" or "cpu").

    Returns:
        A tuple of training loss and training accuracy metrics.
        In the form (train_loss, train_accuracy). For example:

        (0.1112, 0.8743)
    """
    train_loss, train_acc = 0, 0

    model.train() # model in train mode
    for X, y in data_loader:
        # get data to device
        X = X.to(device)
        y = y.to(device)

        # forward pass
        y_logit = model(X)
        loss = loss_fn(y_logit, y)        

        # backward pass
        optimizer.zero_grad() # empty param's grad
        loss.backward() # backward propagation
        optimizer.step() # updata params (take the gradient descent step)

        train_loss += loss.item()
        y_pred_labels = torch.argmax(y_logit, dim=1)
        train_acc += accuracy_fn(y_pred_labels, y)

    train_loss /= len(data_loader)
    train_acc /= len(data_loader)
    return (train_loss, train_acc)

# test the model on test data
def test_step(model: torch.nn.Module,
              data_loader: torch.utils.data.DataLoader,
              loss_fn: torch.nn.Module,
              device: torch.device) -> Tuple[float, float]:
    """Tests a PyTorch model for a single epoch.

    Turns a target PyTorch model to "eval" mode and then performs
    a forward pass on a testing dataset.

    Args:
        model: A PyTorch model to be tested.
        data_loader: A DataLoader instance for the model to be tested on.
        loss_fn: A PyTorch loss function to calculate loss on the test data.
        device: A target device to compute on (e.g. "cuda" or "cpu").

    Returns:
        A tuple of testing loss and testing accuracy metrics.
        In the form (test_loss, test_accuracy). For example:

        (0.0223, 0.8985)
    """

    test_loss , test_acc = 0, 0

    model.eval() # model in evaluation mode
    with torch.inference_mode():
        for X, y in data_loader:

            # get data to device
            X = X.to(device)
            y = y.to(device)
            
            # forward pss
            y_logit = model(X)
            loss = loss_fn(y_logit, y)

            # calculate loss and accuracy per batch
            test_loss += loss.item()
            y_pred_labels = torch.argmax(y_logit, dim=1)
            test_acc += accuracy_fn(y_pred_labels, y)

    test_loss /= len(data_loader)
    test_acc /= len(data_loader)
    return (test_loss, test_acc)

# complete training function for given epochs
def train(model: torch.nn.Module,
          train_dl: torch.utils.data.DataLoader,
          test_dl: torch.utils.data.DataLoader,
          loss_fn: torch.nn.Module,
          optimizer: torch.optim.Optimizer,
          device: torch.device,
          epochs) -> Dict[str, List]:
    """Trains and tests a PyTorch model.

    Passes a target PyTorch models through train_step() and test_step()
    functions for a number of epochs, training and testing the model
    in the same epoch loop.

    Calculates, prints and stores evaluation metrics throughout.

    Args:
        model: A PyTorch model to be trained and tested.
        train_dl: A DataLoader instance for the model to be trained on.
        test_dl: A DataLoader instance for the model to be tested on.
        loss_fn: A PyTorch loss function to calculate loss on both datasets.
        optimizer: A PyTorch optimizer to help minimize the loss function.
        device: A target device to compute on (e.g. "cuda" or "cpu").
        epochs: An integer indicating how many epochs to train for.

    Returns:
        A dictionary of training and testing loss as well as training and
        testing accuracy metrics. Each metric has a value in a list for 
        each epoch.
        In the form: {train_loss: [...],
                    train_acc: [...],
                    test_loss: [...],
                    test_acc: [...]} 
        For example if training for epochs=2: 
                    {train_loss: [2.0616, 1.0537],
                    train_acc: [0.3945, 0.3945],
                    test_loss: [1.2641, 1.5706],
                    test_acc: [0.3400, 0.2973]} 
    """
    
    result = {'Train_Loss': [],
            'Train_Acc': [],
            'Test_Loss': [],
            'Test_Acc': [],
            }

    for epoch in tqdm(range(epochs)):
        trainlss, trainacc = train_step(model, train_dl, loss_fn, optimizer, device)

        testlss, testacc = test_step(model, test_dl, loss_fn, device)

        print(f"Epoch {epoch} | Train Loss {trainlss:.4f} | Train Acc {trainacc:.2f} | Test Loss {testlss:.4f} | Test Acc {testacc:.2f}")

        result['Train_Loss'].append(trainlss)
        result['Train_Acc'].append(trainacc)
        result['Test_Loss'].append(testlss)
        result['Test_Acc'].append(testacc)
    
    return result

def eval(model: torch.nn.Module,
         data_loader: torch.utils.data.DataLoader,
         loss_fn: torch.nn.Module,
         device: torch.device) -> None:
    """
    function to evaluate the model on testing data

    Args:
        model (nn.Module): Model to fit the data.
        data_loader (DataLoader): Iterable over the dataset.
        loss_fn (nn.CrossEntropyLoss): Loss function.
        accuracy_fn (accuracy_fn): Optimizer for gradient update.
    Returns:
        None
    """

    eval_loss , eval_acc = 0, 0

    model.eval()
    with torch.inference_mode():
        for X, y in data_loader:

            # get data to device
            X = X.to(device)
            y = y.to(device)

            # forward pass
            y_logit = model(X)
            loss = loss_fn(y_logit, y)

            # calculate loss and accuracy per batch
            eval_loss += loss
            y_pred_labels = torch.argmax(y_logit, dim=1)
            eval_acc += accuracy_fn(y_pred_labels, y)

    eval_loss /= len(data_loader)
    eval_acc /= len(data_loader)
    
    result = {'model name': model.__class__.__name__,
              'model loss': round(eval_loss.item(), ndigits=3),
              'model acc': round(eval_acc, ndigits=3)}
    return result