# Code referenced from: EPFL, CS-433 (https://github.com/epfml/ML_course)

import numpy as np
from tqdm.notebook import tqdm
import matplotlib.pyplot as plt
import copy
import torch


def regularization_init(model, lambda_300, lambda_meta):
    """Compute l2 regularization for the initial and meta types of models
    
    Args:
        model: torch.nn.Module representing model
        lambda_300: regularization parameter for feature emeddings
        lambda_meta: regularization parameter for meta features

    Returns:
        reg: l2 regularization
    """
    l2_norm_300, l2_norm_meta = 0, 0
    for p in model.parameters():
        l2_norm_300 += p[:, :300].pow(2).sum()
        if p.shape[1] > 300:
            l2_norm_meta += p[:, 300:].pow(2).sum()

    reg = lambda_300 * l2_norm_300 + lambda_meta * l2_norm_meta
    return reg


def regularization_latent(model, lambda_300, lambda_meta, lambda_latent_300, lambda_latent_meta):
    """Compute l2 regularization for the latent model
    
    Args:
        model: torch.nn.Module representing model
        lambda_300: regularization parameter for feature emeddings
        lambda_meta: regularization parameter for meta features
        lambda_latent_300: regularization parameter for latent feature emeddings
        lambda_latent_meta: regularization parameter for latent meta features

    Returns:
        reg: l2 regularization
    """
    l2_norm_300, l2_norm_meta = 0, 0
    l2_latent_300, l2_norm_latent_meta = 0, 0
    for p in model.parameters():
        if p.shape[0] > 1:
            l2_latent_300 += p[:, :300].pow(2).sum()
            l2_norm_latent_meta += p[:, 300:].pow(2).sum()
        else:
            l2_norm_300 += p[:, :300].pow(2).sum()
            l2_norm_meta += p[:, 300:].pow(2).sum()

    return l2_norm_300 * lambda_300 + l2_norm_meta * lambda_meta +\
         l2_latent_300 * lambda_latent_300 + l2_norm_latent_meta * lambda_latent_meta


def train_epoch(
    model,
    model_type,
    optimizer, 
    criterion, 
    train_loader, 
    epoch, 
    device, 
    lambda_300=0, 
    lambda_meta=0, 
    lambda_latent_300=0, 
    lambda_latent_meta=0, 
    verbosity=True 
):
    """Run the training procedure of the provided model for a single epoch.

    Args:
        model: torch.nn.Module representing model
        model_type: type of the model to train: "Init", "Meta", "Latent"
        num_epochs: number of epochs to run the training for
        optimizer: pytorch optimizer
        criterion: pytorch criterion
        train_loader: pytorch DataLoader for training data
        device: device type: "cuda", "cpu"
        lambda_300: regularization parameter for feature emeddings
        lambda_meta: regularization parameter for meta features
        lambda_latent_300: regularization parameter for latent feature emeddings
        lambda_latent_meta: regularization parameter for latent meta features
        verbosity: should model print out the intermidiate results

    Returns:
        train_loss: average loss of the model 
        accuracy: accuracy of the model
    """
    model.train()
    correct = 0
    train_loss = 0
    for data, target in train_loader:
        data = data.to(device)
        
        optimizer.zero_grad()
        if model_type == "Latent":
            ids = target[:, 1]
            output = model(data, ids=ids, device=device).squeeze(1)
        else:
            output = model(data).squeeze(1)
        
        if model_type == "Latent":
            reg = regularization_latent(
                model, lambda_300, lambda_meta, lambda_latent_300, lambda_latent_meta)
        else:
            reg = regularization_init(model, lambda_300, lambda_meta)
        
        target = target[:, 0].type(torch.DoubleTensor).to(device)
        loss = criterion(output, target)
        (loss + reg).backward()
        optimizer.step()

        pred = (output >= 0).type(torch.DoubleTensor).to(device)
        correct += pred.eq(target.view_as(pred)).sum().item()
        train_loss += loss.item() * len(data)
    
    train_loss /= len(train_loader.dataset)
    accuracy = correct / len(train_loader.dataset)
    if verbosity:
        print(
            f"Train Epoch: {epoch} Average loss: {train_loss:.4f}, Accuracy: {correct}/{len(train_loader.dataset)} ({100.0 * accuracy:.2f}%)")

    return train_loss, accuracy


@torch.no_grad()
def validate(model, model_type, device, val_loader, criterion, verbosity=True):
    """Validate the results of the provided model.

    Args:
        model: torch.nn.Module representing model
        model_type: type of the model to train: "Init", "Meta", "Latent"
        device: device type: "cuda", "cpu"
        val_loader: pytorch DataLoader for validation data
        criterion: pytorch criterion
        verbosity: should model print out the intermidiate results

    Returns:
        test_loss: average loss of the model
        accuracy: accuracy of the model
    """
    model.eval()
    test_loss = 0
    correct = 0
    for data, target in val_loader:
        data = data.to(device)
        if model_type == "Latent":
            ids = target[:, 1]
            output = model(data, ids=ids, device=device).squeeze(1)
        else:
            output = model(data).squeeze(1)
        
        target = target[:, 0].type(torch.DoubleTensor).to(device)
        test_loss += criterion(output, target).item() * len(data)
        pred = (output >= 0).type(torch.DoubleTensor).to(device)

        correct += pred.eq(target.view_as(pred)).sum().item()

    test_loss /= len(val_loader.dataset)
    accuracy = correct / len(val_loader.dataset)
    if verbosity:
        print(
            "Validation set: Average loss: {:.4f}, Accuracy: {}/{} ({:.2f}%)".format(
                test_loss,
                correct,
                len(val_loader.dataset),
                accuracy * 100,
            )
        )

    return test_loss, accuracy


@torch.no_grad()
def get_predictions(model, model_type, device, test_loader, criterion, verbosity=True):
    """Validate the results of the provided model.

    Args:
        model: torch.nn.Module representing model
        model_type: type of the model to train: "Init", "Meta", "Latent"
        device: device type: "cuda", "cpu"
        val_loader: pytorch DataLoader for validation data
        criterion: pytorch criterion
        verbosity: should model print out the intermidiate results

    Returns:
        test_loss: average loss of the model
        accuracy: accuracy of the model
    """
    model.eval()
    test_loss = 0
    correct = 0
    for data, target in test_loader:
        data = data.to(device)
        if model_type == "Latent":
            ids = target[:, 1]
            output = model(data, ids=ids, device=device).squeeze(1)
        else:
            output = model(data).squeeze(1)

        target = target[:, 0].type(torch.DoubleTensor).to(device)
        test_loss += criterion(output, target).item() * len(data)
        pred = (output >= 0).type(torch.DoubleTensor).to(device)

        correct += pred.eq(target.view_as(pred)).sum().item()

    test_loss /= len(test_loader.dataset)
    accuracy = 100.0 * correct / len(test_loader.dataset)
    if verbosity:
        print(
            "Test set: Average loss: {:.4f}, Accuracy: {}/{} ({:.2f}%)".format(
                test_loss,
                correct,
                len(test_loader.dataset),
                accuracy,
            )
        )
    
    return accuracy


def run_training(
    model,
    model_type,
    train_loader,
    val_loader,
    num_epochs,
    optimizer_kwargs,
    device="cuda",
    lambda_300=0, 
    lambda_meta=0,
    lambda_latent_300=0,
    lambda_latent_meta=0,
    early_stopping=None,
    verbosity=True,
    plot=True
):
    """Run the training procedure of the provided model for the given number of epochs.

    Args:
        model: torch.nn.Module representing model
        model_type: type of the model to train: "Init", "Meta", "Latent"
        train_loader: pytorch DataLoader for training data
        val_loader: pytorch DataLoader for validation data
        num_epochs: number of epochs to run the training for
        optimizer_kwargs: params for pytorch optimizer
        device: device type: "cuda", "cpu"
        lambda_300: regularization parameter for feature emeddings
        lambda_meta: regularization parameter for meta features
        lambda_latent_300: regularization parameter for latent feature emeddings
        lambda_latent_meta: regularization parameter for latent meta features
        early_stopping: None if no early_stopping, otherwise set the number of epochs 
            for which to tolerate non-improvement of the model
        verbosity: should model print out the intermidiate results
        plot: should model plot the learning curves

    Returns:
        model: best trained model
        best_acc: best accuracy on validation set
    """
    # ===== Model, Optimizer and Criterion ===== #
    model = model.to(device=device)
    optimizer = torch.optim.AdamW(model.parameters(), **optimizer_kwargs)
    criterion = torch.nn.BCEWithLogitsLoss()
    tolerance = early_stopping
    
    # ===== Save Results ===== #
    best_model_wts = copy.deepcopy(model.state_dict())
    best_acc = 0.0 

    # ===== Train Model ===== #
    train_loss_history = []
    train_acc_history = []
    val_loss_history = []
    val_acc_history = []
    for epoch in tqdm(range(1, num_epochs + 1)):
        train_loss, train_acc = train_epoch(
            model, model_type, optimizer, criterion, train_loader, epoch, device, 
            lambda_300, lambda_meta, lambda_latent_300, lambda_latent_meta, verbosity)
        train_loss_history.append(train_loss)
        train_acc_history.append(train_acc)

        val_loss, val_acc = validate(
            model, model_type, device, val_loader, criterion, verbosity)
        val_loss_history.append(val_loss)
        val_acc_history.append(val_acc)
        
        if best_acc < val_acc:
            # save the best model and its result
            best_acc = val_acc
            best_model_wts = copy.deepcopy(model.state_dict())
            tolerance = early_stopping
        elif early_stopping is not None:
            # if earliy stopping is requested decrease tolearance as we didn't get the best model
            tolerance -= 1
            if tolerance < 0:
                # if tolerance is exceeded without model improvements, end training
                break

    # ===== Plot training curves ===== #
    if plot:
        t_train = np.arange(1, len(train_acc_history) + 1)
        t_val = np.arange(1, len(val_acc_history) + 1)

        plt.figure(figsize=(6.4 * 3, 4.8))
        plt.subplot(1, 3, 1)
        plt.plot(t_train, train_acc_history, label="Train")
        plt.plot(t_val, val_acc_history, label="Val")
        plt.legend()
        plt.title(f"Learning curves lambda_300={lambda_300} and lambda_meta={lambda_meta}")
        plt.xlabel("Epoch")
        plt.ylabel("Accuracy")

        plt.subplot(1, 3, 2)
        plt.plot(t_train, train_loss_history, label="Train")
        plt.plot(t_val, val_loss_history, label="Val")
        plt.legend()
        plt.title(f"Learning curves lambda_300={lambda_300} and lambda_meta={lambda_meta}")
        plt.xlabel("Epoch")
        plt.ylabel("Loss")
    
    # ===== Best model result =====
    if verbosity:
        print('*' * 50)
        print(f"Model lambda_300={lambda_300} and lambda_meta={lambda_meta}, the best validation accuracy: {best_acc * 100:.2f}%")
        print('*' * 50)

    model.load_state_dict(best_model_wts)
    return model, best_acc