import sys
import pickle
import torch
import pandas as pd
import copy

import training
import models
import dataset


if __name__ == '__main__':
    # ===== Get model type ===== #
    args = sys.argv
    if len(args) > 1:
        raise ValueError("The model type argument is missing") 
    
    model_type = args[1]
    if model_type not in ["Init", "Meta", "Latent"]:
        raise ValueError("The model type argument can only be one of three options: Init, Meta, or Latent") 

    # ===== Load data ===== #
    with open("../data/embeddings_difference_meta.pickle", "rb") as file:
        btm_matrix = pickle.load(file)

    # ===== Set model parameters ===== #
    size, ratio, ratio_test = len(btm_matrix), 0.7, 0.85
    metadata = False if model_type == "Init" else True
    device="cuda"
    num_features = 300 if model_type == "Init" else 305
    num_epochs = 30
    batch_size = 2048

    # ===== Initialize dataloaders ===== #
    train_set = dataset.BTM_matrix(btm_matrix[:int(size * ratio)], metadata=metadata)
    train_loader = torch.utils.data.DataLoader(
        train_set,
        batch_size=batch_size,
        shuffle=True,
        drop_last=False,
        num_workers=4,
    )

    val_set = dataset.BTM_matrix(btm_matrix[int(size * ratio): int(size * ratio_test)], metadata=metadata)
    val_loader = torch.utils.data.DataLoader(
        val_set,
        batch_size=batch_size,
        shuffle=False,
        num_workers=4,
    )

    test_set = dataset.BTM_matrix(btm_matrix[int(size * ratio_test):], metadata=metadata)
    test_loader = torch.utils.data.DataLoader(
        test_set,
        batch_size=batch_size,
        shuffle=False,
        num_workers=2,
    )

    # ===== Initialize model ===== #
    author_vectors = None
    if model_type == 'Latent':
        with open("data/authors_weights.pickle", "rb") as file:
            author_vectors = pickle.load(file)
        author_vectors = pd.DataFrame.from_dict(author_vectors, orient='index')

    optimizer_kwargs = dict(
        lr=0.001,
        weight_decay=0,
    )

    best_acc, best_lambda_300, best_lambda_meta = 0.0, 0.0, 0.0
    best_model = None
    # ===== Run fine-tune protocol ===== #
    for lambda_300 in [0]:
        for lambda_meta in [0, 0.001, 0.01]:
            print("Regularization params:", lambda_300, lambda_meta)
            print('-' * 50)

            model = models.BTMlatent(num_features, author_vectors) if model_type == "Latent" else models.LogRegression(num_features)
            model, acc = training.run_training(
                model,
                model_type,
                train_loader,
                val_loader,
                num_epochs,
                optimizer_kwargs,
                device=device,
                lambda_300=lambda_300,
                lambda_meta=lambda_meta,
                lambda_latent_300=0,
                lambda_latent_meta=0,
                early_stopping=4,
                verbosity=True,
                plot=False
            )

            if best_acc < acc:
                # save the best current model
                best_acc = acc
                best_lambda_300 = lambda_300
                best_lambda_meta = lambda_meta
                best_model = copy.deepcopy(model.state_dict())

        print('=' * 50)
        print()

    # ===== Get the final result ===== #
    print(f"For BTM-meta model the best accuracy {best_acc * 100:.2f}% for lambdas ({best_lambda_300}, {best_lambda_meta})")
    model = models.BTMlatent(num_features, author_vectors) if model_type == "Latent" else models.LogRegression(num_features)
    model.load_state_dict(best_model)
    model.to(device)
    criterion = torch.nn.BCEWithLogitsLoss()
    training.get_predictions(model, model_type, device, test_loader, criterion, verbosity=True)

    torch.save(model.state_dict(), "../Models/btm-meta-time10%.pth")
