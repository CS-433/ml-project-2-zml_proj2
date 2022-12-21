import torch

class LogRegression(torch.nn.Module):
    """Pytorch module for simple Logistic regression used to train Bradley-Terry Model"""
    def __init__(self, input_dim):
        """Initialize a LogRegression object

        Args:
            input_dim: input dimension to the model
        """
        super(LogRegression, self).__init__()
        # learnable parameters of the model
        self.linear = torch.nn.Linear(input_dim, 1, bias=False)
    
    def forward(self, x):
        """Compute the forward pass of the model

        Args:
            x: tensor shape=(B, D)
        *B - batch size

        Returns:
            x @ w^T, tensor shape=(B,)
        """
        return self.linear(x)


class BTMlatent(torch.nn.Module):
    """Pytorch module for latent Bradley-Terry Model"""
    def __init__(self, input_dim, author_vectors):
        """Initialize a BTMlatent object

        Args:
            input_dim: input dimension to the model
            author_vectors: pandas dataframe of average authors' emebeddings
                representing authors' vectors
        """
        super(BTMlatent, self).__init__()
        self.input_dim = input_dim
        self.author_vectors = author_vectors

        # learnable parameters of the model
        self.latentW = torch.nn.Linear(input_dim, input_dim, bias=False)
        self.non_latentW = torch.nn.Linear(input_dim, 1, bias=False)

    def forward(self, x, ids, device="cuda"):
        """Compute the forward pass of the model

        Args:
            x: tensor shape=(B, D)
            ids: list of authors ids shape=(B,)
            device: "cpu" or "cuda"
        *B - batch size
        
        Returns:
            x @ w^T + Diag(author_v @ W @ x^T), tensor shape=(B,)
        """
        # select sub-dataframe mapping authors' ids to their corresponding authors' vectors
        author_v = torch.from_numpy(self.author_vectors.loc[ids].to_numpy()).to(device)
        return self.non_latentW(x) + (author_v * self.latentW(x)).sum(dim=1, keepdim=True)