from torch.utils.data import Dataset

class BTM_matrix(Dataset):
    """Pytorch dataset representing the Bradley-Terry Model matrix for each author pair"""
    def __init__(self, dataset, size=None, metadata=False):
        """Initialize a BTM_matrix object

        Args:
            dataset: tensor shape=(N, D), each row contains the differnece x2 - x1
                where x1, x2 features of paired tweets
            size: None if we take whole dataset, otherwise take first `size` elements;
                useful for plotting learning curves
            metadata: should we also include the metadata
        """
        self.dataset = dataset
        self.metadata = metadata
        self.size = size
    
    def __len__(self):
        """Returns the length of the dataset
        """
        if self.size is None:
            return len(self.dataset)
        else:
            return self.size
   
    def __getitem__(self, index):
        """Get item at row indexed with `index` of the dataset

        Args:
            index: int

        Returns: 
            x: features
            y: label
        """
        if self.metadata:
            x =  self.dataset[index][0]
        else:
            x =  self.dataset[index][0][:300]
        y =  self.dataset[index][1]

        return (x, y)