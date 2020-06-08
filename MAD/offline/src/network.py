import torch
from torch import nn
import torch.nn.functional as F


class ProductRecommendationModel(nn.Module):
    """
    Defines the neural network for product recommendation
    """

    def __init__(self, embedding_sizes, n_cont, n_classes=3):
        super().__init__()
        self.embeddings = nn.ModuleList([nn.Embedding(categories, size) for
                                         categories, size in embedding_sizes])
        n_emb = sum(e.embedding_dim for e in self.embeddings)
        self.n_emb, self.n_cont, self.n_classes = n_emb, n_cont, n_classes
        self.lin1 = nn.Linear(self.n_emb + self.n_cont, 300)
        self.lin2 = nn.Linear(300, 100)
        self.lin3 = nn.Linear(100, self.n_classes)
        self.bn1 = nn.BatchNorm1d(self.n_cont)
        self.bn2 = nn.BatchNorm1d(300)
        self.bn3 = nn.BatchNorm1d(100)
        self.emb_drop = nn.Dropout(0.6)
        self.drops = nn.Dropout(0.3)


    def forward(self, x_cat, x_cont):
        x = [e(x_cat[:, i]) for i, e in enumerate(self.embeddings)]
        x = torch.cat(x, 1)
        x = self.emb_drop(x)
        x2 = self.bn1(x_cont)
        x = torch.cat([x, x2], 1)
        x = F.relu(self.lin1(x))
        x = self.drops(x)
        x = self.bn2(x)
        x = F.relu(self.lin2(x))
        x = self.drops(x)
        x = self.bn3(x)
        x = self.lin3(x)

        return x
