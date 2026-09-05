import torch
import torch.nn as nn

class MultiHeadAttention(nn.Module):
    def __init__(self, EmbeddingsDim, NumHeads, AttentionDropout = 0.0, ProjectionDropout = 0.0):
        super().__init__()
        assert EmbeddingsDim % NumHeads == 0, "Embeddings Dimension must be divisible by Number of Heads"

        self.NumHeads = NumHeads
        self.EmbeddingsDim = EmbeddingsDim
        self.HeadDim = self.EmbeddingsDim // self.NumHeads

        self.Q = nn.Linear(EmbeddingsDim, EmbeddingsDim) 
        self.K = nn.Linear(EmbeddingsDim, EmbeddingsDim) 
        self.V = nn.Linear(EmbeddingsDim, EmbeddingsDim) 
        self.Projection = nn.Linear(EmbeddingsDim, EmbeddingsDim)
        self.AttentionDropout = nn.Dropout(AttentionDropout)
        self.ProjectionDropout = nn.Dropout(ProjectionDropout)

    def forward(self, Image, ReturnAttention= False):
        BatchSize, Tokens, Embeddings = Image.shape

        Q = self.Q(Image)
        K = self.K(Image)
        V = self.V(Image)

        Q = Q.reshape(BatchSize, Tokens, self.NumHeads, self.HeadDim).transpose(1, 2)
        K = K.reshape(BatchSize, Tokens, self.NumHeads, self.HeadDim).transpose(1, 2)
        V = V.reshape(BatchSize, Tokens, self.NumHeads, self.HeadDim).transpose(1, 2)

        attention = (Q @ K.transpose(-2, -1)) / (self.HeadDim ** 0.5)
        attention = torch.softmax(attention, dim=1)
        attention = self.AttentionDropout(attention)

        output = attention @ V
        output = output.transpose(1 ,2).reshape(BatchSize, Tokens, Embeddings)
        output = self.ProjectionDropout(self.Projection(output))

        if ReturnAttention:
            return output, attention

        return output