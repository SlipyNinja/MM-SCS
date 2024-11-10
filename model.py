import torch
from torch import nn
import torch.nn.functional as F
from torch_geometric.nn import GATConv
from transformers import AlbertModel

class MMSCSModel(nn.Module):
    def __init__(self, albert_model, embedding_dim=768, gat_output_dim=256):
        super(MMSCSModel, self).__init__()
        self.albert = albert_model
        self.embedding_dim = embedding_dim
        self.fc = nn.Linear(embedding_dim * 3 + gat_output_dim, embedding_dim)
        
        # GAT layer with adjusted input feature dimension
        self.gat = GATConv(embedding_dim, gat_output_dim, heads=1, concat=False)
        self.gat_fc = nn.Linear(gat_output_dim, gat_output_dim)
        
        self.cosine_similarity = nn.CosineSimilarity(dim=1, eps=1e-6)

    def forward(self, function_tokens, function_name, api_sequence, docstring_tokens, negative_docstring_tokens, graph_data):
        function_tokens_embeds = self.albert(function_tokens)[1]
        function_name_embeds = self.albert(function_name)[1]
        api_sequence_embeds = self.albert(api_sequence)[1]
        docstring_embeds = self.albert(docstring_tokens)[1]
        neg_docstring_embeds = self.albert(negative_docstring_tokens)[1]

        # Process graph structure
        x, edge_index = graph_data.x.to(function_tokens.device), graph_data.edge_index.to(function_tokens.device)
        
        # Ensure x matches GATConv's in_channels by padding if necessary
        if x.size(1) != self.embedding_dim:
            x = F.pad(x, (0, self.embedding_dim - x.size(1)), "constant", 0)

        gat_output = self.gat(x, edge_index)
        gat_output = F.elu(gat_output)
        gat_output = self.gat_fc(gat_output.mean(dim=0, keepdim=True))  # Aggregate node features
        
        # Expand gat_output to match batch size
        batch_size = function_tokens_embeds.size(0)
        gat_output = gat_output.expand(batch_size, -1)

        # Concatenate features and calculate similarity
        combined_features = torch.cat([function_tokens_embeds, function_name_embeds, api_sequence_embeds, gat_output], dim=1)
        combined_embeddings = self.fc(combined_features)

        pos_similarity = self.cosine_similarity(combined_embeddings, docstring_embeds)
        neg_similarity = self.cosine_similarity(combined_embeddings, neg_docstring_embeds)

        return pos_similarity, neg_similarity
