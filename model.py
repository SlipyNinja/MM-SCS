import json
import torch
from torch import nn
from torch.utils.data import Dataset
import torch.nn.functional as F
from transformers import AlbertTokenizer, AlbertModel

class SmartContractDataset(Dataset):
    def __init__(self, data, tokenizer, max_length):
        self.data = data
        self.tokenizer = tokenizer
        self.max_length = max_length

    def tokenize_sequence(self, sequence):
        return self.tokenizer.encode(sequence, max_length=self.max_length, padding='max_length', truncation=True, return_tensors="pt")

    def parse_graph_structure(self, graph_json):
        try:
            return json.loads(graph_json.replace("'", '"'))
        except json.JSONDecodeError:
            return None

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        row = self.data.iloc[idx]
        function_tokens = self.tokenize_sequence(row['function_tokens']).squeeze(0)
        function_name = self.tokenize_sequence(row['function_name']).squeeze(0)
        api_sequence = self.tokenize_sequence(" ".join(eval(row['api_sequence']) if row['api_sequence'] != "[]" else "")).squeeze(0)
        docstring_tokens = self.tokenize_sequence(row['docstring_tokens']).squeeze(0)
        negative_example = self.data['docstring_tokens'].sample(1).values[0]
        negative_docstring_tokens = self.tokenize_sequence(negative_example).squeeze(0)

        return {
            'function_tokens': function_tokens,
            'function_name': function_name,
            'api_sequence': api_sequence,
            'docstring_tokens': docstring_tokens,
            'negative_docstring_tokens': negative_docstring_tokens,
        }

def collate_fn(batch):
    keys = batch[0].keys()
    return {key: torch.stack([b[key] for b in batch]) for key in keys}

class MMSCSModel(nn.Module):
    def __init__(self, embedding_dim=768):
        super(MMSCSModel, self).__init__()
        self.albert = AlbertModel.from_pretrained("albert-base-v2")
        self.fc = nn.Linear(embedding_dim * 3, embedding_dim)
        self.cosine_similarity = nn.CosineSimilarity(dim=1, eps=1e-6)

    def forward(self, function_tokens, function_name, api_sequence, docstring_tokens, negative_docstring_tokens):
        function_tokens_embeds = self.albert(function_tokens)[1]
        function_name_embeds = self.albert(function_name)[1]
        api_sequence_embeds = self.albert(api_sequence)[1]
        docstring_embeds = self.albert(docstring_tokens)[1]
        neg_docstring_embeds = self.albert(negative_docstring_tokens)[1]

        combined_features = torch.cat([function_tokens_embeds, function_name_embeds, api_sequence_embeds], dim=1)
        combined_embeddings = self.fc(combined_features)

        pos_similarity = self.cosine_similarity(combined_embeddings, docstring_embeds)
        neg_similarity = self.cosine_similarity(combined_embeddings, neg_docstring_embeds)

        return pos_similarity, neg_similarity
