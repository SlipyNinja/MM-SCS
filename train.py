import argparse
import json
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
import torch.nn.functional as F
from transformers import AlbertTokenizer, AlbertModel
from tqdm import tqdm
from torch.cuda.amp import autocast, GradScaler
from torch_geometric.data import Data
from sklearn.preprocessing import OneHotEncoder
from model import MMSCSModel  # Import the model

def parse_args():
    parser = argparse.ArgumentParser(description="Train MM-SCS Model with GAT")
    parser.add_argument('--data_path', type=str, default='./dataset/train.csv', help='Path to the training dataset')
    parser.add_argument('--save_path', type=str, default='./mmscs_model.pth', help='Path to save the trained model')
    parser.add_argument('--batch_size', type=int, default=6, help='Batch size for training')
    parser.add_argument('--epochs', type=int, default=5, help='Number of epochs')
    parser.add_argument('--learning_rate', type=float, default=2e-5, help='Learning rate')
    parser.add_argument('--max_length', type=int, default=128, help='Maximum token length for sequences')
    parser.add_argument('--margin', type=float, default=0.05, help='Margin for MarginRankingLoss')
    return parser.parse_args()

# Helper functions
def tokenize_sequence(sequence, tokenizer, max_length):
    return tokenizer.encode(sequence, max_length=max_length, padding='max_length', truncation=True, return_tensors="pt")

def parse_graph_structure(graph_json):
    try:
        return json.loads(graph_json.replace("'", '"'))
    except json.JSONDecodeError:
        return None

def api_sequence_to_tensor(api_sequence, tokenizer, max_length):
    return tokenizer.encode(" ".join(api_sequence), max_length=max_length, padding='max_length', truncation=True, return_tensors="pt")

class SmartContractDataset(Dataset):
    def __init__(self, data, tokenizer, max_length):
        self.data = data
        self.tokenizer = tokenizer
        self.max_length = max_length

        # Dynamically collect all possible categories and types from nodes
        categories_types = []
        for _, row in data.iterrows():
            graph_structure = parse_graph_structure(row['graph_structure'])
            if graph_structure:
                for node in graph_structure['nodes']:
                    categories_types.append([node['Category'], node['Type']])
        
        # One-hot encode categories and types, ignoring unknown categories
        self.node_encoder = OneHotEncoder(sparse_output=False, handle_unknown='ignore')
        self.node_encoder.fit(categories_types)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        row = self.data.iloc[idx]
        function_tokens = tokenize_sequence(row['function_tokens'], self.tokenizer, self.max_length).long()
        function_name = tokenize_sequence(row['function_name'], self.tokenizer, self.max_length).long()
        api_sequence = api_sequence_to_tensor(eval(row['api_sequence']) if row['api_sequence'] != "[]" else [], self.tokenizer, self.max_length).long()
        docstring_tokens = tokenize_sequence(row['docstring_tokens'], self.tokenizer, self.max_length).long()
        negative_example = self.data['docstring_tokens'].sample(1).values[0]
        negative_docstring_tokens = tokenize_sequence(negative_example, self.tokenizer, self.max_length).long()

        # Parse graph structure for GAT
        graph_structure = parse_graph_structure(row['graph_structure'])
        if graph_structure:
            node_id_map = {node['id']: idx for idx, node in enumerate(graph_structure['nodes'])}
            node_features = []
            for node in graph_structure['nodes']:
                category_type = [[node['Category'], node['Type']]]
                encoded_feature = self.node_encoder.transform(category_type).flatten()
                node_features.append(encoded_feature)
            x = torch.tensor(node_features, dtype=torch.float)
            edges = [(node_id_map[edge['V_s']], node_id_map[edge['V_e']]) for edge in graph_structure['edges']]
            edge_index = torch.tensor(edges, dtype=torch.long).t().contiguous()
            graph_data = Data(x=x, edge_index=edge_index)
        else:
            graph_data = Data(x=torch.zeros((1, self.node_encoder.categories_[0].size + self.node_encoder.categories_[1].size)), edge_index=torch.zeros((2, 0), dtype=torch.long))

        return {
            'function_tokens': function_tokens.squeeze(0),
            'function_name': function_name.squeeze(0),
            'api_sequence': api_sequence.squeeze(0),
            'docstring_tokens': docstring_tokens.squeeze(0),
            'negative_docstring_tokens': negative_docstring_tokens.squeeze(0),
            'graph_data': graph_data
        }

def train_model(model, data_loader, optimizer, scaler, criterion, device, epochs):
    model.train()
    for epoch in range(epochs):
        total_loss = 0.0
        progress_bar = tqdm(data_loader, desc=f"Epoch {epoch+1}/{epochs}")
        for batch in progress_bar:
            function_tokens = batch['function_tokens'].to(device)
            function_name = batch['function_name'].to(device)
            api_sequence = batch['api_sequence'].to(device)
            docstring_tokens = batch['docstring_tokens'].to(device)
            negative_docstring_tokens = batch['negative_docstring_tokens'].to(device)
            graph_data = batch['graph_data'][0].to(device)

            optimizer.zero_grad()
            with autocast():
                pos_similarity, neg_similarity = model(
                    function_tokens, function_name, api_sequence,
                    docstring_tokens, negative_docstring_tokens, graph_data
                )
                target = torch.ones_like(pos_similarity).to(device)
                loss = criterion(pos_similarity, neg_similarity, target)

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            total_loss += loss.item()
            progress_bar.set_postfix(loss=total_loss / (progress_bar.n + 1))

        torch.cuda.empty_cache()
        print(f"Epoch {epoch+1}/{epochs} completed, Average Loss: {total_loss / len(data_loader)}")

def main(args):
    model = MMSCSModel(albert_model, embedding_dim=args.max_length, gat_output_dim=256).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate)
    scaler = GradScaler()
    criterion = nn.MarginRankingLoss(margin=args.margin)

    # Train the model
    train_model(model, data_loader, optimizer, scaler, criterion, device, args.epochs)
    torch.save(model.state_dict(), args.save_path)
    print(f"Model saved to {args.save_path}")

if __name__ == '__main__':
    args = parse_args()
    main(args)

