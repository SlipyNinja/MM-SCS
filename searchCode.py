import argparse
import torch
import pandas as pd
from torch import nn
from torch.utils.data import Dataset, DataLoader
from transformers import AlbertTokenizer, AlbertModel
from tqdm import tqdm

# Parse command-line arguments
def parse_args():
    parser = argparse.ArgumentParser(description="Search for code snippets based on query similarity")
    parser.add_argument('--model_path', type=str, default='./mmscs_model.pth', help='Path to the saved model')
    parser.add_argument('--data_path', type=str, default='./dataset/train.csv', help='Path to the code snippets dataset')
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu', help='Device to use for evaluation (cuda or cpu)')
    parser.add_argument('--batch_size', type=int, default=4, help='Batch size for evaluation')
    parser.add_argument('--max_length', type=int, default=128, help='Maximum token length for sequences')
    return parser.parse_args()

# Dataset for loading code snippets
class CodeDataset(Dataset):
    def __init__(self, data, tokenizer, max_length):
        self.data = data
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        row = self.data.iloc[idx]
        function_tokens = self.tokenize_sequence(row['function_tokens']).long()
        function_name = self.tokenize_sequence(row['function_name']).long()
        api_sequence = self.api_sequence_to_tensor(eval(row['api_sequence']) if row['api_sequence'] != "[]" else []).long()

        return {
            'function_tokens': function_tokens.squeeze(0),
            'function_name': function_name.squeeze(0),
            'api_sequence': api_sequence.squeeze(0),
            'original_function': row['original_function']  # Keep original code
        }

    def tokenize_sequence(self, sequence):
        return self.tokenizer.encode(sequence, max_length=self.max_length, padding='max_length', truncation=True, return_tensors="pt")

    def api_sequence_to_tensor(self, api_sequence):
        return self.tokenizer.encode(" ".join(api_sequence), max_length=self.max_length, padding='max_length', truncation=True, return_tensors="pt")

# Model definition
class MMSCSModel(nn.Module):
    def __init__(self, albert_model, embedding_dim=768):
        super(MMSCSModel, self).__init__()
        self.albert = albert_model
        self.fc = nn.Linear(embedding_dim * 3, embedding_dim)
        self.cosine_similarity = nn.CosineSimilarity(dim=1, eps=1e-6)

    def forward(self, function_tokens, function_name, api_sequence, docstring_tokens):
        function_tokens_embeds = self.albert(function_tokens)[1]
        function_name_embeds = self.albert(function_name)[1]
        api_sequence_embeds = self.albert(api_sequence)[1]
        docstring_embeds = self.albert(docstring_tokens)[1]

        combined_features = torch.cat([function_tokens_embeds, function_name_embeds, api_sequence_embeds], dim=1)
        combined_embeddings = self.fc(combined_features)

        similarity = self.cosine_similarity(combined_embeddings, docstring_embeds)
        return similarity

# Function to search for similar code snippets
def search_code(query, model, code_loader, tokenizer, device, max_length):
    # Convert the query to docstring_tokens format
    query_tokens = tokenizer.encode(query, max_length=max_length, padding='max_length', truncation=True, return_tensors="pt").to(device)

    results = []
    with torch.no_grad():
        for batch in tqdm(code_loader, desc="Searching"):
            function_tokens = batch['function_tokens'].to(device)
            function_name = batch['function_name'].to(device)
            api_sequence = batch['api_sequence'].to(device)
            original_functions = batch['original_function']

            # Calculate similarity
            similarity = model(function_tokens, function_name, api_sequence, query_tokens)
            similarity_scores = similarity.cpu().numpy()

            # Pair similarity scores with code snippets
            for score, function in zip(similarity_scores, original_functions):
                results.append((score, function))

    # Sort by similarity and return the top 5 results
    results = sorted(results, key=lambda x: x[0], reverse=True)
    return results[:5]

def main(args):
    # Set device
    device = args.device

    # Load the code snippets data
    code_data = pd.read_csv(args.data_path)
    code_data = code_data.dropna(subset=['function_tokens', 'docstring_tokens', 'graph_structure', 'api_sequence'])

    # Initialize tokenizer
    tokenizer = AlbertTokenizer.from_pretrained("albert-base-v2")

    # Load model
    albert_model = AlbertModel.from_pretrained("albert-base-v2").to(device)
    model = MMSCSModel(albert_model).to(device)
    model.load_state_dict(torch.load(args.model_path, map_location=device))
    model.eval()

    # Initialize dataset and data loader
    code_dataset = CodeDataset(code_data, tokenizer, args.max_length)
    code_loader = DataLoader(code_dataset, batch_size=args.batch_size, shuffle=False)

    # Get user query and search for similar code snippets
    query = input("Enter your query: ")
    results = search_code(query, model, code_loader, tokenizer, device, args.max_length)

    # Display search results
    print("\nTop matching code snippets:")
    for i, (score, function) in enumerate(results, 1):
        print(f"\nResult {i} - Similarity Score: {score:.4f}")
        print(function)

if __name__ == '__main__':
    args = parse_args()
    main(args)
