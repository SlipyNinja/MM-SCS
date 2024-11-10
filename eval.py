import argparse
import torch
import pandas as pd
from torch import nn
from torch.utils.data import Dataset, DataLoader
from transformers import AlbertTokenizer, AlbertModel
from tqdm import tqdm

# Define the evaluation function
def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate the MM-SCS Model with Configurable Parameters")
    parser.add_argument('--test_data_path', type=str, default='./dataset/test.csv', help='Path to the test dataset')
    parser.add_argument('--model_path', type=str, default='./mmscs_model.pth', help='Path to the saved model')
    parser.add_argument('--batch_size', type=int, default=4, help='Batch size for evaluation')
    parser.add_argument('--max_length', type=int, default=128, help='Maximum token length for sequences')
    return parser.parse_args()

class SmartContractTestDataset(Dataset):
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
        docstring_tokens = self.tokenize_sequence(row['docstring_tokens']).long()

        return {
            'function_tokens': function_tokens.squeeze(0),
            'function_name': function_name.squeeze(0),
            'api_sequence': api_sequence.squeeze(0),
            'docstring_tokens': docstring_tokens.squeeze(0),
        }

    def tokenize_sequence(self, sequence):
        return self.tokenizer.encode(sequence, max_length=self.max_length, padding='max_length', truncation=True, return_tensors="pt")

    def api_sequence_to_tensor(self, api_sequence):
        return self.tokenizer.encode(" ".join(api_sequence), max_length=self.max_length, padding='max_length', truncation=True, return_tensors="pt")

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

def main(args):
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    # Load test data
    test_data = pd.read_csv(args.test_data_path)
    test_data = test_data.dropna(subset=['function_tokens', 'docstring_tokens', 'graph_structure', 'api_sequence'])

    # Initialize tokenizer
    tokenizer = AlbertTokenizer.from_pretrained("albert-base-v2")

    # Initialize and load the model
    albert_model = AlbertModel.from_pretrained("albert-base-v2").to(device)
    model = MMSCSModel(albert_model).to(device)
    model.load_state_dict(torch.load(args.model_path, map_location=device))
    model.eval()

    # Initialize dataset and data loader
    test_dataset = SmartContractTestDataset(test_data, tokenizer, args.max_length)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False)

    # Evaluate the model
    similarities = []
    with torch.no_grad():
        for batch in tqdm(test_loader, desc="Testing"):
            function_tokens = batch['function_tokens'].to(device)
            function_name = batch['function_name'].to(device)
            api_sequence = batch['api_sequence'].to(device)
            docstring_tokens = batch['docstring_tokens'].to(device)

            similarity = model(function_tokens, function_name, api_sequence, docstring_tokens)
            similarities.extend(similarity.cpu().numpy())

    # Output similarity scores
    for i, score in enumerate(similarities):
        print(f"Sample {i+1} - Similarity Score: {score:.4f}")

if __name__ == '__main__':
    args = parse_args()
    main(args)
