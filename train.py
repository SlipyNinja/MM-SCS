import argparse
import pandas as pd
import torch
from torch.utils.data import DataLoader
from torch import nn, optim
from torch.cuda.amp import GradScaler, autocast
from tqdm import tqdm
from model import MMSCSModel, SmartContractDataset, collate_fn
from transformers import AlbertTokenizer

def train_model(args):
    # Set up device
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    # Load data
    data = pd.read_csv(args.data_path).dropna(subset=['function_tokens', 'docstring_tokens', 'graph_structure', 'api_sequence'])
    tokenizer = AlbertTokenizer.from_pretrained("albert-base-v2")
    dataset = SmartContractDataset(data, tokenizer, args.max_length)
    dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True, collate_fn=collate_fn)

    # Initialize model, optimizer, and loss function
    model = MMSCSModel().to(device)
    optimizer = optim.Adam(model.parameters(), lr=args.learning_rate)
    criterion = nn.MarginRankingLoss(margin=args.margin)
    scaler = GradScaler()

    # Training loop
    model.train()
    for epoch in range(args.epochs):
        total_loss = 0.0
        progress_bar = tqdm(dataloader, desc=f"Epoch {epoch+1}/{args.epochs}")
        for batch in progress_bar:
            optimizer.zero_grad()
            with autocast():
                pos_similarity, neg_similarity = model(
                    batch['function_tokens'].to(device),
                    batch['function_name'].to(device),
                    batch['api_sequence'].to(device),
                    batch['docstring_tokens'].to(device),
                    batch['negative_docstring_tokens'].to(device)
                )
                target = torch.ones_like(pos_similarity).to(device)
                loss = criterion(pos_similarity, neg_similarity, target)

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            total_loss += loss.item()
            progress_bar.set_postfix(loss=total_loss / (progress_bar.n + 1))

        print(f"Epoch {epoch+1} completed, Avg Loss: {total_loss / len(dataloader)}")

    torch.save(model.state_dict(), args.save_path)
    print(f"Model saved to {args.save_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train MM-SCS Model")
    parser.add_argument("--data_path", type=str, default="./dataset/train.csv", help="Path to training data (default: ./dataset/train.csv)")
    parser.add_argument("--save_path", type=str, default="./mmscs_model.pth", help="Path to save the trained model (default: ./mmscs_model.pth)")
    parser.add_argument("--batch_size", type=int, default=8, help="Batch size for training (default: 8)")
    parser.add_argument("--epochs", type=int, default=10, help="Number of epochs to train (default: 10)")
    parser.add_argument("--learning_rate", type=float, default=3e-5, help="Learning rate for optimizer (default: 3e-5)")
    parser.add_argument("--max_length", type=int, default=128, help="Maximum token length (default: 128)")
    parser.add_argument("--margin", type=float, default=0.1, help="Margin for MarginRankingLoss (default: 0.1)")
    args = parser.parse_args()

    train_model(args)
