import torch
from torch import nn
from torch.utils.data import DataLoader, random_split

from dataset import BiasDataset
from network import NetworkClassifer

device = (
    torch.accelerator.current_accelerator().type
    if torch.accelerator.is_available()
    else "cpu"
)

if __name__ == "__main__":
    # download from https://zenodo.org/records/13961155
    dataset = BiasDataset("data/dataset_with_labels.parquet")
    val_size = int(0.2 * len(dataset))
    train_set, val_set = random_split(dataset, [len(dataset) - val_size, val_size])

    train_loader = DataLoader(train_set, batch_size=16, shuffle=True)
    val_loader = DataLoader(val_set, batch_size=16)

    model = NetworkClassifer().to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=2e-5, weight_decay=1e-2)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=2)
    criterion = nn.CrossEntropyLoss()

    epochs = 3
    best_val_loss = float("inf")

    for epoch in range(epochs):
        _ = model.train()
        total_loss = 0
        for i, batch in enumerate(train_loader):
            if i % 50 == 0:
                print(f"  batch {i}/{len(train_loader)}")
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["label"].to(device)

            optimizer.zero_grad()
            out = model(input_ids, attention_mask)
            loss = criterion(out, labels)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        model.eval()
        val_loss = 0
        correct = 0
        total = 0

        with torch.no_grad():
            for batch in val_loader:
                input_ids = batch["input_ids"].to(device)
                attention_mask = batch["attention_mask"].to(device)
                labels = batch["label"].to(device)

                out = model(input_ids, attention_mask)
                val_loss += criterion(out, labels).item()
                correct += (out.argmax(dim=1) == labels).sum().item()
                total += labels.size(0)

        avg_val_loss = val_loss / len(val_loader)
        scheduler.step(avg_val_loss)

        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            torch.save(model.state_dict(), "./models/best_distilbert.pth")

        print(
            f"Epoch {epoch + 1}/{epochs} "
            f"train loss: {total_loss / len(train_loader):.4f} "
            f"val loss: {avg_val_loss:.4f} "
            f"val acc: {correct / total:.4f}"
        )
