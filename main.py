import os

import pytreebank
import torch
import torch.optim as optim
from loguru import logger
from pytorch_transformers import (
    BertConfig,
    BertForSequenceClassification,
    BertTokenizer,
)
from torch.utils import data
from tqdm import tqdm

os.environ["CUDA_LAUNCH_BLOCKING"] = "1"
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

logger.info("Loading the tokenizer")
tokenizer = BertTokenizer.from_pretrained("bert-large-uncased")

logger.info("Loading SST")
sst = pytreebank.load_sst()


def rpad(array, n=70):
    """Right padding."""
    current_len = len(array)
    if current_len > n:
        return array[: n - 1]
    extra = n - current_len
    return array + ([0] * extra)


def get_binary_label(label):
    """Convert fine-grained label to binary label."""
    if label < 2:
        return 0
    if label > 2:
        return 1
    raise ValueError("Invalid label")


class SSTDataset(data.Dataset):
    def __init__(self, split="train", root=True, binary=True):
        logger.info(f"Loading SST {split} set")
        self.sst = sst[split]

        logger.info("Tokenizing")
        if root and binary:
            self.data = [
                (
                    rpad(
                        tokenizer.encode("[CLS] " + tree.to_lines()[0] + " [SEP]"), n=66
                    ),
                    get_binary_label(tree.label),
                )
                for tree in self.sst
                if tree.label != 2
            ]
        elif root and not binary:
            self.data = [
                (
                    rpad(
                        tokenizer.encode("[CLS] " + tree.to_lines()[0] + " [SEP]"), n=66
                    ),
                    tree.label,
                )
                for tree in self.sst
            ]
        elif not root and not binary:
            self.data = [
                (rpad(tokenizer.encode("[CLS] " + line + " [SEP]"), n=66), label)
                for tree in self.sst
                for label, line in tree.to_labeled_lines()
            ]
        else:
            self.data = [
                (
                    rpad(tokenizer.encode("[CLS] " + line + " [SEP]"), n=66),
                    get_binary_label(label),
                )
                for tree in self.sst
                for label, line in tree.to_labeled_lines()
                if label != 2
            ]

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        X, y = self.data[index]
        X = torch.tensor(X)
        return X, y


def train(model, dataset, batch_size=32):
    generator = torch.utils.data.DataLoader(
        dataset, batch_size=batch_size, shuffle=True
    )
    model.train()
    train_loss, train_acc = 0.0, 0.0
    for batch, labels in tqdm(generator):
        batch, labels = batch.to(device), labels.to(device)
        optimizer.zero_grad()
        loss, logits = model(batch, labels=labels)
        err = lossfn(logits, labels)
        loss.backward()
        optimizer.step()

        train_loss += loss.item()
        pred_labels = torch.argmax(logits, axis=1)
        train_acc += (pred_labels == labels).sum().item()
    train_loss /= len(dataset)
    train_acc /= len(dataset)
    return train_loss, train_acc


def evaluate(model, dataset, batch_size=32):
    generator = torch.utils.data.DataLoader(
        dataset, batch_size=batch_size, shuffle=True
    )
    # validation
    model.eval()
    loss, acc = 0.0, 0.0
    with torch.no_grad():
        for batch, labels in tqdm(generator):
            batch, labels = batch.to(device), labels.to(device)
            logits = model(batch)[0]
            error = lossfn(logits, labels)
            loss += error.item()
            pred_labels = torch.argmax(logits, axis=1)
            acc += (pred_labels == labels).sum().item()
    loss /= len(dataset)
    acc /= len(dataset)
    return loss, acc


if __name__ == "__main__":
    trainset = SSTDataset("train", root=False, binary=False)
    devset = SSTDataset("dev", root=False, binary=False)
    testset = SSTDataset("test", root=False, binary=False)

    config = BertConfig.from_pretrained("bert-large-uncased")
    config.num_labels = 5
    model = BertForSequenceClassification.from_pretrained(
        "bert-large-uncased", config=config
    )
    # model = BertForSequenceClassification.from_pretrained('bert-large-uncased')
    model = model.to(device)
    lossfn = torch.nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=1e-5)

    for epoch in range(1, 30):
        train_loss, train_acc = train(model, trainset, batch_size=32)
        val_loss, val_acc = evaluate(model, devset, batch_size=32)
        test_loss, test_acc = evaluate(model, testset, batch_size=32)
        logger.info(
            f"{epoch}, {train_loss:.4f}, {val_loss:.4f}, {test_loss:.4f}, "
            f"{train_acc:.3f}, {val_acc:.3f}, {test_acc:.3f}"
        )
        torch.save(model, f"bert_large_all_fine_e{epoch}.pickle")

    logger.success("Done!")
