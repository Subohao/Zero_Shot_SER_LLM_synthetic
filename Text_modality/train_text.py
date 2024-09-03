from model import Text_clf
from utils import get_data_loader, count_parameters, tokenize_and_numericalize_example, get_accuracy, show_metrics
import torch, transformers
import torch.nn as nn
import torch.optim as optim
import numpy as np
from datasets import Dataset as DA
import pandas as pd
import pyarrow as pa
import json, tqdm, collections


transformer_name = "bert-base-uncased"

tokenizer = transformers.AutoTokenizer.from_pretrained(transformer_name)
root_dir = 'PATH_TO_TRAIN_META_DATA'
with open('{}/train_meta_data.json'.format(root_dir), 'r') as f:
    syn_train_dataset = json.load(f)
with open('{}/test_meta_data.json'.format(root_dir), 'r') as f:
    iemo_test_dataset = json.load(f)
train_df = pd.DataFrame(syn_train_dataset['meta_data'])
train_df['label'] = train_df['label'].map({'neu':0, 'hap':1, 'ang':2, 'sad':3})
data_train = DA(pa.Table.from_pandas(train_df))
test_df = pd.DataFrame(iemo_test_dataset['meta_data'])
test_df['label'] = test_df['label'].map({'neu':0, 'hap':1, 'ang':2, 'sad':3})
data_test = DA(pa.Table.from_pandas(test_df))

train_data = data_train.map(
    tokenize_and_numericalize_example, fn_kwargs={"tokenizer": tokenizer}
)
test_data = data_test.map(
    tokenize_and_numericalize_example, fn_kwargs={"tokenizer": tokenizer}
)

pad_index = tokenizer.pad_token_id
test_size = 0.2

train_valid_data = train_data.train_test_split(test_size=test_size)
train_data = train_valid_data["train"]
valid_data = train_valid_data["test"]

train_data = train_data.with_format(type="torch", columns=["ids", "label"])
valid_data = valid_data.with_format(type="torch", columns=["ids", "label"])
test_data = test_data.with_format(type="torch", columns=["ids", "label"])

batch_size = 128

train_data_loader = get_data_loader(train_data, batch_size, pad_index, shuffle=True)
valid_data_loader = get_data_loader(valid_data, batch_size, pad_index)
test_data_loader = get_data_loader(test_data, batch_size, pad_index)

transformer = transformers.AutoModel.from_pretrained(transformer_name)

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
Model_text_clf = Text_clf(transformer=transformer, input_dim=768, class_num=4, freeze=True).to(device)
print(f"The model has {count_parameters(Model_text_clf):,} trainable parameters")
epochs = 200
lr = 7e-4
optimizer = optim.Adam(Model_text_clf.parameters(), lr=lr)
criterion = nn.CrossEntropyLoss().cuda()

def train(data_loader, model, criterion, optimizer, device):
    model.train()
    epoch_losses = []
    epoch_accs = []
    for batch in tqdm.tqdm(data_loader, desc="training..."):
        ids = batch["ids"].to(device)
        label = batch["label"].to(device)
        prediction = model(ids)
        loss = criterion(prediction, label)
        accuracy = get_accuracy(prediction, label)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        epoch_losses.append(loss.item())
        epoch_accs.append(accuracy.item())
    return np.mean(epoch_losses), np.mean(epoch_accs)

def evaluate(data_loader, model, criterion, device):
    model.eval()
    epoch_losses = []
    epoch_accs = []
    true_final = []
    pred_final = []
    with torch.no_grad():
        for batch in tqdm.tqdm(data_loader, desc="evaluating..."):
            ids = batch["ids"].to(device)
            label = batch["label"].to(device)
            prediction = model(ids)
            predicted_classes = prediction.argmax(dim=-1)
            loss = criterion(prediction, label)
            true_final.extend(label.data.cpu().numpy())
            pred_final.extend(predicted_classes.data.cpu().numpy())
            accuracy = get_accuracy(prediction, label)
            epoch_losses.append(loss.item())
            epoch_accs.append(accuracy.item())
    return np.mean(epoch_losses), np.mean(epoch_accs), true_final, pred_final


n_epochs = 50
best_valid_loss = float("inf")
metrics = collections.defaultdict(list)

for epoch in range(n_epochs):
    train_loss, train_acc = train(
        train_data_loader, Model_text_clf, criterion, optimizer, device
    )
    valid_loss, valid_acc, _, _ = evaluate(valid_data_loader, Model_text_clf, criterion, device)
    metrics["train_losses"].append(train_loss)
    metrics["train_accs"].append(train_acc)
    metrics["valid_losses"].append(valid_loss)
    metrics["valid_accs"].append(valid_acc)
    if valid_loss < best_valid_loss:
        best_valid_loss = valid_loss
        torch.save(Model_text_clf.state_dict(), "transformer.pt")
    print(f"epoch: {epoch}")
    print(f"train_loss: {train_loss:.3f}, train_acc: {train_acc:.3f}")
    print(f"valid_loss: {valid_loss:.3f}, valid_acc: {valid_acc:.3f}")

Model_text_clf.load_state_dict(torch.load("transformer.pt"))

test_loss, test_acc, true_final, pred_final = evaluate(test_data_loader, Model_text_clf, criterion, device)
show_metrics(true_final, pred_final)