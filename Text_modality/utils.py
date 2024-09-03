import torch
import os
from torch.nn.utils.rnn import pad_sequence
import torch.nn as nn
from sklearn.metrics import recall_score, confusion_matrix, accuracy_score, classification_report

def get_collate_fn(pad_index):
    def collate_fn(batch):
        batch_ids = [i["ids"] for i in batch]
        batch_ids = nn.utils.rnn.pad_sequence(
            batch_ids, padding_value=pad_index, batch_first=True
        )
        batch_label = [i["label"] for i in batch]
        batch_label = torch.stack(batch_label)
        batch = {"ids": batch_ids, "label": batch_label}
        return batch

    return collate_fn

def get_accuracy(prediction, label):
    batch_size, _ = prediction.shape
    predicted_classes = prediction.argmax(dim=-1)
    correct_predictions = predicted_classes.eq(label).sum()
    accuracy = correct_predictions / batch_size
    return accuracy

def get_data_loader(dataset, batch_size, pad_index, shuffle=False):
    collate_fn = get_collate_fn(pad_index)
    data_loader = torch.utils.data.DataLoader(
        dataset=dataset,
        batch_size=batch_size,
        collate_fn=collate_fn,
        shuffle=shuffle,
    )
    return data_loader

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def tokenize_and_numericalize_example(example, tokenizer):
    ids = tokenizer(example["transcription"], truncation=True)["input_ids"]
    return {"ids": ids}

def pad_collate(batch):
    (xx, yy) = zip(*batch)
    x_lens = [len(x) for x in xx]
    # y_lens = [len(y) for y in yy]
    # pdb.set_trace()
    xx_pad = pad_sequence(xx, batch_first=True, padding_value=0)
    # yy_pad = pad_sequence(yy, batch_first=True, padding_value=0)
    return xx_pad, x_lens, torch.tensor(yy)

def show_metrics(true_final, pred_final):
    UAR = recall_score(true_final, pred_final, average='macro')
    ACC = accuracy_score(true_final, pred_final)
    print("UAR : {:.2f}%".format(UAR*100))
    print("ACC: {:.2f}%".format(ACC*100))
    emo = ['Neutral', 'Happy', 'Angry', 'Sad']
    report = classification_report(true_final, pred_final, target_names=emo)
    cf_mat = confusion_matrix(true_final, pred_final)
    recall_record = []
    for i in range(len(emo)):
        recall = cf_mat[i, i]/sum(cf_mat[i, :])
        recall_record.append(recall)
        print('{} Recall : {:.2f}%'.format(emo[i], recall*100))
    print(report)