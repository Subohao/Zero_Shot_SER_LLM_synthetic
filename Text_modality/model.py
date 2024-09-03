import torch
import torch.nn as nn
import numpy as np
import pdb

class SelfAttentionPooling(nn.Module):
    """
    Implementation of SelfAttentionPooling
    Original Paper: Self-Attention Encoding and Pooling for Speaker Recognition
    https://arxiv.org/pdf/2008.01077v1.pdf
    """
    def __init__(self, input_dim):
        super(SelfAttentionPooling, self).__init__()
        self.W = nn.Linear(input_dim, 1)
        self.softmax = nn.functional.softmax

    def forward(self, batch_rep, att_mask=None):
        """
            N: batch size, T: sequence length, H: Hidden dimension
            input:
                batch_rep : size (N, T, H)
            attention_weight:
                att_w : size (N, T, 1)
            return:
                utter_rep: size (N, H)
        """
        att_logits = self.W(batch_rep).squeeze(-1)
        if att_mask is not None:
            att_logits = att_mask + att_logits
        att_w = self.softmax(att_logits, dim=-1).unsqueeze(-1)
        utter_rep = torch.sum(batch_rep * att_w, dim=1)

        return utter_rep

class Text_clf(nn.Module):
    def __init__(self, transformer, input_dim, class_num, freeze):
        super(Text_clf, self).__init__()
        self.transformer = transformer
        if freeze:
            for param in self.transformer.parameters():
                param.requires_grad = False
        self.input_dim = input_dim
        self.class_num = class_num
        self.conv1 = nn.Conv1d(self.input_dim, 128, 1)
        self.conv2 = nn.Conv1d(128, 64, 8, padding=4)
        self.conv3 = nn.Conv1d(64, 32, 4, padding=2)
        self.relu = nn.ReLU()
        self.clf = nn.Linear(32, self.class_num)
        self.pooling = nn.AdaptiveAvgPool1d(1)
    def forward(self, x):
        output = self.transformer(x, output_attentions=True)
        hidden = output.last_hidden_state
        # hidden = [batch size, seq len, hidden dim]
        # attention = output.attentions[-1]
        # attention = [batch size, n heads, seq len, seq len]
        # cls_hidden = hidden[:, 0, :]
        # prediction = self.fc(torch.tanh(cls_hidden))
        # # prediction = [batch size, output dim]
        #x : [B, T, H]
        hidden = hidden.permute(0, 2, 1)
        # pdb.set_trace()
        hidden = self.relu(self.conv1(hidden))
        hidden = self.relu(self.conv2(hidden))
        hidden = self.relu(self.conv3(hidden))
        x_pool = self.pooling(hidden)
        out = self.clf(x_pool.squeeze(2))
        return out
