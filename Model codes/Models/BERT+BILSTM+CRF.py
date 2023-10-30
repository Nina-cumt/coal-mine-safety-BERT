# coding: UTF-8
import torch
import torch.nn as nn
from sklearn.metrics import f1_score, classification_report
import numpy as np
from pytorch_pretrained import BertModel, BertTokenizer
from torch.autograd import Variable
from .CRF import *


class Config(object):
    """配置参数"""

    def __init__(self):
        self.model_name = 'Bert_Bilstm_crf'

        self.train_data_path = './data/train/source.txt'  # 文本训练集
        self.train_label_path = './data/train/target.txt'  # 标签训练集
        self.dev_data_path = './data/dev/source.txt'  # 文本验证集
        self.dev_label_path = './data/dev/target.txt'  # 标签验证集
        self.save_path = './Result/Save_path/' + self.model_name + '.ckpt'  # 模型训练结果
        self.bert_path = './bert_pretrain'

        #self.tokenizer = BertTokenizer.from_pretrained('./bert_pretrain')
        self.tokenizer = BertTokenizer.from_pretrained(self.bert_path)

        # self.vocab_class = {'B_T': 0, 'I_T': 1, 'B_LOC': 2, 'I_LOC': 3, 'B_ORG': 4,
        #                     'I_ORG': 5, 'B_PER': 6, 'I_PER': 7, 'O': 8}  # 词性类别名单
        self.vocab_class = {'B-Facility': 0,
                            'I-Facility': 1,
                            'B-Materials': 2,
                            'I-Materials': 3,
                            'B-Atmospheric': 4,
                            'I-Atmospheric': 5,
                            'B-Geology': 6,
                            'I-Geology': 7,
                            'B-Person': 8,
                            'I-Person': 9,
                            'B-Project': 10,
                            'I-Project': 11,
                            'B-Task': 12,
                            'I-Task': 13,
                            'B-Index': 14,
                            'I-Index': 15,
                            'B-Place': 16,
                            'I-Place': 17,
                            'B-Text': 18,
                            'I-Text': 19,
                            'B-Management': 20,
                            'I-Management': 21,
                            'B-Method': 22,
                            'I-Method': 23, 'O': 24}
        self.tagset_size = len(self.vocab_class)
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')  # 设备

        self.num_epochs = 511  # epoch数
        self.batch_size = 128
        self.pad_size = 150  # 每句话处理成的长度(短填长切)
        #self.learning_rate = 5e-5  # 学习率
        self.learning_rate = 0.005  # 学习率

        self.learning_rate_decay = 0.0005  # 学习率衰减
        #self.learning_rate_decay = 5e-6  # 学习率衰减
        self.hidden_size = 128
        self.embedding_dim = 768
        self.num_layers = 1
        self.dropout = 0.5


class Model(nn.Module):

    def __init__(self, config):
        super(Model, self).__init__()
        self.embedding_dim = config.embedding_dim
        self.num_layers = config.num_layers
        self.hidden_size = config.hidden_size
        self.device=config.device

        self.bert = BertModel.from_pretrained(config.bert_path)
        self.lstm = nn.LSTM(config.embedding_dim, config.hidden_size, num_layers=config.num_layers,
                            bidirectional=True, dropout=config.dropout,batch_first=True)
        self.dropout = nn.Dropout(config.dropout)
        self.crf = CRF(config.tagset_size)
        self.fc = nn.Linear(config.hidden_size * 2, config.tagset_size)

    def init_hidden(self, batch_size):
        return torch.randn(2 * self.num_layers, batch_size, self.hidden_size).to(self.device),torch.randn(2 * self.num_layers, batch_size, self.hidden_size).to(self.device)

    def forward(self, input):
        context = input[0]  # 输入的句子
        mask = input[1]
        batch_size = context.size(0)
        seq_len = context.size(1)
        with torch.no_grad():
            embeds, _ = self.bert(context, attention_mask=mask, output_all_encoded_layers=False)
        hidden = self.init_hidden(batch_size=batch_size)
        out, hidden = self.lstm(embeds, hidden)
        out = out.contiguous().view(-1, self.hidden_size * 2)
        out = self.dropout(out)
        out = self.fc(out)
        out = out.contiguous().view(batch_size, seq_len, -1)
        return out

    def loss(self, features, mask, label):
        loss_value = self.crf.negative_log_loss(features, mask, label)
        batch_size = features.size(0)
        loss_value /= float(batch_size)
        return loss_value

    def predict(self, bert_encode, output_mask):
        predicts = self.crf.get_batch_best_path(bert_encode, output_mask)
        return predicts

    def acc_f1(self, y_pred, y_true):
        y_pred = y_pred.cpu().numpy()
        y_true = y_true.cpu().numpy()
        f1 = f1_score(y_true, y_pred, average="macro")
        correct = np.sum((y_true == y_pred).astype(int))
        acc = correct / y_pred.shape[0]
        print('acc: {}'.format(acc))
        print('f1: {}'.format(f1))
        return acc, f1

    def class_report(self, y_pred, y_true):
        y_true = y_true.cpu().numpy()
        y_pred = y_pred.cpu().numpy()
        classify_report = classification_report(y_true, y_pred)
        print('\n\nclassify_report:\n', classify_report)
