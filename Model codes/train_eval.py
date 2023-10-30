# coding: UTF-8
import re

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn import metrics
import time
from sklearn.metrics import f1_score, classification_report

from pytorch_pretrained import BertTokenizer
from pytorch_pretrained.optimization import *
from utils import *
from tqdm import tqdm

def train(config, model, train_dataset, dev_dataset):
    start_time = time.time()
    optimizer = torch.optim.Adam(model.parameters(), lr=config.learning_rate)
    total_batch = 0  # 记录进行到多少batch

    checkpoint = torch.load(config.save_path)
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    epoch = checkpoint['epoch']
    loss = checkpoint['loss']
    model.eval()

    epoch = checkpoint['epoch']
    start_epoch = checkpoint['epoch'] + 1


    for epoch in range(config.num_epochs):
        for i, batch in enumerate(train_dataset):
            spend_time=get_time_dif(start_time)
            model.zero_grad()
            model.train()
            batch= tuple(_.to(config.device) for _ in batch)
            input_id,input_mask,label,output_mask = batch
            encode=model(batch)
            loss=model.loss(encode,input_mask,label)
            loss.backward()
            optimizer.step()
            if total_batch % 50 == 0 :
                print('step: {} |  epoch: {}|  loss: {:.4f}|  time: {}|'
                      .format(total_batch, epoch, loss.item(),spend_time))
            total_batch+=1
        dev(model,epoch,dev_dataset,config)
        torch.save(model.state_dict(),config.save_path)



def linetrain(config, model, train_dataset, dev_dataset):
    start_time = time.time()
    total_batch = 0  # 记录进行到多少batch

    checkpoint = torch.load(config.save_path)
    model.load_state_dict(checkpoint)
    optimizer = torch.optim.Adam(model.parameters(), lr=config.learning_rate)
    #继续开始的epoch次数
    epoch1 = 505
    #loss = checkpoint['loss']

    for epoch in range(epoch1,config.num_epochs):
        for i, batch in enumerate(train_dataset):
            spend_time=get_time_dif(start_time)
            model.zero_grad()
            model.train()
            batch= tuple(_.to(config.device) for _ in batch)
            input_id,input_mask,label,output_mask = batch
            encode=model(batch)
            loss=model.loss(encode,input_mask,label)
            loss.backward()
            optimizer.step()
            if total_batch % 50 == 0 :
                print('step: {} |  epoch: {}|  loss: {:.4f}|  time: {}|'
                      .format(total_batch, epoch, loss.item(),spend_time))
            total_batch+=1
        if epoch % 5 == 0:
            dev(model,epoch,dev_dataset,config)
        torch.save(model.state_dict(),config.save_path)

def dev(model,epoch,dev_dataset,config):
    model.eval()
    model.to(config.device)
    count=0
    length = 0
    y_predicts,y_labels=[],[]
    eval_loss,eval_acc,eval_f1=0,0,0
    with torch.no_grad():
        for i,batch in tqdm(enumerate(dev_dataset)):
            batch= tuple(_.to(config.device) for _ in batch)
            input_ids, input_mask, label_ids, output_mask = batch

            encode = model(batch)

            count += 1

            predicts = model.predict(encode, output_mask)

            predicts = predicts.view(1, -1)
            predicts = predicts[predicts != -1]

            #归一并去掉-1
            label_ids1 = label_ids.view(1, -1)
            label_ids1 = label_ids1[label_ids1 != -1]

            if len(label_ids1) == len(predicts):
                eval_los = model.loss(encode, output_mask, label_ids)
                eval_loss = eval_los + eval_loss

                y_predicts.append(predicts)
                y_labels.append(label_ids1)
                length += input_ids.size(0)


        eval_predicted = torch.cat(y_predicts, dim=0)
        eval_labeled = torch.cat(y_labels, dim=0)
        model.acc_f1(eval_predicted, eval_labeled)
        model.class_report(eval_predicted, eval_labeled)
        print('eval  epoch : {}|  loss : {}'.format(epoch,eval_loss/length))

PAD, CLS = '[PAD]', '[CLS]'  # padding符号, bert中综合信息符号


def clean(text):

    text = re.sub(r"\s+", " ", str(text))
    #text = text.replace(" ", "")
    text = text.replace("\n", "")
    return text.strip()


def predict(model,config,input_str=""):
    model.eval()  # 取消batchnorm和dropout,用于评估阶段
    #model.to(config.device)
    #VOCAB = config.bert_path  # your path for model and vocab
    #tokenizer = BertTokenizer.from_pretrained(VOCAB)

    with torch.no_grad():
            #input_str = input("请输入文本: ")
            #input_ids = config.tokenizer.encode(input_str,add_special_tokens=True)  # add_spicial_tokens=True，为自动为sentence加上[CLS]和[SEP]
            #tokens = input_str.split()  # 分词
            input_str = clean(input_str)
            #tokens = [one for one in input_str]
            tokens = config.tokenizer.tokenize(input_str)

            if len(tokens) > config.pad_size - 2:  # 大于最大长度进行截断
                tokens = tokens[0:(config.pad_size - 2)]
            # token to index
            tokens_c_s = '[CLS] ' + ''.join(tokens) + ' [SEP]'

            tokenized_text = config.tokenizer.tokenize(tokens_c_s)
            input_ids = config.tokenizer.convert_tokens_to_ids(tokenized_text)
            input_mask = [1] * len(input_ids)
            print(''.join(tokenized_text))

            if len(input_ids) < config.pad_size:
                input_mask = input_mask + ([0] * (config.pad_size - len(input_ids)))
                input_ids = input_ids + ([0] * (config.pad_size - len(input_ids)))

            output_mask = [1] * len(tokens)
            label_ids = [-1] * len(tokens)
            output_mask = [0] + output_mask + [0]
            label_ids = [-1] + label_ids + [-1]
            if len(output_mask) <  config.pad_size:
                label_ids += ([-1] * (config.pad_size - len(output_mask)))
                output_mask += ([0] * ( config.pad_size - len(output_mask)))


            input_ids_tensor = torch.LongTensor(input_ids).reshape(1, -1)
            input_mask_tensor = torch.LongTensor(input_mask).reshape(1, -1)
            output_mask_tensor = torch.LongTensor(output_mask).reshape(1, -1)
            label_ids_tensor = torch.LongTensor(label_ids).reshape(1, -1)

            input_ids_tensor = input_ids_tensor.to(config.device)
            input_mask_tensor = input_mask_tensor.to(config.device)
            output_mask_tensor = output_mask_tensor.to(config.device)
            label_ids_tensor = label_ids_tensor.to(config.device)
            try:
                assert len(input_ids_tensor[0]) == config.pad_size
                assert len(input_mask_tensor[0]) ==  config.pad_size
                assert len(label_ids_tensor[0]) ==  config.pad_size
                assert len(output_mask_tensor[0]) ==  config.pad_size
            except:
                return 'Error'
            #input_ids, input_mask, label_ids, output_mask
            #bert_encode = model(input_ids_tensor, input_mask_tensor)

            batch = input_ids_tensor, input_mask_tensor, label_ids_tensor, output_mask_tensor

            bert_encode = model(batch)

            predicts = model.predict(bert_encode, output_mask_tensor)
            predicts = predicts[predicts != -1]
            #print('paths:{}'.format(predicts))
            entitielables = []
            for p in predicts:
                for tag,val in config.vocab_class.items():
                    if p == val:
                        entitielables.append(tag)
            # print(entitielables)
            # print(tokens)
            # print(predicts)
            # print(len(entitielables))
            # print(len(tokens))
            # print(len(predicts))
            #提取出来实体
            entities=[]
            temp=''
            for i in range(len(entitielables)):
                if entitielables[i] == 'O' or entitielables[i] == 'Oc':
                    if temp != '':
                        entities.append((temp,entitielables[i-1].split('-')[1]))
                        temp=''
                    else:
                        continue
                else:
                    temp=temp+tokens[i]
            print(entities)
            return entities

def predict_e(model,config,input_str=""):
    model.eval()  # 取消batchnorm和dropout,用于评估阶段
    #model.to(config.device)
    #VOCAB = config.bert_path  # your path for model and vocab
    #tokenizer = BertTokenizer.from_pretrained(VOCAB)

    with torch.no_grad():
            #input_str = input("请输入文本: ")
            #input_ids = config.tokenizer.encode(input_str,add_special_tokens=True)  # add_spicial_tokens=True，为自动为sentence加上[CLS]和[SEP]
            #tokens = input_str.split()  # 分词
            input_str = clean(input_str)
            #tokens = [one for one in input_str]
            tokens = config.tokenizer.tokenize(input_str)

            if len(tokens) > config.pad_size - 2:  # 大于最大长度进行截断
                tokens = tokens[0:(config.pad_size - 2)]
            # token to index
            tokens_c_s = '[CLS] ' + ''.join(tokens) + ' [SEP]'

            tokenized_text = config.tokenizer.tokenize(tokens_c_s)
            input_ids = config.tokenizer.convert_tokens_to_ids(tokenized_text)
            input_mask = [1] * len(input_ids)
            print(''.join(tokenized_text))

            if len(input_ids) < config.pad_size:
                input_mask = input_mask + ([0] * (config.pad_size - len(input_ids)))
                input_ids = input_ids + ([0] * (config.pad_size - len(input_ids)))

            output_mask = [1] * len(tokens)
            label_ids = [-1] * len(tokens)
            output_mask = [0] + output_mask + [0]
            label_ids = [-1] + label_ids + [-1]
            if len(output_mask) <  config.pad_size:
                label_ids += ([-1] * (config.pad_size - len(output_mask)))
                output_mask += ([0] * ( config.pad_size - len(output_mask)))


            input_ids_tensor = torch.LongTensor(input_ids).reshape(1, -1)
            input_mask_tensor = torch.LongTensor(input_mask).reshape(1, -1)
            output_mask_tensor = torch.LongTensor(output_mask).reshape(1, -1)
            label_ids_tensor = torch.LongTensor(label_ids).reshape(1, -1)

            input_ids_tensor = input_ids_tensor.to(config.device)
            input_mask_tensor = input_mask_tensor.to(config.device)
            output_mask_tensor = output_mask_tensor.to(config.device)
            label_ids_tensor = label_ids_tensor.to(config.device)
            try:
                assert len(input_ids_tensor[0]) == config.pad_size
                assert len(input_mask_tensor[0]) ==  config.pad_size
                assert len(label_ids_tensor[0]) ==  config.pad_size
                assert len(output_mask_tensor[0]) ==  config.pad_size
            except:
                return 'Error'
            #input_ids, input_mask, label_ids, output_mask
            #bert_encode = model(input_ids_tensor, input_mask_tensor)

            batch = input_ids_tensor, input_mask_tensor, label_ids_tensor, output_mask_tensor

            bert_encode = model(batch)

            predicts = model.predict(bert_encode, output_mask_tensor)
            predicts = predicts[predicts != -1]
            #print('paths:{}'.format(predicts))
            entitielables = []
            for p in predicts:
                for tag,val in config.vocab_class.items():
                    if p == val:
                        entitielables.append(tag)
            # print(entitielables)
            # print(tokens)
            # print(predicts)
            # print(len(entitielables))
            # print(len(tokens))
            # print(len(predicts))
            #提取出来实体
            entities=[]
            temp=''
            for i in range(len(entitielables)):
                try:
                    if entitielables[i] == 'O' or entitielables[i] == 'Oc':
                        if temp != '':
                            entities.append((temp,entitielables[i-1].split('-')[1]))
                            temp=''
                        else:
                            continue
                    else:
                        temp=temp+tokens[i]
                except:
                    continue
            return entities