# coding: UTF-8
import time
import torch
import numpy as np
from torch import optim

from train_eval import train,predict,linetrain
from importlib import import_module
import argparse
from utils import built_train_dataset, built_dev_dataset, get_time_dif
import os
os.environ["CUDA_VISIBLE_DEVICES"] ='0'

parser = argparse.ArgumentParser(description='中文Ner—Pytorch版本')
parser.add_argument('--doing', type=str, required=True, help='choose a action: train,test,predict')
parser.add_argument('--model', type=str, required=True, help='choose a model: Bert,Albert,Xlnet,Gpt-2')
parser.add_argument('--input', type=str, required=False, help='choose a model: str')
args = parser.parse_args()


if __name__ == '__main__':

    model_name = args.model
    x = import_module('Models.' + model_name)
    config = x.Config()
    np.random.seed(1)
    torch.manual_seed(1)
    torch.cuda.manual_seed_all(1)
    torch.backends.cudnn.deterministic = True  # 保证每次结果一样



    if args.doing=='train':
        start_time = time.time()
        print("Loading Datas...")
        train_dataset = built_train_dataset(config)
        dev_dataset = built_dev_dataset(config)
        time_dif = get_time_dif(start_time)
        print("Time usage:", time_dif)
        model = x.Model(config).to(config.device)
        train(config, model, train_dataset, dev_dataset)
    if args.doing == 'linetrain':
        start_time = time.time()
        print("Loading Datas...")
        train_dataset = built_train_dataset(config)
        dev_dataset = built_dev_dataset(config)
        time_dif = get_time_dif(start_time)
        print("Time usage:", time_dif)
        model = x.Model(config).to(config.device)
        linetrain(config, model, train_dataset, dev_dataset)

    if args.doing=='predict':
        newmodel = x.Model(config).to(config.device)
        #加载
        # print(newmodel.state_dict())
        # print('#'*200)
        newmodel.load_state_dict(torch.load(config.save_path))
        # print(newmodel.state_dict())
        # print('#' * 200)
        str1=args.input
        #str1='4.1.2中心站硬件一般包括传输接口、主机、打印机、UPS电源、投影仪或电视墙、网络交换机、服务器、防火墙和配套设备等。中心站均应采用当时主流技术的通用产品，并满足可靠性、开放性和可维护性等要求。\n'
        predict(newmodel,config,str1)
