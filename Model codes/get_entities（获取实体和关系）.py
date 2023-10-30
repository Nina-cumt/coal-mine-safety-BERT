# coding: UTF-8
import operator
import re
import json
import datetime
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn import metrics
import time
from sklearn.metrics import f1_score, classification_report
import pandas as pd
from pytorch_pretrained import BertTokenizer
from pytorch_pretrained.optimization import *
from utils import *
from tqdm import tqdm
import jieba
from importlib import import_module
from train_eval import predict_e

model_name='bert'
os.environ["CUDA_VISIBLE_DEVICES"] ='0'
x = import_module('Models.' + model_name)
config = x.Config()
np.random.seed(1)
torch.manual_seed(1)
torch.cuda.manual_seed_all(1)
torch.backends.cudnn.deterministic = True  # 保证每次结果一样


#定义关系字典
Use=['使用','操纵','利用','运用','行使','应用','研究']
Implement=['实施','执行','实行','施行','履行','推行']
Document=['制定','编制','编写','编辑']
Examine=['审核','审批','稽核']
Establish=['建立','创办','创立','建设','开发','设立','成立']
Equip=['装配','装置','安装','装载','装备','拥有','有']
Include=['包含','包括','容纳','含有']
Distance=['间距','间隙','间隔']
Measure=['测定','检测','测量','探测']
ALL = Use+Implement+Document+Examine+Establish+Equip+Include+Distance+Measure
#构建关系分类

RelationshipDict={'使用':Use,'任务':Implement,'编制':Document,'审批':Examine,'生产':Establish,
                  '装备':Equip,'包含':Include,'间距':Distance,'测定':Measure}


def get_labeld_e():
    train_content_path = './data/train/source.txt'
    train_label_path = './data/train/target.txt'
    dev_content_path = './data/dev/source.txt'
    dev_label_path = './data/dev/target.txt'
    entities_dict = {}
    #测试集合
    with open(dev_content_path, 'r', encoding='utf-8') as df_de:
        with open(dev_label_path, 'r', encoding='utf-8') as lf_de:
            dev_data = df_de.readlines()
            dev_label = lf_de.readlines()
            for word, label in zip(dev_data, dev_label):
                #每一行
                tokens = word.split()
                label = label.split()
                temp = ''
                for i in range(len(label)):
                    if label[i] == 'O' or label[i] == 'Oc':
                        if temp != '':
                            if temp not in entities_dict:
                                try:
                                    entities_dict[temp]= label[i - 1].split('-')[1]
                                except:
                                    print(label[i - 1])
                                temp = ''
                        else:
                            continue
                    else:
                        temp = temp + tokens[i]
    #训练集合
    with open(train_content_path, 'r', encoding='utf-8') as df_train:
        with open(train_label_path, 'r', encoding='utf-8') as lf_train:
            train_data = df_train.readlines()
            train_label = lf_train.readlines()
            for word1, label1 in zip(train_data, train_label):
                # 每一行
                tokens1 = word1.split()
                label1 = label1.split()
                temp1 = ''
                for j in range(len(label1)):
                    if label1[j] == 'O' or label1[j] == 'Oc':
                        if temp1 != '':
                            if temp1 not in entities_dict:
                                try:
                                    entities_dict[temp1] = label1[j - 1].split('-')[1]
                                except:
                                    print(label1[j - 1])
                                temp1 = ''
                        else:
                            continue
                    else:
                        temp1 = temp1 + tokens1[j]
    print(entities_dict)
    return entities_dict


def save_cibiao(fencidict):
    with open('./data/fencidict.txt', 'w', encoding='utf-8') as f:
        for k, v in fencidict.items():
            f.write(k + '\n')
    print('自定义分词表保存完毕！')

def file_name(dirr):
    for root, dirs, files in os.walk(dirr):
        # print('root_dir:', root)  # 当前目录路径
        # print('sub_dirs:', dirs)  # 当前路径下所有子目录
        # print('files:', files)  # 当前路径下所有非目录子文件
        break
    return dirs,files



def clean(text):
    # text = re.sub(r"(回复)?(//)?\s*@\S*?\s*(:| |$)", " ", text)  # 去除正文中的@和回复/转发中的用户名
    # text = re.sub(r"\[\S+\]", "", text)  # 去除表情符号
    # text = re.sub(r"#\S+#", "", text)  # 保留话题内容
    text = re.sub(r"#\S+#", "", text)  # 去除话题
    text = re.sub(r"\s+", " ", text)  # 合并正文中过多的空格
    text = text.replace(" ", "")  # 去除无意义的词语
    text = text.replace("\n", "")  # 去除无意义的词语
    return text.strip()


def save_entities(entities_dict):
    pd1= pd.DataFrame(columns=[':label','title'])
    i=0
    for k, v in entities_dict.items():
        pd1.loc[i,':label'] = v
        pd1.loc[i, 'title'] = k
        i=i+1
    pd1.to_csv('./data/entities.csv',index_label=':id')


def get_guanxi():
    #读取实体列表,转换为dict
    e_id_dict={}
    pd1 = pd.read_csv('./data/entities.csv')
    shitilist = []
    for i,r in pd1.iterrows():
        e_id_dict[r['title']]=r[':id']
        shitilist.append(r['title'])
    #读取文本
    _ ,train_files = file_name('./data/handle/train_set/')
    _, other_files = file_name('./data/handle/another/')
    #加载自定义
    jieba.load_userdict('./data/fencidict.txt')
    #加载关系词
    for a in ALL:
        jieba.add_word(a)
    #创建实体和关系词全组
    allkeywordlist = list(set(shitilist+ALL))
    #创建pd，用于保存关系表
    pd2= pd.DataFrame(columns=[':START_ID',':END_ID',':TYPE'])
    hang=0
    #提取关系，训练集
    for tfiles in  tqdm(train_files):
        #每个文档
        file_object = open('./data/handle/train_set/'+tfiles, 'r', encoding='utf-8')
        try:
            all_the_text = file_object.read()
            all_the_text = clean(all_the_text)
        except:
            print(tfiles + '编码错误！')
            continue
        finally:
            file_object.close()
        #分句
        all_the_text_list = all_the_text.split('\\n')
        #对分句进行分词

        for line in all_the_text_list:
            # 每个文档的每句话
            cleanlist = []
            fencilist_line = jieba.lcut(line)
            #print(fencilist_line)
            #只保留实体和关系词
            for l in fencilist_line:
                if l in allkeywordlist:
                    cleanlist.append(l)
            #提取关系
            num = len(cleanlist)
            if num>=2:
                new_e = ''
                for m in range(num):
                    #遍历找出最新的实体
                    if cleanlist[m] in shitilist:
                        #提取值关系
                        if m+1<num:
                            if cleanlist[m+1] in shitilist:
                                #print(cleanlist[m], '取值', cleanlist[m+1])
                                pd2.loc[hang, ':START_ID'] = e_id_dict[cleanlist[m]]
                                pd2.loc[hang, ':END_ID'] = e_id_dict[cleanlist[m+1]]
                                pd2.loc[hang, ':TYPE'] = '取值'
                                hang = hang + 1
                        new_e = cleanlist[m]
                        continue
                    #如果找到的关系词，#除了取值以外的
                    if cleanlist[m] in ALL:
                        start_e = new_e
                        #找出下一个实体
                        end_e=''
                        for mm in range(m, num):
                            if cleanlist[mm] in shitilist:
                                end_e = cleanlist[mm]
                                break
                        #判断找到开始实体和结束实体没有
                        if start_e!='' and end_e !='':
                            #print(start_e,cleanlist[m],end_e)
                            #判断关系属于哪种
                            for k,v in RelationshipDict.items():
                                if cleanlist[m] in v:
                                    pd2.loc[hang,':START_ID']=e_id_dict[start_e]
                                    pd2.loc[hang, ':END_ID'] = e_id_dict[end_e]
                                    pd2.loc[hang, ':TYPE'] = k
                                    hang=hang+1
                                    break

    # 提取关系，其他集
    for ofiles in  tqdm(other_files):
        # 每个文档
        file_object = open('./data/handle/another/' + ofiles, 'r', encoding='utf-8')
        try:
            all_the_text = file_object.read()
            all_the_text = clean(all_the_text)
        except:
            print(ofiles+'编码错误！')
            continue
        finally:
            file_object.close()
        # 分句
        all_the_text_list = all_the_text.split('\\n')
        # 对分句进行分词

        for line in all_the_text_list:
            # 每个文档的每句话
            cleanlist = []
            fencilist_line = jieba.lcut(line)
            # print(fencilist_line)
            # 只保留实体和关系词
            for l in fencilist_line:
                if l in allkeywordlist:
                    cleanlist.append(l)
            # 提取关系
            num = len(cleanlist)
            if num >= 2:
                new_e = ''
                for m in range(num):
                    # 遍历找出最新的实体
                    if cleanlist[m] in shitilist:
                        # 提取值关系
                        if m + 1 < num:
                            if cleanlist[m + 1] in shitilist:
                                #print(cleanlist[m], '取值', cleanlist[m + 1])
                                pd2.loc[hang, ':START_ID'] = e_id_dict[cleanlist[m]]
                                pd2.loc[hang, ':END_ID'] = e_id_dict[cleanlist[m + 1]]
                                pd2.loc[hang, ':TYPE'] = '取值'
                                hang = hang + 1
                        new_e = cleanlist[m]
                        continue
                    # 如果找到的关系词，#除了取值以外的
                    if cleanlist[m] in ALL:
                        start_e = new_e
                        # 找出下一个实体
                        end_e = ''
                        for mm in range(m, num):
                            if cleanlist[mm] in shitilist:
                                end_e = cleanlist[mm]
                                break
                        # 判断找到开始实体和结束实体没有
                        if start_e != '' and end_e != '':
                            #print(start_e, cleanlist[m], end_e)
                            # 判断关系属于哪种
                            for k, v in RelationshipDict.items():
                                if cleanlist[m] in v:
                                    pd2.loc[hang, ':START_ID'] = e_id_dict[start_e]
                                    pd2.loc[hang, ':END_ID'] = e_id_dict[end_e]
                                    pd2.loc[hang, ':TYPE'] = k
                                    hang = hang + 1
                                    break
    #保存
    pd2.to_csv('./data/relationship_dirty.csv',index=False)
    print('保存关系脏数据成功！')

#清洗关系
def clean_guanxi():
    e_label_dict = {}
    e_name_dict={}
    pd1 = pd.read_csv('./data/entities.csv')
    #读取实体数据
    for i,r in pd1.iterrows():
        e_label_dict[r[':id']]=r[':label']
        e_name_dict[r[':id']] = r['title']
    #创建pd，用于保存关系表
    df2= pd.DataFrame(columns=[':START_ID',':END_ID',':TYPE'])
    hang=0

    df1 = pd.read_csv('./data/relationship_dirty.csv')
    #清洗
    for i ,r in tqdm(df1.iterrows()):
        #实体相同的直接跳过
        if r[':START_ID']==r[':END_ID']:
            continue
        #取值关系判断
        if r[':TYPE']=='取值':
            quzhilist=['Task','Place','Project','Person','Atmospheric']
            if (e_label_dict[ r[':START_ID'] ] in quzhilist ) and (e_label_dict[ r[':END_ID'] ] == 'Index'):
                #添加
                df2.loc[hang, ':START_ID'] = r[':START_ID']
                df2.loc[hang, ':END_ID'] = r[':END_ID']
                df2.loc[hang, ':TYPE'] = r[':TYPE']
                hang = hang + 1
        # 使用关系判断
        elif r[':TYPE']=='使用':
            shiyong_t=[('Person','Facility'),('Person','Materials'),('Person','Method'),('Task','Facility'),
                       ('Task','Materials'),('Task','Method'),('Place','Facility'),('Place','Method'),
                       ('Facility','Materials'),('Place','Materials'),('Geology','Method'),('Project','Method'),
                       ('Project','Materials')]
            for t in shiyong_t:
                if operator.eq(t, (e_label_dict[ r[':START_ID'] ],e_label_dict[ r[':END_ID'] ])):
                    # 添加
                    df2.loc[hang, ':START_ID'] = r[':START_ID']
                    df2.loc[hang, ':END_ID'] = r[':END_ID']
                    df2.loc[hang, ':TYPE'] = r[':TYPE']
                    hang = hang + 1
                    #print((e_name_dict[ r[':START_ID'] ],e_name_dict[ r[':END_ID'] ]))
                    break
        # 任务关系判断
        elif r[':TYPE'] == '任务':
            quzhilist = ['Person','Method', 'Place', 'Facility',  'Geology']
            if (e_label_dict[r[':START_ID']] in quzhilist) and (e_label_dict[r[':END_ID']] == 'Task'):
                # 添加
                df2.loc[hang, ':START_ID'] = r[':START_ID']
                df2.loc[hang, ':END_ID'] = r[':END_ID']
                df2.loc[hang, ':TYPE'] = r[':TYPE']
                hang = hang + 1
        # 编制关系判断
        elif r[':TYPE'] == '编制':
            quzhilist = ['Person', 'Task', 'Place', 'Facility', 'Geology','Method','Project']
            if (e_label_dict[r[':START_ID']] in quzhilist) and (e_label_dict[r[':END_ID']] == 'Text'):
                # 添加
                df2.loc[hang, ':START_ID'] = r[':START_ID']
                df2.loc[hang, ':END_ID'] = r[':END_ID']
                df2.loc[hang, ':TYPE'] = r[':TYPE']
                hang = hang + 1

        # 审批关系判断
        elif r[':TYPE'] == '审批':
            quzhilist = ['Person']
            if (e_label_dict[r[':START_ID']] in quzhilist) and (e_label_dict[r[':END_ID']] == 'Text'):
                # 添加
                df2.loc[hang, ':START_ID'] = r[':START_ID']
                df2.loc[hang, ':END_ID'] = r[':END_ID']
                df2.loc[hang, ':TYPE'] = r[':TYPE']
                hang = hang + 1
        # 生产关系判断
        elif r[':TYPE']=='生产':
            shiyong_t=[('Geology','Project'),('Geology','Management'),('Person','Project'),('Person','Management'),
                       ('Place','Project'),('Task','Management')]
            for t in shiyong_t:
                if operator.eq(t, (e_label_dict[ r[':START_ID'] ],e_label_dict[ r[':END_ID'] ])):
                    # 添加
                    df2.loc[hang, ':START_ID'] = r[':START_ID']
                    df2.loc[hang, ':END_ID'] = r[':END_ID']
                    df2.loc[hang, ':TYPE'] = r[':TYPE']
                    hang = hang + 1
                    break
        # 装备关系判断
        elif r[':TYPE']=='装备':
            shiyong_t=[('Task','Person'),('Task','Facility'),('Person','Facility'),('Person','Person'),
                       ('Facility','Facility'),('Place','Facility'),('Project','Person'),('Project','Facility')]
            for t in shiyong_t:
                if operator.eq(t, (e_label_dict[ r[':START_ID'] ],e_label_dict[ r[':END_ID'] ])):
                    # 添加
                    df2.loc[hang, ':START_ID'] = r[':START_ID']
                    df2.loc[hang, ':END_ID'] = r[':END_ID']
                    df2.loc[hang, ':TYPE'] = r[':TYPE']
                    hang = hang + 1
                    break
        # 包含关系判断
        elif r[':TYPE'] == '包含':
            shiyong_t = [('Method', 'Method'), ('Geology', 'Geology'), ('Geology', 'Place'), ('Place', 'Place'),
                         ('Task', 'Task'), ('Method', 'Task')]
            for t in shiyong_t:
                if operator.eq(t, (e_label_dict[r[':START_ID']], e_label_dict[r[':END_ID']])):
                    # 添加
                    df2.loc[hang, ':START_ID'] = r[':START_ID']
                    df2.loc[hang, ':END_ID'] = r[':END_ID']
                    df2.loc[hang, ':TYPE'] = r[':TYPE']
                    hang = hang + 1
                    break

        # 间距关系判断
        elif r[':TYPE'] == '间距':
            shiyong_t = [('Facility', 'Place'), ('Facility', 'Facility'), ('Place', 'Place')]
            for t in shiyong_t:
                if operator.eq(t, (e_label_dict[r[':START_ID']], e_label_dict[r[':END_ID']])):
                    # 添加
                    df2.loc[hang, ':START_ID'] = r[':START_ID']
                    df2.loc[hang, ':END_ID'] = r[':END_ID']
                    df2.loc[hang, ':TYPE'] = r[':TYPE']
                    hang = hang + 1
                    break

        # 测定关系判断
        elif r[':TYPE'] == '测定':
            shiyong_t = [('Place', 'Atmospheric'), ('Task', 'Atmospheric')]
            for t in shiyong_t:
                if operator.eq(t, (e_label_dict[r[':START_ID']], e_label_dict[r[':END_ID']])):
                    # 添加
                    df2.loc[hang, ':START_ID'] = r[':START_ID']
                    df2.loc[hang, ':END_ID'] = r[':END_ID']
                    df2.loc[hang, ':TYPE'] = r[':TYPE']
                    hang = hang + 1
                    break
    #保存
    #去重复
    df2= df2.drop_duplicates()
    df2.to_csv('./data/relationship.csv',index=False)
    print('清洗保存关系数据成功！')


def get_labeld_unlabel():
    unlabel_entities_dict = {}
    #打开未标记的数据
    _, other_files = file_name('./data/handle/another/')
    for ofiles in  tqdm(other_files):
        # 每个文档
        file_object = open('./data/handle/another/' + ofiles, 'r', encoding='utf-8')
        try:
            all_the_text = file_object.read()
            all_the_text = clean(all_the_text)
        except:
            print(ofiles + '编码错误！')
            continue
        finally:
            file_object.close()
        # 分句
        all_the_text_list = all_the_text.split('\\n')
        #对每个分句实体识别
        for str1 in all_the_text_list:
            newmodel = x.Model(config).to(config.device)
            newmodel.load_state_dict(torch.load(config.save_path))
            entities = predict_e(newmodel, config, str1)
            #清洗实体
            entities = clean_e(entities)
            #print(entities)
            for ee,v in entities.items():
                if ee not in unlabel_entities_dict:
                    unlabel_entities_dict[ee]=v
    print(unlabel_entities_dict)
    return unlabel_entities_dict

def clean_e(entities):
    newdict={}
    try:
        for e ,v in entities:
            e = re.sub('[*<>《》（）()、。，：\\n]', '', e)
            if len(e)>=2 and e not in newdict:
                newdict[e]=v
    except Exception as ee:
        print(ee)
        print(entities)
        pass
    return newdict
if __name__ == '__main__':
    #从标记过得数据中提取实体
    entities_dict_labled = get_labeld_e()
    # 从未标记数据中提取实体
    unlabel_entities_dict = get_labeld_unlabel()
    # 合并实体并保存
    dictMerged = entities_dict_labled.copy()
    dictMerged.update(unlabel_entities_dict)
    #保存所有的实体到csv
    save_entities(dictMerged)
    #保存实体当做自定义分词表：
    save_cibiao(dictMerged)
    #根据实体抽取关系，存脏数据
    get_guanxi()
    #清洗关系，检测是否符合要求
    clean_guanxi()
