#! -*- coding: utf-8 -*-
# RoFormer-Sim base 基本例子
# 测试环境：tensorflow 1.14 + keras 2.3.1 + bert4keras 0.10.6

import numpy as np
from bert4keras.backend import keras, K
from bert4keras.models import build_transformer_model
from bert4keras.tokenizers import Tokenizer
from bert4keras.snippets import sequence_padding, AutoRegressiveDecoder
#from bert4keras.snippets import uniout
import pandas as pd
from tqdm import tqdm

maxlen = 64

# 模型配置
config_path = './chinese_roformer-sim-char-ft_L-12_H-768_A-12/bert_config.json'
checkpoint_path = './chinese_roformer-sim-char-ft_L-12_H-768_A-12/bert_model.ckpt'
dict_path = './chinese_roformer-sim-char-ft_L-12_H-768_A-12/vocab.txt'

# 建立分词器
tokenizer = Tokenizer(dict_path, do_lower_case=True)  # 建立分词器

# 建立加载模型
roformer = build_transformer_model(
    config_path,
    checkpoint_path,
    model='roformer',
    application='unilm',
    with_pool='linear'
)

encoder = keras.models.Model(roformer.inputs, roformer.outputs[0])
seq2seq = keras.models.Model(roformer.inputs, roformer.outputs[1])


class SynonymsGenerator(AutoRegressiveDecoder):
    """seq2seq解码器
    """
    @AutoRegressiveDecoder.wraps(default_rtype='probas')
    def predict(self, inputs, output_ids, step):
        token_ids, segment_ids = inputs
        token_ids = np.concatenate([token_ids, output_ids], 1)
        segment_ids = np.concatenate([segment_ids, np.ones_like(output_ids)], 1)
        return self.last_token(seq2seq).predict([token_ids, segment_ids])

    def generate(self, text, n=1, topp=0.95, mask_idxs=[]):
        token_ids, segment_ids = tokenizer.encode(text, maxlen=maxlen)
        for i in mask_idxs:
            token_ids[i] = tokenizer._token_mask_id
        output_ids = self.random_sample([token_ids, segment_ids], n,
                                        topp=topp)  # 基于随机采样
        return [tokenizer.decode(ids) for ids in output_ids]


synonyms_generator = SynonymsGenerator(
    start_id=None, end_id=tokenizer._token_end_id, maxlen=maxlen
)


def gen_synonyms(text, n=100, k=20, mask_idxs=[]):
    ''''含义： 产生sent的n个相似句，然后返回最相似的k个。
    做法：用seq2seq生成，并用encoder算相似度并排序。
    '''
    global synonyms_generator
    r = synonyms_generator.generate(text, n, mask_idxs=mask_idxs)
    r = [i for i in set(r) if i != text]
    r = [text] + r
    X, S = [], []
    for t in r:
        x, s = tokenizer.encode(t)
        X.append(x)
        S.append(s)
    X = sequence_padding(X)
    S = sequence_padding(S)
    Z = encoder.predict([X, S])
    Z /= (Z**2).sum(axis=1, keepdims=True)**0.5
    argsort = np.dot(Z[1:], -Z[0]).argsort()
    return [r[i + 1] for i in argsort[:k]]

def creat_data(path1,path2):
    pd1=pd.read_excel(path1)
    newpd = pd.DataFrame()
    hang = 0
    #pd1=pd1.head(5)
    for i,r in tqdm(pd1.iterrows()):
        text = r['问题']
        lable = r['问题答案']
        newpd.loc[hang, 'question'] = text
        newpd.loc[hang, 'answer'] = lable
        hang = hang + 1
        #数据增强
        res = gen_synonyms(text, n=50, k=5)
        print(res)
        for line in res:
            newpd.loc[hang, 'question'] = line
            newpd.loc[hang, 'answer'] = lable
            hang = hang + 1
    #合并
    pd2=pd.read_excel(path2)
    newpd = pd.concat([newpd,pd2], axis=0)
    newpd = newpd.sample(frac=1).reset_index(drop=True)
    newpd.to_csv('./data_q/questionData.csv',index=False,encoding='utf-8-sig')

if __name__ == '__main__':
    path1='./data_q/wenti.xlsx'
    path2 = './data_q/jieshi.xlsx'
    creat_data(path1,path2)

