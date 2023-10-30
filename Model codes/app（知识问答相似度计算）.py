#! -*- coding: utf-8 -*-
import numpy as np
from flask import Flask, render_template, request, jsonify
from neo_db.query_graph import query,get_KGQA_answer,get_answer_profile
app = Flask(__name__)
import pandas as pd
from bert4keras.backend import keras
from bert4keras.models import build_transformer_model
from bert4keras.tokenizers import Tokenizer
from bert4keras.snippets import sequence_padding
from keras.models import Model
import operator
import tensorflow as tf



from tensorflow.compat.v1 import ConfigProto
from tensorflow.compat.v1 import InteractiveSession
config = ConfigProto()
config.gpu_options.allow_growth = True
session = InteractiveSession(config=config)


#graph = tf.get_default_graph()
graph = tf.compat.v1.get_default_graph()
sess =  keras.backend.get_session()

maxlen = 64

# bert配置
config_path = './chinese_roformer-sim-char-ft_L-12_H-768_A-12/bert_config.json'
checkpoint_path = './chinese_roformer-sim-char-ft_L-12_H-768_A-12/bert_model.ckpt'
dict_path = './chinese_roformer-sim-char-ft_L-12_H-768_A-12/vocab.txt'

# 建立分词器
tokenizer = Tokenizer(dict_path, do_lower_case=True)  # 建立分词器

# 建立加载模型
bert = build_transformer_model(
    config_path,
    checkpoint_path,
    model='roformer',
    with_pool='linear',
    application='unilm',
    return_keras_model=False,
)

encoder = keras.models.Model(bert.model.inputs, bert.model.outputs[0])

#防止错误
texts1 = ['测试测试1！', '测试测试2！']
X, S = [], []
for t in texts1:
    x, s = tokenizer.encode(t, maxlen=maxlen)
    X.append(x)
    S.append(s)
X = sequence_padding(X)
S = sequence_padding(S)

Z = encoder.predict([X, S])
Z /= (Z**2).sum(axis=1, keepdims=True)**0.5
print ((Z[0] * Z[1]).sum())


pd_question = pd.read_csv('./data_q/questionData.csv')
dict_question={}
list_question=[]
for i,r in pd_question.iterrows():
    if r['question'] not in dict_question:
        dict_question[r['question']] = r['answer']
        list_question.append(r['question'])

def similarity(text1, text2):
    """"计算text1与text2的相似度
    """
    global tokenizer,encoder,graph,sess

    # from tensorflow.compat.v1 import ConfigProto
    # from tensorflow.compat.v1 import InteractiveSession
    # config = ConfigProto()
    # config.gpu_options.allow_growth = True

    texts = [text1, text2]
    X, S = [], []
    for t in texts:
        x, s = tokenizer.encode(t, maxlen=maxlen)
        X.append(x)
        S.append(s)
    X = sequence_padding(X)
    S = sequence_padding(S)
    with sess.as_default():
        with graph.as_default():
            Z = encoder.predict([X, S])
    Z /= (Z**2).sum(axis=1, keepdims=True)**0.5
    return (Z[0] * Z[1]).sum()


def muti_similarity(text1, textlist):
    res_dict={}
    for text in textlist:
        if text not in res_dict:
            tempscores = similarity(text1,text)
            res_dict[text]=tempscores
            if tempscores>=0.95:
                break
    res_dict = sorted(res_dict.items(), key=operator.itemgetter(1), reverse=True)

    return res_dict

def AI_answer(question):
    similarity_dict = muti_similarity(question, list_question)
    if similarity_dict[0][1]>=0.85:
        #获取问题答案
        return {'ok':'success','res':dict_question[similarity_dict[0][0]]}
    else:
        return {'ok':'fail','res':'您的问题我暂时无法回答，会继续努力的！'}


@app.route('/', methods=['GET', 'POST'])
@app.route('/index', methods=['GET', 'POST'])

def index(name=None):
    return render_template('index.html', name = name)


@app.route('/search', methods=['GET', 'POST'])
def search():
    return render_template('search.html')


@app.route('/KGQA', methods=['GET', 'POST'])
def KGQA():
    return render_template('KGQA.html')

@app.route('/get_profile',methods=['GET','POST'])
def get_profile():
    name = request.args.get('character_name')
    json_data = get_answer_profile(name)
    return jsonify(json_data)

@app.route('/KGQA_answer', methods=['GET', 'POST'])
def KGQA_answer():
    question = request.args.get('name')
    print(question)
    anser = AI_answer(question)
    res = get_answer_profile(anser['ok'])
    json_data={'answer':anser['res'],'pic':res}
    return jsonify(json_data)

@app.route('/search_name', methods=['GET', 'POST'])
def search_name():
    name = request.args.get('name')
    json_data=query(str(name))
    return jsonify(json_data)

@app.route('/get_all_relation', methods=['GET', 'POST'])
def get_all_relation():
    return render_template('all_relation.html')

e_Dict = {'Facility': 0, 'Materials': 1, 'Atmospheric': 2, 'Geology': 3, 'Person': 4,
                    'Project': 5, 'Task': 6, 'Index': 7, 'Place': 8,
                    'Text': 9, 'Management': 10, 'Method': 11}

if __name__ == '__main__':
    # showmax_num=1500
    # jsontext = {'data': [],'links':[]}
    # #生成数据json
    # e_pd = pd.read_csv('./data/entities.csv',encoding='utf-8')
    # r_pd = pd.read_csv('./data/relationship.csv',encoding='utf-8')
    # num=0

    app.debug=True
    app.run()
