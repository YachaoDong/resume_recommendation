# -*- coding: utf-8 -*-
'''
@Time    : 7/15/2022 11:14 AM
@Author  : dong.yachao
'''
import numpy as np
import gensim
from gensim.models.doc2vec import Doc2Vec, TaggedDocument
# from gensim.models import Doc2Vec

import jieba as jb
import os
import pandas as pd
from joblib import Parallel, delayed
from tqdm import tqdm
import time
import random
import json



# 读取停词表 添加不同形式的 空格
def get_stop_words(stop_words_path=r'D:\CodeFiles\data\stopwords-master\baidu_stopwords.txt'):
    stop_words = [i.strip() for i in open(stop_words_path, 'r', encoding='utf8').readlines()]
    stop_words.extend(['\n', '\xa0', '\u3000', '\u2002'])
    return stop_words

# 对字符串x进行分词 并使用 " " 连接，返回的是分词后的字符串
def get_str(sentence, stop_words=None):
    if stop_words is None:
        stop_words = ['\n', '\xa0', '\u3000', '\u2002']
    return ' '.join([i for i in jb.cut(sentence) if i not in stop_words])



def doc2vec_infer(model, doc_text, alpha=0.01, epochs=100):
    model.random.seed(0)
    doc_vec = model.infer_vector(doc_text.split(' '), alpha=alpha, epochs=epochs)
    # 如果doc2vec 转换的向量为空或者不完全，使用-0.1填充
    if doc_vec.size != 20:
        doc_vec = np.full(20, -0.1)
    return doc_vec



if __name__ == '__main__':
    train_data_path = r'D:\CodeFiles\data\recruitment_data'
    jd_df = pd.read_csv(os.path.join(train_data_path, 'table2_jd.csv'), sep='\t')
    user_df = pd.read_csv(os.path.join(train_data_path, 'table1_user.csv'))
    stop_words = get_stop_words(stop_words_path=r'D:\CodeFiles\data\stopwords-master\baidu_stopwords.txt')

    # jd
    # [[以空格分词]...]
    job_description = Parallel(n_jobs=-1)(delayed(get_str)(jd_df.loc[ind]['job_description\n'], stop_words=stop_words)
                                  for ind in tqdm(jd_df.index))
    # print('job_description:', job_description)
    jd_df['jd_word'] = job_description

    user_df['experience'] = user_df['experience'].apply(lambda x: ' '.join(x.split('|')
                                                                           if isinstance(x, str) else 'nan'))
    experience = list(user_df['experience'].values)

    # print('job_description:', job_description)
    # print('experience:', experience)

    # random.shuffle(job_description)
    # random.shuffle(experience)
    all_words = job_description + experience
    random.shuffle(all_words)
    # save to json
    jd_exp_doc = {'job_description': job_description, 'experience': experience, 'all': all_words}
    with open(r"D:\CodeFiles\AI4recruitment\ai_recruitment\models\exp0\jd_exp_doc.json", "w", encoding="utf-8") as f1:
        # indent参数保证json数据的缩进，美观
        # ensure_ascii=False才能输出中文，否则就是Unicode字符
        json.dump(jd_exp_doc, f1, indent=2, ensure_ascii=False)

    with open(r"D:\CodeFiles\AI4recruitment\ai_recruitment\models\exp0\train_corpus.txt", "w", encoding="utf-8") as f2:
        for line in all_words:
            f2.write(line + "\n")

    # 根据TaggedDocumnet生成训练语料
    print('gen doc..')
    documents = [TaggedDocument(doc, [i]) for i, doc in enumerate(all_words)]
    model = Doc2Vec(documents, dm=1, alpha=0.1, vector_size=20, min_alpha=0.025)
    print('training..')
    model.train(documents, total_examples=model.corpus_count, epochs=100)
    model.save('./models/d2v_model')


    # test
    # 模型加载
    # model_dm = Doc2Vec.load("./models/d2vmodel")
    # # 模型预测
    # test_text = ['独立', '工程', '预算', '编制']
    # start_time = time.time()
    # # model_dm.random.seed(0)
    # inferred_vector_dm_1 = model_dm.infer_vector(test_text, alpha=0.01, epochs=100)
    # inferred_vector_dm_2 = model_dm.infer_vector(test_text, alpha=0.01, epochs=100)
    # print('infer time:', (time.time()-start_time)/2)
    # print('inferred_vector_dm_1:', inferred_vector_dm_1)
    # print('inferred_vector_dm_2:', inferred_vector_dm_2)
    #
    # sims = model_dm.dv.most_similar([inferred_vector_dm], topn=10)
    # print('sims:', sims)
    #
    # # print('test_text:', test_text)
    # # for raw_index, sim in sims:
    # #     sentence = documents[raw_index]
    # #     print(sentence, sim, len(sentence[0]))


