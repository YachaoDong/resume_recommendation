# # -*- coding: utf-8 -*-
# '''
# @Time    : 7/15/2022 7:30 PM
# @Author  : dong.yachao
# '''
#
# import pandas as pd
# import os
#
# #
# # train_data_path = r'D:\CodeFiles\data\recruitment_data'
# # jd_df = pd.read_csv(os.path.join(train_data_path, 'table2_jd.csv'), sep='\t')
# # user_df = pd.read_csv(os.path.join(train_data_path, 'table1_user.csv'))
# #
# #
# # user_df['experience'] = user_df['experience'].apply(lambda x: ' '.join(x.split('|')
# #                                                                        if isinstance(x, str) else 'nan'))
# # experience = user_df['experience']
# # print(type(experience))
# # print(experience)
# #
# # print(type(experience.values))
# # print(experience.values)
# # print(len(experience.values))
# # print(list(experience.values))
# # import numpy as np
# # from sklearn.metrics.pairwise import cosine_similarity, paired_distances
# #
# # x = np.array([[0.26304135, 0.91725843, 0.61099966, 0.40816231, 0.93606288, 0.52462691], [0.26304135, 0.91725843, 0.61099966, 0.40816231, 0.93606288, 0.52462691]])
# # print(x)
# # y = np.array([[0.03756129, 0.50223667, 0.66529424, 0.57392135, 0.20479857, 0.27286363], [0.26304135, 0.91725843, 0.61099966, 0.40816231, 0.93606288, 0.52462691]])
# # print(y)
# # # 余弦相似度
# # simi = cosine_similarity(x, y)
# # print('cosine similarity:', np.diagonal(simi))
#
# from gensim.models import Doc2Vec
# import gensim
# import json
# # test
# # 模型加载
# model_dm = Doc2Vec.load("D:\CodeFiles\AI4recruitment\\ai_recruitment\models\d2v_model")
# with open(r"D:\CodeFiles\AI4recruitment\ai_recruitment\models\job_description_cut.json", "r",
#           encoding="utf-8") as f1:
#     tmp_cut = list(json.load(f1)['tmp_cut'])
#
# # sentence = tmp_cut[0].split(' ')
#
#
# # 模型预测
# f2 = open(r"D:\CodeFiles\AI4recruitment\ai_recruitment\models\train_corpus.txt", "r",
#           encoding="utf-8")
# corpus = f2.readlines()
# documents = [gensim.models.doc2vec.TaggedDocument(doc, [i]) for i, doc in enumerate(corpus)]
#
# # documents = gensim.models.doc2vec.TaggedLineDocument(r"D:\CodeFiles\AI4recruitment\ai_recruitment\models\train_corpus.txt")
#
# search_sentence = corpus[100]
# print('search_sentence:', search_sentence)
# model_dm.random.seed(0)
# vec_1 = model_dm.infer_vector(search_sentence.split(), alpha=0.01, epochs=100)
# model_dm.random.seed(0)
# vec_2 = model_dm.infer_vector([search_sentence], alpha=0.01, epochs=100)
#
# sims = model_dm.dv.most_similar([vec_1], topn=3)
# print('sims 1:', sims)
# for raw_index, sim in sims:
#     sentence = documents[raw_index]
#     print(sentence, sim, len(sentence[0]))
#
# sims = model_dm.dv.most_similar([vec_2], topn=3)
# print('sims 2:', sims)
# for raw_index, sim in sims:
#     sentence = documents[raw_index]
#     print(sentence, sim, len(sentence[0]))
#
# from sklearn.metrics.pairwise import cosine_similarity, paired_distances
# import numpy as np
# s1 = "相关 专业 毕业 实习生 优先 ， 免费 提供 住宿 ， 吃苦耐劳 ， 北京 、 河北 出差 ， 入 职后 缴纳 保险 。 1 、 甲方 沟通 , 甲方 独立 设计 出 工程 效果图 、 施工图 。 2 、 工地 情况 配合 项目经理 编写 施工 方案 、 施工 组织 设计 , 制定 施工进度 表 。 施工 中 : 3 、 施工图 汇制 , 应 标注 , 做法 详细 , 指导 施工 。 准确 提供 施工队 需 施工图 。 4 、 交底 记录 5 天内 提出 优化 后 施工 方案 ( 注 : 提升 整体 效果 、 利旧 、 降低成本 、 控制 经济指标 ) 5 、 施工图 , 现场 放线 排尺 , 做好 放线 记录 。 6 、 配合 项目经理 放线 情况 甲方 讲解 、 沟通 , 甲方 确认 。 做好 施工图 变更 。 7 、 现场 放线 排尺 情况 交底 施工队 正确 施工 。 8 、 现场 放线 排尺 提 材料 计划 。 材料 计划 准确 无 浪费 , 保证 材料 利用 最大化 。 9 、 配合 采购 人员 选 材料 样品 、 定 规格 。 选择 材料 样品 时 , 保证 施工 效果 、 质量 前提 下 , 成本 因素 。 10 、 配合 项目经理 选定 材料 样品 甲方 讲解 , 征得 甲方 同意 。 11 、 效果图 、 施工图 、 水电 施工图 、 平 、 立 、 剖面图 、 竣工 图 绘制 应 详细 , 应 保证 各图 之间 相互 、 统一 , 保证 决算 顺利进行 。 12 、 工程 竣工 时 竣工 图 绘制 , 审计 状态 。 13 、 施工图 需 至少 绘制 2 套 , 一套 绘制 , 以备 维修 提供 参考 ; 一套 审计 图纸 , 预算 修改 , 配合 预算 做好 决算 , 出 好 洽商 需 图纸 , 决算 需 修改 图纸 预算 期限内 , 材料 、 工艺 做法 标注 预算 提前 沟通 。"
# s2 = '任职 ： 25 - 38 岁 ， 大专 以上学历 ， 沟通 抗压 能力 ， 条件 优秀者 放宽 条件 。 工作 时间 ： 8 小时 / 天 ， 上 6 休 1 。 薪资 待遇 ： 缴纳 五险 一金 ， 税后 4000 元 / 月 。 工作 地点 ： 浦口 桥北 地区 。'
#
# s3 = '  甲方 沟通 , 甲方 独立 设计 出 工程 效果图 、 施工图 。标明 物品 名称 、 数量 、 单价 、 规格 、 库存量 、 申购量 内容 。 （ 司机 放宽 38 周岁 ） 合法 中国 公民 ； 2 、 身体健康 、 品行 、 工地 情况 配合 项目经理 编写 施工 方案 、 施工 组织 设计 , 制定 施工进度 表 。 施工 中 :'
#
#
# model_dm.random.seed(0)
# vec_1 = model_dm.infer_vector(s1.split(), alpha=0.01, epochs=100)
# vec_2 = model_dm.infer_vector(s2.split(), alpha=0.01, epochs=100)
# vec_3 = model_dm.infer_vector(s3.split(), alpha=0.01, epochs=100)
#
# model_dm.random.seed(0)
# vec1 = model_dm.infer_vector([s1], alpha=0.01, epochs=100)
# vec2 = model_dm.infer_vector([s2], alpha=0.01, epochs=100)
# vec3 = model_dm.infer_vector([s3], alpha=0.01, epochs=100)
#
# vec_array_1 = np.array([vec_1, vec_2])
# vec_array_2 = np.array([vec_3])
#
# vec_array1 = np.array([vec1, vec2])
# vec_array2 = np.array([vec3])
#
# simi_1 = cosine_similarity(vec_array_2, vec_array_1)
# print('simi_1:', simi_1)
#
# simi_2 = cosine_similarity(vec_array2, vec_array1)
# print('simi_2:', simi_2)
#
#
import json

# def get_city_code(data, city_code):
#     for i in data:
#         if i.has_key('children'):

# with open(r'D:\CodeFiles\AI4recruitment\ai_recruitment\data_files\workspaces.json', 'r', encoding='utf-8') as f1:
#     data = json.load(f1)['list']
#     city_code = {}
#     for province in data:
#         city_code[province['label']] = province['value']
#
#         if 'children' in province:
#             for city in province['children']:
#                 city_code[city['label']] = city['value']
#
#                 if 'children' in city:
#                     for town in city['children']:
#                         city_code[town['label']] = town['value']
#     print(city_code)
#     print(len(city_code))
#     with open(r'D:\CodeFiles\AI4recruitment\ai_recruitment\data_files\city_code.json', 'w', encoding='utf-8') as f2:
#         json.dump(city_code, f2, indent=2, ensure_ascii=False)


import pandas as pd
import numpy as np
import pickle
import os
import json
from sklearn.metrics.pairwise import cosine_similarity, paired_distances

def read_d2v(jd_D2V_path=r'D:\CodeFiles\AI4recruitment\ai_recruitment\models\exp4\jd_D2V.csv',
             user_D2V_path=r'D:\CodeFiles\AI4recruitment\ai_recruitment\models\exp4\user_D2V.csv'):
    jd_D2V = pd.read_csv(jd_D2V_path).values
    user_D2V = pd.read_csv(user_D2V_path).values

    print('jd_D2V:', jd_D2V.shape)
    print('user_D2V:', user_D2V.shape)

    jd_D2V_ = []
    for i in jd_D2V:
        # print(i)
        val = i[0].replace('\n', '')
        # val = val.split()
        val = eval(val)
        jd_D2V_.append(np.array(val))

    user_D2V_ = []
    for i in user_D2V_:
        # print(i)
        val = i[0].replace('\n', '')
        # val = val.split()
        user_D2V_.append(np.array(val))

    user_D2V_ = np.array(user_D2V_)
    jd_D2V_ = np.array(jd_D2V_)

    simi = cosine_similarity(user_D2V_, jd_D2V_)
    res = np.diagonal(simi)

    return res

def load_d2v(jd_path=r'D:\CodeFiles\AI4recruitment\ai_recruitment\models\exp4\jd_D2V.dat',
             user_path=r'D:\CodeFiles\AI4recruitment\ai_recruitment\models\exp4\user_D2V.dat'):
    with open(jd_path, 'rb') as f1:
        jd_data = pickle.load(f1)

    with open(user_path, 'rb') as f2:
        user_data = pickle.load(f2)

    print(f'jd_data:{jd_data.astype().dtype}, user_data:{user_data.shape}')
    # simi.shape:(4500, 269534)
    simi = cosine_similarity(user_data, jd_data)
    simi_val = np.diagonal(simi)
    print('simi_val.shape:', simi_val.shape)

    with open(os.path.join(r'D:\CodeFiles\AI4recruitment\ai_recruitment\models\exp0', 'user_jd_simi.dat'),
              "wb") as f1:
        pickle.dump(simi_val, f1)


    #print(f'jd_data:{type(jd_data)}, user_data:{user_data.shape}')

# from elasticsearch import Elasticsearch
# def connect_es(es_ip='172.31.210.220'):
#     es = Elasticsearch(hosts="http://172.31.210.220:9200")
#       # index：选择数据库
#     print(es.search(index='rec_position'))


if __name__ == '__main__':
    load_d2v()
    # res = read_d2v()

    # connect_es()







































































