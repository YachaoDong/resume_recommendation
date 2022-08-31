# -*- coding: utf-8 -*-
'''
@Time    : 7/18/2022 4:48 PM
@Author  : dong.yachao
'''
import argparse
import os
import pickle
import json
import pandas as pd
import numpy as np
import lightgbm as lgb      # 分类模型
from gensim.models.doc2vec import Doc2Vec  # doc2vec模型
from infer_feats_utils import gen_user_feats, gen_jd_feats, gen_jd_user_feats, del_attr
from connet_data import get_data_from_sql


def get_final_feats(user_df, jd_df, doc2vec_model, use_tfidf=False,  use_doc2vec=True,
                    tfidf_enc_user=None, svd_tag_user=None, tfidf_enc_jd=None, svd_tag_jd=None):
    '''
    将从数据库中(connect.py)读取的 岗位和简历数据 处理成最后可以输入到模型推理的特征df
    @param user_df:
    @param jd_df:
    @param doc2vec_model:
    @param use_doc2vec:
    @param tfidf_enc_user:
    @param svd_tag_user:
    @param tfidf_enc_jd:
    @param svd_tag_jd:
    @return: 可以输入到模型训练的数据jd_user_df, 原始index jd_usr_id_df
    '''
    # 1.生成简历特征
    user_df = gen_user_feats(user_df, doc2vec_model, use_tfidf=use_tfidf,  use_doc2vec=use_doc2vec, tfidf_enc=tfidf_enc_user, svd_tag=svd_tag_user)
    # 2.生成岗位特征
    jd_df = gen_jd_feats(jd_df, doc2vec_model, use_cut_json=False, use_tfidf=use_tfidf,  use_doc2vec=use_doc2vec, tfidf_enc=tfidf_enc_jd, svd_tag=svd_tag_jd)
    # 3.拼接user_df和jd_df
    user_df = pd.DataFrame(np.repeat(user_df.values, len(jd_df), axis=0), columns=user_df.columns)
    jd_user_df = pd.concat([user_df, jd_df], axis=1)
    # 4.生成 user_jd的交叉特征
    jd_user_df = gen_jd_user_feats(jd_user_df, use_doc2vec=use_doc2vec)
    # 5.获取原始数据 id，用于后续返回结果
    jd_usr_id_df = jd_user_df[['jd_no', 'user_id']]
    # 6.删除除训练需要以外的特征字段
    del_attr(jd_user_df)

    # 删除不用的变量，释放资源
    del user_df
    del jd_df

    return jd_user_df, jd_usr_id_df


def predict(input_X, delivered_model, satisfied_model):
    '''
    用于模型预测
    @param input_X: 经过get_final_feats处理可以训练的数据
    @param delivered_model: 预测用户是否投递模型
    @param satisfied_model: 预测企业是否满意模型
    @return: 最终匹配得分
    '''
    delivered_pred = delivered_model.predict(input_X, num_iteration=delivered_model.best_iteration)
    satisfied_pred = satisfied_model.predict(input_X, num_iteration=satisfied_model.best_iteration)
    final_pred = delivered_pred * 0.6 + satisfied_pred * 0.4
    return final_pred


def sort_pred(final_pred):
    '''
    对模型预测进行排序
    TODO： 可增加规则排序，在相同城市区间，使用预测得分排序 在final_pred基础上+1，依次类推
    @param final_pred: 经过predict得出最后综合匹配得分的数据
    @return:
    '''
    final_pred = np.array(final_pred)
    sort_index = np.argsort(final_pred, axis=1)

    return sort_index


def init_model(args):
    # 加载推荐分类模型
    delivered_model = lgb.Booster(model_file=args.delivered_model)
    satisfied_model = lgb.Booster(model_file=args.satisfied_model)

    model_doc2vec, tfidf_enc_user, svd_tag_user, tfidf_enc_jd, svd_tag_jd = [None]*5

    # 加载doc2vec模型
    if args.use_doc2vec:
        model_doc2vec = Doc2Vec.load(args.doc2vec_model)
        model_doc2vec.random.seed(0)

    # 加载tfidf特征
    if args.use_tfidf:
        with open(args.tfidf_enc_user, 'rb') as fr:
            tfidf_enc_user = pickle.load(fr)
        with open(args.svd_tag_user, 'rb') as fr:
            svd_tag_user = pickle.load(fr)
        with open(args.tfidf_enc_jd, 'rb') as fr:
            tfidf_enc_jd = pickle.load(fr)
        with open(args.svd_tag_jd, 'rb') as fr:
            svd_tag_jd = pickle.load(fr)

    return delivered_model, satisfied_model, model_doc2vec, tfidf_enc_user, svd_tag_user, tfidf_enc_jd, svd_tag_jd


# 本地测试调用
def main(args):

    # 读取数据库数据
    jd_df, user_df = get_data_from_sql(server=args.server,
                                       user=args.user,
                                       password=args.password,
                                       database=args.database,
                                       status=args.status,
                                       query_jd_id=None,
                                       query_user_id=None)
    # 加载模型
    delivered_model, satisfied_model, model_doc2vec,  tfidf_enc_user, svd_tag_user, tfidf_enc_jd, svd_tag_jd = init_model(args)

    jd_user_df, jd_usr_id_df = get_final_feats(user_df, jd_df, doc2vec_model=model_doc2vec,
                                               use_tfidf=args.use_tfidf, use_doc2vec=args.use_doc2vec,
                                               tfidf_enc_user=tfidf_enc_user, svd_tag_user=svd_tag_user,
                                               tfidf_enc_jd=tfidf_enc_jd, svd_tag_jd=svd_tag_jd)
    final_pred = predict(input_X=jd_user_df.values, delivered_model=delivered_model, satisfied_model=satisfied_model)
    # 结果排序，返回排序后的index
    jd_usr_id_df['final_pred'] = final_pred
    # 可增加按照 是否为同一城市、是否行业相同等值排序
    final_index = jd_usr_id_df.sort_values(['final_pred'])['user_id', 'jd_no']

    return final_index



#  flask 推测调用的函数
def infer_flask(recommend_id, user_flag, args,
                delivered_model, satisfied_model, model_doc2vec,
                tfidf_enc_user, svd_tag_user, tfidf_enc_jd, svd_tag_jd):

    # 读取数据库数据
    query_user_id = None
    query_jd_id = None
    if user_flag:
        query_user_id = recommend_id
    else:
        query_jd_id = recommend_id

    jd_df, user_df = get_data_from_sql(server=args.server,
                                       user=args.user,
                                       password=args.password,
                                       database=args.database,
                                       status=args.status,
                                       query_jd_id=query_jd_id,
                                       query_user_id=query_user_id
                                       )

    jd_user_df, jd_usr_id_df = get_final_feats(user_df, jd_df, doc2vec_model=model_doc2vec,
                                               use_tfidf=args.use_tfidf, use_doc2vec=args.use_doc2vec,
                                               tfidf_enc_user=tfidf_enc_user, svd_tag_user=svd_tag_user,
                                               tfidf_enc_jd=tfidf_enc_jd, svd_tag_jd=svd_tag_jd)

    final_pred = predict(input_X=jd_user_df.values, delivered_model=delivered_model, satisfied_model=satisfied_model)
    # 结果排序，返回排序后的index
    # TODO 可增加按照 是否为同一城市、是否行业相同等值排序
    sort_feats = ['final_pred']
    if args.use_rule_sort:
        jd_usr_id_df[args.rule_feats] = jd_user_df[args.rule_feats]
        # 先规则，再pred_score进行排序
        sort_feats = args.rule_feats + sort_feats
    jd_usr_id_df['final_pred'] = final_pred


    if user_flag:
        final_index = jd_usr_id_df.sort_values(sort_feats,  ascending=False)['jd_no']
    else:
        final_index = jd_usr_id_df.sort_values(['final_pred'])['user_id']

    # print('final_pred:', jd_usr_id_df['final_pred'].values[final_index])

    # 清理内存
    del jd_df, user_df, jd_user_df, jd_usr_id_df, final_pred

    return final_index.values



if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=" AI for Recruitment Model ")
    # 连接数据库的信息
    parser.add_argument('--server', type=str, default='172.31.210.242', help='The name of the cpu to be queried. ')
    parser.add_argument('--user', type=str, default='root', help='SQL user name')
    parser.add_argument('--password', type=str, default='bluecloud123', help='SQL password')
    parser.add_argument('--database', type=str, default='gongchuang', help='The name of the database')

    # 模型信息
    parser.add_argument('--delivered_model', type=str,
                        default=r'D:\CodeFiles\AI4recruitment\ai_recruitment\models\exp6\lgb_delivered_model.txt',
                        help='match the topK cpus to output')
    parser.add_argument('--satisfied_model', type=str,
                        default=r'D:\CodeFiles\AI4recruitment\ai_recruitment\models\exp6\lgb_satisfied_model.txt',
                        help='match the topK cpus to output')

    parser.add_argument('--use_doc2vec', type=bool, default=True, help='The name of the database')
    parser.add_argument('--use_tfidf', type=bool, default=False, help='The name of the database')

    # doc2vec模型
    parser.add_argument('--doc2vec_model', type=str,
                        default=r'D:\CodeFiles\AI4recruitment\ai_recruitment\models\exp0\d2v_model',
                        help='The name of the database')

    # tfidf模型
    parser.add_argument('--tfidf_enc_user', type=str,
                        default=r'D:\CodeFiles\AI4recruitment\ai_recruitment\models\exp3\tfidf_enc_user.dat',
                        help='match the topK cpus to output')
    parser.add_argument('--svd_tag_user', type=str,
                        default=r'D:\CodeFiles\AI4recruitment\ai_recruitment\models\exp3\svd_tag_user.dat',
                        help='match the topK cpus to output')
    parser.add_argument('--tfidf_enc_jd', type=str,
                        default=r'D:\CodeFiles\AI4recruitment\ai_recruitment\models\exp3\tfidf_enc_jd.dat',
                        help='match the topK cpus to output')
    parser.add_argument('--svd_tag_jd', type=str,
                        default=r'D:\CodeFiles\AI4recruitment\ai_recruitment\models\exp3\svd_tag_jd.dat',
                        help='match the topK cpus to output')
    args = parser.parse_args()

    final_index = main(args)





