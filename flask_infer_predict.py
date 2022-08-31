# -*- coding: utf-8 -*-
'''
@Time    : 8/3/2022 2:38 PM
@Author  : dong.yachao
'''

import argparse
import yaml
import os
import pickle
import json
import pandas as pd
import numpy as np
import lightgbm as lgb      # 分类模型


import flask
from flask import Flask, request, jsonify
#  WSGI
from gevent import pywsgi

# 导入模型包
from infer_feats_utils import gen_user_feats, gen_jd_feats, gen_jd_user_feats, del_attr
from connet_data import get_data_from_sql
from infer_predict import init_model, infer_flask


app = Flask(__name__)




# predict
@app.route("/recommend_stu", methods=["POST"])
def recommend_stu():
    # Initialize the data dictionary that will be returned from the view.
    data = {"success": False}
    print('start!!')
    # Ensure an image was properly uploaded to our endpoint.
    if flask.request.method == 'POST':
        if flask.request.form.get("recommend_id"):
            recommend_id = str(flask.request.form.to_dict()["recommend_id"])
            print("Recieve recommend_id from server:", recommend_id)
            user_flag = True

            res_recommend_id = infer_flask(recommend_id, user_flag, args,
                                           delivered_model, satisfied_model, model_doc2vec,
                                           tfidf_enc_user, svd_tag_user, tfidf_enc_jd, svd_tag_jd)

            print('res_recommend_id:', res_recommend_id)
            data['recommend_id'] = list(res_recommend_id)

            # Indicate that the request was a success.
            data["success"] = True

            # Return the data dictionary as a JSON response.
            print('data:', data)
            return jsonify(data)

# predict 传入学生简历ID
@app.route("/recommend_company", methods=["POST"])
def recommend_company():
    # Initialize the data dictionary that will be returned from the view.
    data = {"success": False}
    print('start!!')
    # Ensure an image was properly uploaded to our endpoint.
    if flask.request.method == 'POST':
        if flask.request.form.get("recommend_id"):

            recommend_id = flask.request.form.to_dict()["recommend_id"]
            print("Recieve recommend_id from server:", recommend_id)
            user_flag = True

            res_recommend_id = infer_flask(recommend_id, user_flag, args,
                                           delivered_model, satisfied_model, model_doc2vec,
                                           tfidf_enc_user, svd_tag_user, tfidf_enc_jd, svd_tag_jd)
            print('res_recommend_id:', res_recommend_id)
            data['recommend_id'] = list(res_recommend_id)

            # Indicate that the request was a success.
            data["success"] = True

            # Return the data dictionary as a JSON response.
            print('data:', data)
            return jsonify(data)

# # linux test:  python infer_predict.py --delivered_model /home/all_projects/ai_recruitment/models/exp6/lgb_delivered_model.txt
# --satisfied_model /home/all_projects/ai_recruitment/models/exp6/lgb_satisfied_model.txt
# --doc2vec_model /home/all_projects/ai_recruitment/models/exp0/d2v_model
# --local_host "172.31.210.242"
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=" AI for Recruitment Model ")
    # config.yml
    parser.add_argument('--cfg', type=str, default=r'D:\CodeFiles\AI4recruitment\gongchaung_all\gongchuang\ai_recruitment\configs\local_test.yml', help='SQL user name')

    # 连接数据库的信息
    parser.add_argument('--server', type=str, default='172.31.210.242', help='The name of the cpu to be queried. ')
    parser.add_argument('--user', type=str, default='root', help='SQL user name')
    parser.add_argument('--password', type=str, default='bluecloud123', help='SQL password')
    parser.add_argument('--database', type=str, default='gongchuang', help='数据库名字')
    parser.add_argument('--status', type=int, default=3, help='岗位数据库中的status，分为正式数据、缓存数据等')

    # 模型信息
    parser.add_argument('--delivered_model', type=str,
                        default=r'./models/exp6/lgb_delivered_model.txt',
                        help='lgb模型：预测用户是否投递模型')
    parser.add_argument('--satisfied_model', type=str,
                        default=r'./models/exp6/lgb_satisfied_model.txt',
                        help='lgb模型：预测HR是否满意模型')

    parser.add_argument('--use_doc2vec', type=bool, default=True, help='是否使用doc2vec特征进行预测')
    parser.add_argument('--use_tfidf', type=bool, default=False, help='是否使用tfidf特征，这里暂时不使用')

    # doc2vec模型
    parser.add_argument('--doc2vec_model', type=str,
                        default=r'./models/exp0/d2v_model',
                        help='doc2vec_model')

    # tfidf模型
    parser.add_argument('--tfidf_enc_user', type=str, default=r'D:\CodeFiles\AI4recruitment\ai_recruitment\models\exp3\tfidf_enc_user.dat', help='暂不使用')
    parser.add_argument('--svd_tag_user', type=str, default=r'D:\CodeFiles\AI4recruitment\ai_recruitment\models\exp3\svd_tag_user.dat', help='暂不使用')
    parser.add_argument('--tfidf_enc_jd', type=str, default=r'D:\CodeFiles\AI4recruitment\ai_recruitment\models\exp3\tfidf_enc_jd.dat', help='暂不使用')
    parser.add_argument('--svd_tag_jd', type=str, default=r'D:\CodeFiles\AI4recruitment\ai_recruitment\models\exp3\svd_tag_jd.dat', help='暂不使用')

    # 结构后处理，排序信息  rule_feats
    parser.add_argument('--use_rule_sort', type=bool, default=False, help='是否使用规则排序，即先按照是否为同一城市、学历要求是否相符合等等规则先进行排序')
    # all rule feats:['same_user_city', 'gt_edu', 'min_desire_salary_num', 'max_desire_salary_num', 'same_jd_sub', 'same_desire_industry']
    parser.add_argument('--rule_feats', type=list, default=['same_user_city', 'min_desire_salary_num', 'same_desire_industry'], help='使用规则排的特征')


    # flask 本地信息，ip port
    parser.add_argument('--local_host', type=str, default="172.31.26.230", help='当前server ip地址')
    parser.add_argument('--flask_port', type=int, default=5000, help='flask 端口')

    args = parser.parse_args()

    # 使用 yaml更新部分参数
    with open(args.cfg, 'r', encoding='utf-8') as f1:
        cfg = yaml.load(f1, Loader=yaml.FullLoader)
        args.server = cfg["SQL"]["server"]
        args.user = cfg["SQL"]["user"]
        args.password = cfg["SQL"]["password"]
        args.database = cfg["SQL"]["database"]

        args.local_host = cfg["LOCAL_INFO"]["local_host"]
        args.flask_port = cfg["LOCAL_INFO"]["flask_port"]

        args.doc2vec_model = cfg["MODEL"]["doc2vec_model"]
        args.delivered_model = cfg["MODEL"]["delivered_model"]
        args.satisfied_model = cfg["MODEL"]["satisfied_model"]

        args.use_rule_sort = cfg['POST_PROCESS']["use_rule_sort"]
        args.rule_feats = cfg['POST_PROCESS']["rule_feats"]
    print("---------- args content ----------")
    for k, v in sorted(vars(args).items()):
        print(k, '=', v)
    print("---------- args content ----------")


    # 初始化模型
    delivered_model, satisfied_model, model_doc2vec, tfidf_enc_user, svd_tag_user, tfidf_enc_jd, svd_tag_jd = init_model(args)

    # Method 1.
    app.debug = True
    # 172.31.18.19
    server = pywsgi.WSGIServer((args.local_host, args.flask_port), app)
    server.serve_forever()

    # Method 2.
    # server = make_server('127.0.0.1', 5000, app)
    # server.serve_forever()