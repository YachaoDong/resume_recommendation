# -*- coding：utf-8 -*-
'''
包含tfidf 和 doc2vec
'''

import pandas as pd
import numpy as np
from sklearn.model_selection import KFold  # k折交叉验证
from sklearn.model_selection import train_test_split
import lightgbm as lgb      # 分类模型
import warnings
from feats_utils import gen_jd_feats, gen_user_feats, process_action_df, gen_jd_user_feats, offline_eval_map
from gensim.models.doc2vec import Doc2Vec
import pickle
import os


warnings.filterwarnings('ignore')

'''
jd:     18个字段
user:   13个字段
'''



def train_model(train_, pred, label, cate_cols, save_path, is_shuffle=True, use_cate=True):
    '''
    all_data[all_data['satisfied'] != -1],  训练集
    all_data[all_data['satisfied'] == -1],  测试集
    use_feats,                              pred使用的特征列名
    'satisfied',                            label 预测 satisfied
    ['live_city_id', 'city'],               cate_cols
     use_cate=True
    '''


    n_splits = 5
    folds = KFold(n_splits=n_splits, shuffle=is_shuffle, random_state=1024)

    train_[f'{label}_pred'] = 0

    fold_importance_df = pd.DataFrame()
    fold_importance_df["Feature"] = pred

    print(f'Use {len(pred)} features ...')
    auc_scores = []
    params = {
        'learning_rate': 0.01,
        'boosting_type': 'gbdt',
        'objective': 'binary',
        'metric': 'auc',
        'num_leaves': 63,
        'feature_fraction': 0.8,
        'bagging_fraction': 0.8,
        'bagging_freq': 5,
        'seed': 1,
        'bagging_seed': 1,
        'feature_fraction_seed': 7,
        'min_data_in_leaf': 20,
        'nthread': -1,
        'verbose': -1
    }
    # train_user_id = train_['user_id'].unique()
    train_user_id = train_['user_id']
    # 划分训练集 测试集
    train_user_idx = np.arange(len(train_user_id))
    train_idx, test_idx = train_test_split(train_user_idx, test_size=0.1)
    # 训练集
    train_x, train_y = train_.loc[train_['user_id'].isin(train_user_id[train_idx]), pred], train_.loc[
        train_['user_id'].isin(train_user_id[train_idx]), label]

    # 测试集
    test_x, test_y = train_.loc[train_['user_id'].isin(train_user_id[test_idx]), pred], train_.loc[
        train_['user_id'].isin(train_user_id[test_idx]), label]


    # 是否使用分类特征
    if use_cate:
        dtrain = lgb.Dataset(train_x, label=train_y, categorical_feature=cate_cols)
        dtest = lgb.Dataset(test_x, label=test_y, categorical_feature=cate_cols)
    else:
        dtrain = lgb.Dataset(train_x, label=train_y)
        dtest = lgb.Dataset(test_x, label=test_y)


    clf = lgb.train(
        params=params,
        train_set=dtrain,
        num_boost_round=10000,
        valid_sets=[dtest],
        early_stopping_rounds=100,
        verbose_eval=100
    )

    # 验证集结果
    train_.loc[train_['user_id'].isin(train_user_id[test_idx]), f'{label}_pred'] = \
        clf.predict(test_x, num_iteration=clf.best_iteration)

    fold_importance_df[f'important_feats'] = clf.feature_importance()

    fold_importance_df.sort_values(by='important_feats', ascending=False, inplace=True)
    fold_importance_df[['Feature', 'important_feats']].to_csv(os.path.join(save_path, 'feat_imp_base2.csv'), index=False, encoding='utf8')

    print('saveing model in :', os.path.join(save_path, f'lgb_{label}_model.txt'))
    clf.save_model(os.path.join(save_path, f'lgb_{label}_model.txt'))
    # print('auc score', np.mean(auc_scores))

    return train_[['user_id', 'jd_no', f'{label}_pred', label]]


def sub_on_line(train_, pred, label, cate_cols, is_shuffle=True, use_cate=False):
    '''
    all_data[all_data['satisfied'] != -1],  训练集
    all_data[all_data['satisfied'] == -1],  测试集
    use_feats,                              pred使用的特征列名
    'satisfied',                            label 预测 satisfied
    ['live_city_id', 'city'],               cate_cols
     use_cate=True
    '''

    n_splits = 5
    folds = KFold(n_splits=n_splits, shuffle=is_shuffle, random_state=1024)

    train_[f'{label}_pred'] = 0
    fold_importance_df = pd.DataFrame()
    fold_importance_df["Feature"] = pred
    print(f'Use {len(pred)} features ...')
    auc_scores = []
    params = {
        'learning_rate': 0.01,
        'boosting_type': 'gbdt',
        'objective': 'binary',
        'metric': 'auc',
        'num_leaves': 63,
        'feature_fraction': 0.8,
        'bagging_fraction': 0.8,
        'bagging_freq': 5,
        'seed': 1,
        'bagging_seed': 1,
        'feature_fraction_seed': 7,
        'min_data_in_leaf': 20,
        'nthread': -1,
        'verbose': -1
    }
    train_user_id = train_['user_id'].unique()
    for n_fold, (train_idx, valid_idx) in enumerate(folds.split(train_user_id), start=1):
        print(f'the {n_fold} training start ...')
        # 训练
        train_x, train_y = train_.loc[train_['user_id'].isin(train_user_id[train_idx]), pred], train_.loc[
            train_['user_id'].isin(train_user_id[train_idx]), label]
        # 验证
        valid_x, valid_y = train_.loc[train_['user_id'].isin(train_user_id[valid_idx]), pred], train_.loc[
            train_['user_id'].isin(train_user_id[valid_idx]), label]
        print(f'for train user:{len(train_idx)}\nfor valid user:{len(valid_idx)}')
        # 是否使用分类特征
        if use_cate:
            dtrain = lgb.Dataset(train_x, label=train_y, categorical_feature=cate_cols)
            dvalid = lgb.Dataset(valid_x, label=valid_y, categorical_feature=cate_cols)
        else:
            dtrain = lgb.Dataset(train_x, label=train_y)
            dvalid = lgb.Dataset(valid_x, label=valid_y)

        clf = lgb.train(
            params=params,
            train_set=dtrain,
            num_boost_round=10000,
            valid_sets=[dvalid],
            early_stopping_rounds=100,
            verbose_eval=100
        )

        auc_scores.append(clf.best_score['valid_0']['auc'])
        # 特征重要性
        fold_importance_df[f'fold_{n_fold}_imp'] = clf.feature_importance()
        # 验证集结果
        train_.loc[train_['user_id'].isin(train_user_id[valid_idx]), f'{label}_pred'] = \
            clf.predict(valid_x, num_iteration=clf.best_iteration)

    five_folds = [f'fold_{f}_imp' for f in range(1, n_splits + 1)]
    fold_importance_df['avg_imp'] = fold_importance_df[five_folds].mean(axis=1)
    fold_importance_df.sort_values(by='avg_imp', ascending=False, inplace=True)
    fold_importance_df[['Feature', 'avg_imp']].to_csv('feat_imp_base.csv', index=False, encoding='utf8')


    print('auc score', np.mean(auc_scores))
    return train_[['user_id', 'jd_no', f'{label}_pred', label]]


def del_attr(df, use_feats):
    use_feats = use_feats + ['user_id', 'jd_no', 'satisfied', 'delivered']
    all_feats = df.columns
    del_feats = [i for i in all_feats if i not in use_feats]

    # 删除 公司名字列、删除最高学历列、是否要求管理经验、语言需求
    # jd_drop_list = ['company_name', 'max_edu_level', 'is_mangerial', 'resume_language_required',
    #                 'start_date', 'end_date', 'is_travel', 'require_nums']
    # user_drop_list = ['cur_industry_id', 'cur_salary_id', 'start_work_date', 'cur_salary_id', 'birthday', 'cur_jd_type']

    df.drop(del_feats, axis=1, inplace=True)

if __name__ == "__main__":
    # 工作年限（103: 一年到三年，305: 三年到五年，510: 五年到十年，1099: 十年以上）
    use_feats = ['jd_tfidf_0', 'jd_tfidf_1', 'jd_tfidf_2', 'jd_tfidf_3', 'jd_tfidf_4', 'jd_tfidf_5', 'jd_tfidf_6',
                 'jd_tfidf_7', 'jd_tfidf_8', 'jd_tfidf_9',

                 'user_tfidf_0', 'user_tfidf_1', 'user_tfidf_2', 'user_tfidf_3', 'user_tfidf_4', 'user_tfidf_5',
                 'user_tfidf_6', 'user_tfidf_7', 'user_tfidf_8', 'user_tfidf_9',

                 'same_user_city', 'gt_edu', 'min_desire_salary_num', 'max_desire_salary_num', 'same_jd_sub',
                 'same_desire_industry']

    # 保存模型
    save_path = f"./models/"
    exp_list = os.listdir(save_path)
    exp_num = max([int(i.split('exp')[-1]) for i in exp_list])
    save_path = os.path.join(save_path, 'exp' + str(exp_num+1))
    if os.path.isdir(save_path):
        exp_name = save_path.split('/')[-1]
        new_exp_name = 'exp' + str(int(exp_name[-1])+1)
        os.makedirs(save_path.replace(exp_name, new_exp_name))
    else:
        os.makedirs(save_path)
    print('save_path:', save_path)

    # 学历
    degree_map = {'其他': 0, '初中': 1, '中技': 2, '中专': 2, '高中': 2, '大专': 3, '本科': 4,
                  '硕士': 5, 'MBA': 5, 'EMBA': 5, '博士': 6}
    # 读取训练集、测试集
    sub_path = './submit/'
    # 训练、测试集路径
    train_data_path = 'D:\CodeFiles\data\\recruitment_data\\'
    # ------------ user ------------
    train_user = pd.read_csv(train_data_path + 'table1_user.csv')
    # ------- jd ----------
    train_jd = pd.read_csv(train_data_path + 'table2_jd.csv', sep='\t')
    # -------- 行为处理 是否浏览、是否投递、HR是否认可 --------
    train_action = pd.read_csv(train_data_path + 'table3_action.csv')

    use_doc2vec = True
    # 加载doc2vec 模型
    model_doc2vec = None
    if use_doc2vec:
        use_feats.append('jd_D2V')
        use_feats.append('user_D2V')
        use_feats.append('jd_user_dis')

        model_doc2vec = Doc2Vec.load("./models/exp0/d2v_model")
        model_doc2vec.random.seed(0)

    # 特征工程
    # train_jd = train_jd.iloc[:100]
    # train_user = train_user.iloc[:100]

    # jd feats
    train_jd, tfidf_enc_jd, svd_tag_jd, jd_D2V = gen_jd_feats(train_jd, model_doc2vec, use_doc2vec=use_doc2vec)
    # user feats
    train_user, tfidf_enc_user, svd_tag_user, user_D2V = gen_user_feats(train_user, model_doc2vec, use_doc2vec=use_doc2vec)

    # save doc2vec
    if use_doc2vec:
        with open(os.path.join(save_path, 'jd_D2V.dat'), "wb") as f1:
            pickle.dump(np.array(jd_D2V), f1)
        with open(os.path.join(save_path, 'user_D2V.dat'), "wb") as f2:
            pickle.dump(np.array(user_D2V), f2)

        # train_jd[['jd_D2V']].to_csv(os.path.join(save_path, 'jd_D2V.csv'), index=False, encoding='utf8')
        # train_user[['user_D2V']].to_csv(os.path.join(save_path, 'user_D2V.csv'), index=False, encoding='utf8')


    # jd + user + label
    train_jd_user = process_action_df(train_action, train_jd, train_user)
    # cross feats
    all_data = gen_jd_user_feats(train_jd_user, use_doc2vec=use_doc2vec)
    # 删除不使用特征的列
    del_attr(all_data, use_feats)
    print('all_data.columns: ', all_data.columns)


    # 保存
    pickle.dump(tfidf_enc_jd, open(os.path.join(save_path, "tfidf_enc_jd.dat"), "wb"))
    pickle.dump(svd_tag_jd, open(os.path.join(save_path, "svd_tag_jd.dat"), "wb"))
    pickle.dump(tfidf_enc_user, open(os.path.join(save_path, "tfidf_enc_user.dat"), "wb"))
    pickle.dump(svd_tag_user, open(os.path.join(save_path, "svd_tag_user.dat"), "wb"))
    all_data.to_csv(os.path.join(save_path, 'all_feats_data.csv'))



    # TODO
    # use_feats = [c for c in all_data.columns if c not in ['user_id', 'jd_no', 'delivered', 'satisfied'] +
    #              ['desire_jd_industry_id', 'desire_jd_type_id', 'cur_industry_id', 'cur_jd_type', 'experience',
    #              'jd_title', 'jd_sub_type', 'job_description\n']]
    # train_, pred, label, cate_cols, save_path, is_shuffle=True, use_cate=True
    train_pred_sat = train_model(train_=all_data[all_data['satisfied'] != -1],
                                 pred=use_feats,
                                 label='satisfied',
                                 cate_cols=['desire_jd_city', 'city'],
                                 save_path=save_path,
                                 use_cate=False)

    train_pred_dev = train_model(train_=all_data[all_data['delivered'] != -1],
                                 pred=use_feats,
                                 label='delivered',
                                 cate_cols=['desire_jd_city', 'city'],
                                 save_path=save_path,
                                 use_cate=False)

    train_pred_sat['merge_pred'] = train_pred_sat['satisfied_pred'] * 0.8 + \
                                   train_pred_dev['delivered_pred'] * 0.2


    train_pred_sat = train_pred_sat.merge(all_data[all_data['delivered'] != -1][['user_id', 'jd_no', 'delivered']],
                                          on=['user_id', 'jd_no'], how='left')

    dev_map = offline_eval_map(train_pred_sat, 'delivered', 'merge_pred')
    sat_map = offline_eval_map(train_pred_sat, 'satisfied', 'merge_pred')

    print('dev map:', round(dev_map, 4), 'sat map:', round(sat_map, 4), 'final score:',
          round(0.7 * sat_map + 0.3 * dev_map, 4))















