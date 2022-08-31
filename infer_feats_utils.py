# -*- coding: utf-8 -*-
'''
@Time    : 7/13/2022 2:04 PM
@Author  : dong.yachao
'''
import pandas as pd
import numpy as np
import jieba as jb

from sklearn.decomposition import TruncatedSVD  # 降维，(文本数，词汇数) -> (文本数，主题数)

# 文本预处理，将文本转换为向量 (文本数，) -> (文本数，词汇数)
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import LabelEncoder
import re
import warnings
from tqdm import tqdm
from joblib import Parallel, delayed
import os
from doc2vec import doc2vec_infer

from sklearn.metrics.pairwise import cosine_similarity, paired_distances
import json

warnings.filterwarnings('ignore')

# 学历
degree_map_user = {"null": 0, '其他': 0, '初中': 1, '中技': 2, '中专': 2, '高中': 2, '大专': 3, '本科': 4,
              '硕士': 5, 'MBA': 5, 'EMBA': 5, '博士': 6}

degree_map = {"null": 0, '不限': 0, '不限制': 0, '专科生及以上': 3, '本科生及以上': 4, '硕士生及以上': 5}


def del_attr(df, use_tfidf=False, use_doc2vec=True):
    '''
    删除 除训练以外的特征
    @param df:
    @param use_tfidf:
    @param use_doc2vec:
    @return:
    '''
    use_feats = ['same_user_city', 'gt_edu', 'min_desire_salary_num',
                 'max_desire_salary_num', 'same_jd_sub', 'same_desire_industry']
    if use_tfidf:
        for i in range(10):
            use_feats.append(f'jd_tfidf_{i}')
            use_feats.append(f'user_tfidf_{i}')

    if use_doc2vec:
        for i in range(20):
            use_feats.append(f'user_D2V_{i}')
            use_feats.append(f'jd_D2V_{i}')

    all_feats = df.columns
    del_feats = [i for i in all_feats if i not in use_feats]
    df.drop(del_feats, axis=1, inplace=True)

# 根据字段获取 职位 最低薪资
def get_min_salary(x):
    try:
        if not re.search(r'\d', str(x)):
            return -1
        else:
            min_salary = str(x).split('~')[0]
            if not re.search(r'\d', min_salary):
                return -1
            else:
                return int(min_salary.split('k')[0])
    except:
        print('get_min_salary error, it has return -1')
        return -1

# 根据字段获取 职位 最高薪资
def get_max_salary(x):
    try:
        if not re.search(r'\d', str(x)):
            return -1
        else:
            max_salary = str(x).split('~')[1]
            if not re.search(r'\d', max_salary):
                return -1
            else:
                return int(max_salary.split('k')[0])
    except:
        print('get_max_salary error, it has return -1')
        return -1

# 获取期望最低薪资
def get_desire_min_salary(x):
    try:
        min_list = [int(i.split('K')[0]) for i in x.split(',')]
        return min(min_list)
    except:
        print(f'get_desire_min_salary error, the origin x is {x}, and it has returned default -1.')
        return -1

# 获取期望最高薪资
def get_desire_max_salary(x):
    try:
        max_list = [int(i.split('K')[0]) for i in x.split(',')]
        return max(max_list)
    except:
        print(f'get_desire_max_salary error, the origin x is {x}, and it has returned -1.')
        return -1


def get_user_edu(x):
    try:
        edu_list = [degree_map_user[i] for i in str(x).split(',')]
        return max(edu_list)
    except:
        print(f'get_user_edu error, the origin x is {x}, and it has returned -1.')
        return -1


# 获取岗位工作地点编码
def get_jd_city_id(x):
    try:
        x = eval(str(x))
        city_list = []
        for kv in x:
            try:
                city_list.append(int(kv["code"]))
            except:
                city_list.append(-1)
        # print('city_list:', city_list)
        return city_list
    except:
        print(f'get_jd_city_id error, the origin x is {x}, and it has returned [-1].')
        return [-1]


# 获取学生期望城市编码
def get_user_city_id(x):
    try:
        city_list = [int(i) for i in x.split(',')]
        return city_list
    except:
        print(f'get_jd_city_id error, the origin x is {x}, and it has returned [-1].')
        return [-1]


# 简历：判断 职位城市 和 期望工作城市 是否一致(或包含于)
def is_same_user_city(df):
    # ["北京市","天津市","河北省","山西省","内蒙古自治区","辽宁省"]
    jd_city_id = df['city']
    desire_jd_city = df['desire_jd_city_id']

    if set(jd_city_id) & set(desire_jd_city):
        return True
    else:
        return False


def get_tfidf(df, text_list, which_flag, tfidf_enc=None, svd_tag=None):

    if tfidf_enc is None:
        tfidf_enc = TfidfVectorizer(ngram_range=(1, 2))
    if svd_tag is None:
        # 降维
        svd_tag = TruncatedSVD(n_components=10, n_iter=20, random_state=2022)

    text_list = text_list * 11
    tfidf_vec = tfidf_enc.fit_transform(text_list)
    tag_svd = svd_tag.fit_transform(tfidf_vec)[0]
    tag_svd = tag_svd.reshape((1, 10))


    tag_svd = pd.DataFrame(tag_svd)
    # 赋予列名
    tag_svd.columns = [f'{which_flag}_tfidf_{i}' for i in range(10)]
    # 拼接，在列方向上
    df = pd.concat([df, tag_svd], axis=1)
    return df,  tfidf_enc, svd_tag

# 读取停词表 添加不同形式的 空格
# win:r"D:\CodeFiles\AI4recruitment\ai_recruitment\data_files\stopwords_files\baidu_stopwords.txt"
def get_stop_words(stop_words_path=None):
    if stop_words_path is None:
        stop_words_path = os.path.join(os.path.split(os.path.abspath(__file__))[0], "data_files", "stopwords_files", "baidu_stopwords.txt")
        # print("stop_words_path:", stop_words_path)
    stop_words = [i.strip() for i in open(stop_words_path, 'r', encoding='utf8').readlines()]
    stop_words.extend(['\n', '\xa0', '\u3000', '\u2002'])
    return stop_words

# 对字符串x进行分词 并使用 " " 连接，返回的是分词后的字符串
def get_str(x, stop_words=None):
    if stop_words is None:
        stop_words = ['\n', '\xa0', '\u3000', '\u2002']
    return ' '.join([i for i in jb.cut(x) if i not in stop_words])


def get_doc_vec_simi(df):
    user_D2V_cols = []
    jd_D2V_cols = []
    for i in range(20):
        user_D2V_cols.append(f'user_D2V_{i}')
        jd_D2V_cols.append(f'jd_D2V_{i}')

    simi = cosine_similarity(df[user_D2V_cols].values, df[jd_D2V_cols].values)
    df['jd_user_dis'] = np.diagonal(simi)
    del simi
    return df


def gen_jd_feats(jd_df, doc2vec_model, use_cut_json=False, use_tfidf=False, use_doc2vec=False, tfidf_enc=None, svd_tag=None):
    '''
    传入从数据库(connect_data.py)中读取的岗位df数据，用于生成基础特征和doc2vec特征
    @param jd_df: 企业岗位数据
    @param doc2vec_model: 文本转向量模型
    @param use_cut_json: 是否使用已保存的分词json文件，快速读取分词结果
    @param use_tfidf: 是否使用tfidft特征，因为 n_dim=min(n_samples, n_feats), 所以对于单样本生成该特征的维度=1，暂时不用
    @param use_doc2vec: 是否使用doc2vec特征
    @param tfidf_enc: 传入tfidf 模型，
    @param svd_tag: 传入降维模型
    @return: 包含基础特征+doc2vec特征的jd_df
    '''
    # 薪水
    jd_df['min_salary'] = jd_df['payment'].apply(get_min_salary)
    jd_df['max_salary'] = jd_df['payment'].apply(get_max_salary)
    # 学历
    jd_df['min_edu_level_num'] = jd_df['educate_bg'].map(degree_map)
    # 工作地点
    jd_df['city'] = jd_df['workplace'].apply(get_jd_city_id)

    # 获取分词
    if use_cut_json:
        with open(r"D:\CodeFiles\AI4recruitment\ai_recruitment\models\job_description_cut.json", "r",
                  encoding="utf-8") as f1:
            tmp_cut = list(json.load(f1)['tmp_cut'])
    else:
        # 对每个jd的岗位描述 做分词，tmp_cut：list, 长度为岗位的个数
        stop_words = get_stop_words()
        #print('jd_df job_description]', jd_df['job_description\n'])
        tmp_cut = Parallel(n_jobs=-1)(delayed(get_str)(jd_df.loc[ind]['job_description\n'], stop_words=stop_words)
                                      for ind in tqdm(jd_df.index))
    # 构建 tfidf 特征
    if use_tfidf:
        jd_df, tfidf_enc_jd, svd_tag_jd = get_tfidf(jd_df, tmp_cut, which_flag='jd', tfidf_enc=tfidf_enc, svd_tag=svd_tag)

    # 构建 Doc2Vec特征
    if use_doc2vec:
        jd_D2V = Parallel(n_jobs=-1)(delayed(doc2vec_infer)(model=doc2vec_model, doc_text=sentence)
                                      for sentence in tqdm(tmp_cut))
        # jd_D2V: list[array[]]
        jd_D2V = np.array(jd_D2V)
        # print('jd_D2V:', jd_D2V)
        # 如果jd_D2V部分为空


        # 将20维特征写入df中
        jd_D2V = pd.DataFrame(jd_D2V)
        # 赋予列名
        jd_D2V.columns = [f'jd_D2V_{i}' for i in range(20)]
        # 拼接，在列方向上
        jd_df = pd.concat([jd_df, jd_D2V], axis=1)
        del jd_D2V
    del tmp_cut
    return jd_df


def gen_user_feats(user_df, doc2vec_model, use_tfidf=False, use_doc2vec=False, tfidf_enc=None, svd_tag=None):
    '''
    传入从数据库(connect_data.py)中读取的简历df数据，用于生成基础特征和doc2vec特征
    @param user_df: 简历df数据
    @param doc2vec_model:
    @param use_tfidf:
    @param use_doc2vec:
    @param tfidf_enc:
    @param svd_tag:
    @return: 包含基础特征+doc2vec特征的user_df
    '''
    # 期望工作城市处理，一共有三个，如："691,698,-"， 匹配 所有数字,返回['691', '698'], 用于判断居住地和期望地是否一致
    user_df['desire_jd_city_id'] = user_df['base_code'].apply(get_user_city_id)
    # user_df['desire_jd_city_id'].fillna([-1], inplace=True)

    # 薪资
    user_df[['start_salary', 'end_salary']] = user_df[['start_salary', 'end_salary']].astype(str)
    user_df['min_desire_salary'] = user_df['start_salary'].apply(get_desire_min_salary)
    user_df['max_desire_salary'] = user_df['end_salary'].apply(get_desire_max_salary)

    # 学历
    user_df['cur_degree_id_num'] = user_df['education'].apply(get_user_edu)

    # 获取分词
    stop_words = get_stop_words()
    tmp_cut = Parallel(n_jobs=-1)(delayed(get_str)(user_df.loc[ind]['experience'], stop_words=stop_words)
                                  for ind in tqdm(user_df.index))
    # 构建 tfidf 特征
    if use_tfidf:
        user_df, tfidf_enc_user, svd_tag_user = get_tfidf(user_df, tmp_cut, which_flag='user', tfidf_enc=tfidf_enc, svd_tag=svd_tag)

    # 构建Doc2Vec特征
    if use_doc2vec:
        user_D2V = Parallel(n_jobs=-1)(delayed(doc2vec_infer)(model=doc2vec_model, doc_text=user_df.loc[ind]['experience'])
                                       for ind in tqdm(user_df.index))
        # user_D2V:list[array[]]
        user_D2V = np.array(user_D2V)
        print('user_D2V:', user_D2V)
        # 如果user_D2V为空,使用-0.1填充
        if user_D2V.shape != (1, 20):
            user_D2V = np.full((1, 20), -0.1)

        # 将20维特征写入df中
        user_D2V = pd.DataFrame(user_D2V)
        # 赋予列名
        user_D2V.columns = [f'user_D2V_{i}' for i in range(20)]
        # 拼接，在列方向上
        user_df = pd.concat([user_df, user_D2V], axis=1)

        del user_D2V
    del tmp_cut

    return user_df

# 推理中没有用到
def process_action_df(action_df, jd_df, user_df):
    '''
    将 jd_df + user_df + action_df组合一起构建训练集
    @param action_df: 训练集标签，包含是否投递和是否满意
    @param jd_df:     企业方 特征
    @param user_df:  用户方 特征
    @return: train_df 训练集
    '''

    # 删除重复数据
    action_df = action_df.drop_duplicates()
    # 排序
    action_df.sort_values(['user_id', 'jd_no'], inplace=True)
    # 删除 'user_id' 和 'jd_no' 一样的重复数据
    action_df = action_df.drop_duplicates(subset=['user_id', 'jd_no'], keep='last')
    # 筛选出 只在训练集出现过的jd_id
    action_df = action_df[action_df['jd_no'].isin(jd_df['jd_no'].unique())]
    # 按照user_id key，以action_df的user_id为基准合并
    train_df = action_df.merge(user_df, on='user_id', how='left')
    train_df = train_df.merge(jd_df, on='jd_no', how='left')
    return train_df


def gen_jd_user_feats(jd_user_df, use_doc2vec=False):
    '''
    将合并后的 jd_user 生成共同的特恒
    @param jd_user_df:
    @param use_doc2vec:
    @return:
    '''
    # 工作地点和期望城市匹配
    jd_user_df['city'].fillna(-1, inplace=True)
    jd_user_df['same_user_city'] = jd_user_df.apply(lambda x: bool(set(x['city']) & set(x['desire_jd_city_id'])), axis=1).astype(int)

    # 学历匹配
    jd_user_df['gt_edu'] = (jd_user_df['cur_degree_id_num'] >= jd_user_df['min_edu_level_num']).astype(int)

    # 薪资匹配
    jd_user_df['min_desire_salary_num'] = (jd_user_df['min_desire_salary'] <= jd_user_df['min_salary']).astype(int)
    jd_user_df['max_desire_salary_num'] = (jd_user_df['max_desire_salary'] <= jd_user_df['max_salary']).astype(int)

    # 期望行业是否匹配
    jd_user_df['jd_industry_code'] = jd_user_df['jd_industry_code'].astype(str)
    jd_user_df['user_industry_code'] = jd_user_df['user_industry_code'].astype(str)
    #jd_user_df['user_industry_code'] = jd_user_df.apply(lambda x: x['user_industry_code'].split(','))
    jd_user_df['same_desire_industry'] = jd_user_df.apply(lambda x: bool(x['jd_industry_code'] in x['user_industry_code']), axis=1).astype(int)

    # 期望职位 和 岗位职位 是否匹配
    jd_user_df['post_code'] = jd_user_df['post_code'].astype(str)
    jd_user_df['position_code'] = jd_user_df['position_code'].astype(str)
    # jd_user_df['position_code'] = jd_user_df.apply(lambda x: x['position_code'].split(','))
    jd_user_df['same_jd_sub'] = jd_user_df.apply(lambda x: bool(x['post_code'] in x['position_code']), axis=1).astype(int)

    # TODO 计算 d(岗位描述, 工作经验) 文本相似度
    # if use_doc2vec:
    #     jd_user_df = get_doc_vec_simi(jd_user_df)

    return jd_user_df




if __name__ == '__main__':
    train_data_path = r'F:\myProject\all_datasets\recruitment_data'
    train_jd = pd.read_csv(os.path.join(train_data_path, 'table2_jd.csv'), sep='\t')

    jd_df = gen_jd_feats(train_jd)









































