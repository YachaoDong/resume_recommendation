# -*- coding: utf-8 -*-
'''
@Time    : 7/13/2022 2:04 PM
@Author  : dong.yachao
'''
import pandas as pd
import numpy as np
import jieba as jb
from sklearn.model_selection import KFold  # k折交叉验证
from sklearn.model_selection import train_test_split
import lightgbm as lgb      # 分类模型
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
import pickle

warnings.filterwarnings('ignore')

# 处理招聘数据信息
def modified_jd_df(jd_path):
    tmp_list = []
    tmp_file = open(jd_path, encoding='utf8')
    for i, j in enumerate(tmp_file.readlines()):
        if i == 175425:
            j = j.replace('销售\t|置业顾问\t|营销', '销售|置业顾问|营销')
        tmp = j.split('\t')
        tmp_list.append(tmp)
    tmp_file.close()
    return pd.DataFrame(tmp_list[1:], columns=tmp_list[0])

# 根据字段获得最低薪资  期望薪资是10位或者12位（遇到9位或者11位前面或者后面补0，超过上面列表部分为脏数据）
def get_min_salary(x):
    if len(x) == 12:
        return int(x[:6])
    elif len(x) == 10:
        return int(x[:5])
    elif len(x) == 11:
        return int(x[:5])
    elif len(x) == 9:
        return int(x[:4])
    else:
        return -1

# 根据字段获取最高薪资
def get_max_salary(x):
    if len(x) == 12:
        return int(x[6:])
    elif len(x) == 10:
        return int(x[5:])
    elif len(x) == 11:
        return int(x[5:])
    elif len(x) == 9:
        return int(x[4:])
    else:
        return -1

# 简历：判断 职位城市 和 期望工作城市 是否一致(或包含于)
def is_same_user_city(df):
    jd_city_id = int(df['city'])
    desire_jd_city = df['desire_jd_city_id']
    return jd_city_id in desire_jd_city

# 计算 岗位标题 和 求职者的经验 的重合度
def jieba_cnt(df):
    experience = df['experience']    # 简历经验 + 专业 （对应 专业+ 工作经验）

    jd_title = df['jd_title']        # 职位标题  （对应职位名称）
    jd_sub_type = df['jd_sub_type']  # 职位子类+岗位描述 (对应职位关键字+职位描述)
    if isinstance(experience, str) and isinstance(jd_sub_type, str):
        # 对 职位标题 和 职位子类 采用搜索模式(使得分得的词比较短)分词，然后进行并集
        tmp_set = set(jb.cut_for_search(jd_title)) | set(jb.cut_for_search(jd_sub_type))
        # 对简历经验进行分词
        experience = set(jb.cut_for_search(experience))
        # 计算 岗位标题 和 求职者的经验 的重合度
        tmp_cnt = 0
        for t in tmp_set:
            if t in experience:
                tmp_cnt += 1
        return tmp_cnt
    else:
        return 0

# 判断 最近工作行业 和 期望工作行业 是否相同(有交集)
def cur_industry_in_desire(df):
    cur_industry_id = df['cur_industry_id']
    desire_jd_industry_id = df['desire_jd_industry_id']
    if isinstance(cur_industry_id, str) and isinstance(desire_jd_industry_id, str):
        return cur_industry_id in desire_jd_industry_id
    else:
        return -1

# 判断 简历期望职类 是否和 招聘职位子类 有交集
def desire_in_jd(df):
    desire_jd_type_id = df['desire_jd_type_id']
    jd_sub_type = df['jd_sub_type']
    if isinstance(jd_sub_type, str) and isinstance(desire_jd_type_id, str):
        return jd_sub_type in desire_jd_type_id
    else:
        return -1


def get_tfidf_(df, names, merge_id):
    # 文本->向量矩阵，ngram_range：允许使用1个词语和2个词语的组合
    tfidf_enc_tmp = TfidfVectorizer(ngram_range=(1, 2))
    # 对简历中的 experience 文本 -> 向量矩阵
    tfidf_vec_tmp = tfidf_enc_tmp.fit_transform(df[names])
    # 矩阵分解降维。(文本数，词汇数) -> (文本数，主题数) n_components：目标输出维度，n_iter：迭代次数
    svd_tag_tmp = TruncatedSVD(n_components=10, n_iter=20, random_state=2022)

    tag_svd_tmp = svd_tag_tmp.fit_transform(tfidf_vec_tmp)
    tag_svd_tmp = pd.DataFrame(tag_svd_tmp)
    # 对experience获得的特征矩阵 赋予列名
    tag_svd_tmp.columns = [f'{names}_svd_{i}' for i in range(10)]
    return pd.concat([df[[merge_id]], tag_svd_tmp], axis=1)

def get_tfidf(df, text_list, which_flag):

    tfidf_enc = TfidfVectorizer(ngram_range=(1, 2))
    tfidf_vec = tfidf_enc.fit_transform(text_list)
    # 降维
    svd_tag = TruncatedSVD(n_components=10, n_iter=20, random_state=2022)
    tag_svd = svd_tag.fit_transform(tfidf_vec)
    tag_svd = pd.DataFrame(tag_svd)
    # 赋予列名
    tag_svd.columns = [f'{which_flag}_tfidf_{i}' for i in range(10)]
    # 拼接，在列方向上
    df = pd.concat([df, tag_svd], axis=1)
    return df, tfidf_enc, svd_tag



# 读取停词表 添加不同形式的 空格
def get_stop_words(stop_words_path=r'D:\CodeFiles\data\stopwords-master\baidu_stopwords.txt'):
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



def offline_eval_map(train_df, label, pred_col):
    tmp_train = train_df.copy()
    tmp_train['rank'] = tmp_train.groupby('user_id')[pred_col].rank(ascending=False, method='first')
    tmp_x = tmp_train[tmp_train[label] == 1]
    tmp_x[f'{label}_index'] = tmp_x.groupby('user_id')['rank'].rank(ascending=True, method='first')
    tmp_x['score'] = tmp_x[f'{label}_index'] / tmp_train['rank']
    return tmp_x.groupby('user_id')['score'].mean().mean()


# 学历
degree_map = {'其他': 0, '初中': 1, '中技': 2, '中专': 2, '高中': 2, '大专': 3, '本科': 4,
              '硕士': 5, 'MBA': 5, 'EMBA': 5, '博士': 6}

def gen_jd_feats(jd_df, doc2vec_model, use_cut_json=True, use_tfidf=False, use_doc2vec=False):
    # 获取分词
    # if use_cut_json:
    #     with open(r"D:\CodeFiles\AI4recruitment\ai_recruitment\models\exp0\job_description_cut.json", "r",
    #               encoding="utf-8") as f1:
    #         tmp_cut = list(json.load(f1)['tmp_cut'])
    # else:
    #     # 对每个jd的岗位描述 做分词，tmp_cut：list, 长度为岗位的个数
    #     tmp_cut = Parallel(n_jobs=-1)(delayed(get_str)(jd_df.loc[ind]['job_description\n'])
    #                                   for ind in tqdm(jd_df.index))

    # save tmp_cut
    # job_description_cut = {'tmp_cut': tmp_cut}
    # with open(r"D:\CodeFiles\AI4recruitment\ai_recruitment\models\job_description_cut.json", "w", encoding="utf-8") as f1:
    #     # ensure_ascii=False才能输出中文，否则就是Unicode字符
    #     json.dump(job_description_cut, f1, indent=2, ensure_ascii=False)

    # 构建 tfidf 特征
    # if use_tfidf:
    #     jd_df, tfidf_enc_jd, svd_tag_jd = get_tfidf(jd_df, tmp_cut, which_flag='jd')

    if use_doc2vec:
        # 构建 Doc2Vec特征
        #   加载已经构建好的特征 jd_D2V.shape(269534, 20)
        with open(r'D:\CodeFiles\AI4recruitment\ai_recruitment\models\exp4\jd_D2V.dat', 'rb') as f1:
            jd_D2V = pickle.load(f1)
        #   从头加载
        # jd_D2V = Parallel(n_jobs=-1)(delayed(doc2vec_infer)(model=doc2vec_model, doc_text=sentence)
        #                               for sentence in tqdm(tmp_cut))

        # 将20维特征写入df中
        jd_D2V = pd.DataFrame(jd_D2V)
        # 赋予列名
        jd_D2V.columns = [f'jd_D2V_{i}' for i in range(20)]
        # 拼接，在列方向上
        jd_df = pd.concat([jd_df, jd_D2V], axis=1)

        del jd_D2V

    return jd_df


def gen_user_feats(user_df, doc2vec_model, use_tfidf=False, use_doc2vec=False):
    # 期望工作城市处理，一共有三个，如："691,698,-"， 匹配 所有数字,返回['691', '698'], 用于判断居住地和期望地是否一致

    def process_city_id(x):
        x = re.findall('\d+', x)
        if len(x) == 0 or x == ['0']:
            return [-1]
        else:
            return [int(i) for i in x]

    # user_df['desire_jd_city_id'].fillna([-1], inplace=True)
    user_df['desire_jd_city_id'] = user_df['desire_jd_city_id'].apply(lambda x: process_city_id(x))

    # 期望薪资转 str 类型
    user_df['desire_jd_salary_id'] = user_df['desire_jd_salary_id'].astype(str)
    # 期望薪资 最低值
    user_df['min_desire_salary'] = user_df['desire_jd_salary_id'].apply(get_min_salary)
    # 期望薪资最高值
    user_df['max_desire_salary'] = user_df['desire_jd_salary_id'].apply(get_max_salary)
    # 删除期望薪资列
    user_df.drop(['desire_jd_salary_id'], axis=1, inplace=True)

    # experience 分词
    user_df['experience'] = user_df['experience'].apply(lambda x: ' '.join(x.split('|')
                                                                           if isinstance(x, str) else 'nan'))
    # 构建 tfidf 特征
    if use_tfidf:
        user_df, tfidf_enc_user, svd_tag_user = get_tfidf(user_df, list(user_df['experience']), which_flag='user')

    # 构建Doc2Vec特征
    if use_doc2vec:
        # 使用已经构建好的
        with open(r'D:\CodeFiles\AI4recruitment\ai_recruitment\models\exp4\user_D2V.dat', 'rb') as f2:
            user_D2V = pickle.load(f2)
        # 从头构建
        # user_D2V = Parallel(n_jobs=-1)(delayed(doc2vec_infer)(model=doc2vec_model, doc_text=user_df.loc[ind]['experience'])
        #                                for ind in tqdm(user_df.index))


        # 将20维特征写入df中
        user_D2V = pd.DataFrame(user_D2V)
        # 赋予列名
        user_D2V.columns = [f'user_D2V_{i}' for i in range(20)]
        # 拼接，在列方向上
        user_df = pd.concat([user_df, user_D2V], axis=1)

        del user_D2V

    return user_df

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
    action_df.sort_values(['user_id', 'jd_no', 'delivered', 'satisfied'], inplace=True)
    # 删除 'user_id' 和 'jd_no' 一样的重复数据
    action_df = action_df.drop_duplicates(subset=['user_id', 'jd_no'], keep='last')
    # 筛选出 只在训练集出现过的jd_id
    action_df = action_df[action_df['jd_no'].isin(jd_df['jd_no'].unique())]
    # 按照user_id key，以action_df的user_id为基准合并
    train_df = action_df.merge(user_df, on='user_id', how='left')
    train_df = train_df.merge(jd_df, on='jd_no', how='left')

    del train_df['browsed']
    del user_df
    del jd_df
    return train_df


def gen_jd_user_feats(jd_user_df, use_doc2vec=False):
    # 特征工程

    # 期望工作城市 是否和 工作地城市一致
    jd_user_df['city'].fillna(-1, inplace=True)
    jd_user_df['city'] = jd_user_df['city'].astype(int)
    jd_user_df['same_user_city'] = jd_user_df.apply(is_same_user_city, axis=1).astype(int)

    jd_user_df['desire_jd_city'] = jd_user_df['desire_jd_city_id'].apply(lambda x: int(x[0]))

    # 学历匹配
    jd_user_df['min_edu_level'] = jd_user_df['min_edu_level'].apply(lambda x: x.strip() if isinstance(x, str) else x)
    jd_user_df['cur_degree_id'] = jd_user_df['cur_degree_id'].apply(lambda x: x.strip() if isinstance(x, str) else x)
    jd_user_df['min_edu_level_num'] = jd_user_df['min_edu_level'].map(degree_map)
    jd_user_df['cur_degree_id_num'] = jd_user_df['cur_degree_id'].map(degree_map)

    # jd_user_df['same_edu'] = (jd_user_df['min_edu_level'] == jd_user_df['cur_degree_id']).astype(int)
    jd_user_df['gt_edu'] = (jd_user_df['cur_degree_id_num'] >= jd_user_df['min_edu_level_num']).astype(int)

    # 薪资匹配
    jd_user_df['min_desire_salary_num'] = (jd_user_df['min_desire_salary'] <= jd_user_df['min_salary']).astype(int)
    jd_user_df['max_desire_salary_num'] = (jd_user_df['max_desire_salary'] <= jd_user_df['max_salary']).astype(int)

    # 期望职位 和 岗位职位 是否匹配
    jd_user_df['same_desire_industry'] = jd_user_df.apply(cur_industry_in_desire, axis=1).astype(int)
    jd_user_df['same_jd_sub'] = jd_user_df.apply(desire_in_jd, axis=1).astype(int)

    # TODO  更换为 d(期望行业, 所属行业), d(期望职位, 职位名称)

    jd_user_df.drop(['cur_degree_id_num', 'cur_degree_id', 'min_years',
                   'start_work_date', 'start_date', 'end_date', 'key', 'min_edu_level'], axis=1, inplace=True)


    # TODO 计算 d(岗位描述, 工作经验) 文本相似度
    # if use_doc2vec:
    #     jd_user_df = get_doc_vec_simi(jd_user_df)

    return jd_user_df




if __name__ == '__main__':
    train_data_path = r'F:\myProject\all_datasets\recruitment_data'
    train_jd = pd.read_csv(os.path.join(train_data_path, 'table2_jd.csv'), sep='\t')

    jd_df = gen_jd_feats(train_jd)









































