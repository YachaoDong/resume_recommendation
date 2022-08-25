# -*- coding: utf-8 -*-
'''
@Time    : 7/19/2022 11:35 AM
@Author  : dong.yachao
'''

import pandas as pd
import numpy as np
import pymysql
from pymysql.converters import escape_string
import re
import collections
import copy
from sklearn import preprocessing
from sklearn.cluster import KMeans
import yaml
from sklearn.neighbors import KNeighborsClassifier, RadiusNeighborsClassifier
# 距离度量方式
from sklearn.metrics.pairwise import cosine_distances, cosine_similarity, euclidean_distances


'''
1. 从数据库中读取各种表获取企业岗位、学生简历 数据;
2. 字段mapping，转换为能做特征工程的 user_df, jd_df, user_jd_df。
'''


# 1. 从数据库读取原始数据 raw_data
class Connect_SQL:
    def __init__(self, server, user, password, database):
        '''
        使用 pymssql 连接指定的数据库
        :param server: 数据库IP地址
        :param user_name: 数据库用户名称
        :param password: 数据库用户密码
        :param database: 连接的数据库名称
        '''
        self.server = server
        self.user = user
        self.password = password
        self.database = database

    def __get_connect(self):
        # 得到数据库连接信息，返回connection
        if not self.database:
            raise(NameError, "没有设置数据库信息")
        self.conn = pymysql.connect(host=self.server, user=self.user, password=self.password, database=self.database,
                                    charset='utf8')

        # 以下使用connection中的游标进行操作
        if not self.conn:
            raise (NameError, "连接数据库失败")
        else:
            return self.conn

    def exec_query(self, sql_cmd):
        '''
        执行查询语句
        返回一个包含tuple的list，list是元素的记录行，tuple记录每行的字段数值
        :param sql_cmd: mysql 执行操作的语句命令
        :return: data_frame, 使用 pandas读取数据库的数据结构
        '''
        self.conn = self.__get_connect()
        # 使用pandas执行sql查询语句
        data_frame = pd.read_sql(sql_cmd, self.conn)
        # 查询完毕关闭数据库连接
        self.conn.close()
        return data_frame



def get_data_from_sql(server='172.31.210.218',
                      user='root',
                      password='123456',
                      database='migration',
                      status=3,
                      query_jd_id=None,
                      query_user_id=None
                      ):
    # 1.连接数据库
    recruit_sql = Connect_SQL(server, user, password, database)

    # ------------------ 企业端端数据处理 ------------------------------- #
    # 2. 读取企业岗位端 数据表
    jd_key = 'recruit_id', 'workplace', 'industry', 'post_name','payment', 'educate_bg', 'demands', 'require', 'keyword','grade_require', 'post_code', 'industry_code'
    # [recruit_id, post_name岗位名称, post_code, occupation_name, occupation_code,
    #  demands岗位描述, description, educate_bg学历要求, payment面议, require职位要求, keyword关键字,
    #  welfare_tag福利, workplace工作地, grade_require, industry行业]
    if query_jd_id is not None:
        jd_df = recruit_sql.exec_query("SELECT '%s' FROM rec_position WHERE recruit_id='%s' AND status='%d'" % (jd_key, query_jd_id, status))
    else:
        jd_df = recruit_sql.exec_query("SELECT `recruit_id`, `workplace`, `industry`, `post_name`,`payment`, `educate_bg`, `demands`, `require`, `keyword`,`grade_require`, `post_code`, `industry_code` FROM rec_position WHERE status='%d'" % (status))
    # print('jd_df:', jd_df)
    # 2.1 jd_id 描述信息处理、合并
    # jd_df = jd_df[['recruit_id', 'workplace', 'industry', 'post_name',
    #               'payment', 'educate_bg', 'demands', 'require', 'keyword',
    #               'grade_require', 'post_code', 'industry_code']]
    # 1.None值处理
    jd_df.fillna({'workplace': -1, 'industry': "null", 'post_name': "null",
                  'payment': "null ~ null", 'educate_bg': "不限", 'demands': "null",
                  'require': "null", 'keyword': "null",
                  'grade_require': -1, 'post_code': -1, 'industry_code': -1}, inplace=True)
    # 2. 特征合并
    jd_df['job_description\n'] = jd_df['demands'] + jd_df['require'] + jd_df['keyword'] +\
                                 jd_df['industry'] + jd_df['post_name']

    # 3.特征选取、改名
    jd_df = jd_df[['recruit_id', 'workplace', 'payment', 'educate_bg',
                  'post_code', 'industry_code', 'job_description\n', 'grade_require']]
    rename_jd_dict = {'recruit_id': 'jd_no',
                      'industry_code': 'jd_industry_code'}
    jd_df = jd_df.rename(columns=rename_jd_dict)


    # ------------------ 学生端数据处理 ------------------------------- #
    # 3. 读取学生简历端 数据表
    user_intention_key = '`stu_id`,`base_code`,`position_name`,`position_code`,`industry`,`industry_code`,`start_salary`,`end_salary`'
    user_edu_exp_key = 'stu_id,education,major'
    user_work_exp_key = 'stu_id,position_name,job_content'
    if query_user_id is not None:
        query_user_id = str(query_user_id)
        print('query_user_id:', query_user_id)
        # 3.1学生期望表 [stu_id, position_name期望职位, base期望地点, industry期望行业, salary, create_time]
        user_intention_df = recruit_sql.exec_query(
            "SELECT `stu_id`,`base_code`,`position_name`,`position_code`,`industry`,`industry_code`,`start_salary`,`end_salary` FROM stu_job_hunting_intention WHERE stu_id='%s'" % (query_user_id))

        # 3.2学生教育经历表 [stu_id, education软件工程, major专业, create_time]
        user_edu_exp_df = recruit_sql.exec_query(
            "SELECT `stu_id`,`education`,`major` FROM stu_educational_experience WHERE stu_id='%s'" % (query_user_id))

        # 3.3学生工作经历表 [stu_id, position_name工作经历职位名称, job_content工作内容, create_time]
        user_work_exp_df = recruit_sql.exec_query(
            "SELECT `stu_id`,`position_name`,`job_content` FROM stu_work_experience WHERE stu_id='%s'" % (query_user_id))
    else:
        # 3.1学生期望表 [stu_id, position_name期望职位, base期望地点, industry期望行业, salary, create_time]
        user_intention_df = recruit_sql.exec_query(
            "SELECT `stu_id`,`base_code`,`position_name`,`position_code`,`industry`,`industry_code`,`start_salary`,`end_salary` FROM stu_job_hunting_intention")
        # 3.2学生教育经历表 [stu_id, education软件工程, major专业, create_time]
        user_edu_exp_df = recruit_sql.exec_query(
            "SELECT `stu_id`,`education`,`major` FROM stu_educational_experience")
        # 3.3学生工作经历表 [stu_id, position_name工作经历职位名称, job_content工作内容, create_time]
        user_work_exp_df = recruit_sql.exec_query(
            "SELECT `stu_id`,`position_name`,`job_content` FROM stu_work_experience")

    # 3.1 user_id 描述信息处理、合并

    # user_intention_df = user_intention_df[['stu_id', 'base_code',
    #                                       'position_name', 'position_code',
    #                                       'industry', 'industry_code',
    #                                       'start_salary', 'end_salary']]
    # 1.user期望信息合并
    if user_intention_df.empty:
        print('user_intention_df.empty')
        user_intention_df = user_intention_df.append({'stu_id': query_user_id, 'base_code': -1, 'position_name': "null", 'position_code': -1,
                  'industry': "null", 'industry_code': "null", 'start_salary': -1, 'end_salary': -1},
                       ignore_index=True)
    # None 值处理
    user_intention_df.fillna({'base_code': -1, 'position_name': "null", 'position_code': -1,
                  'industry': "null", 'industry_code': "null", 'start_salary': -1, 'end_salary': -1}, inplace=True)
    # 特征合并
    user_intention_df = user_intention_df.groupby(by='stu_id').agg(lambda x: ",".join(list(x.astype(str)))).reset_index()

    # 2.user教育经历
    if user_edu_exp_df.empty:
        print('user_edu_exp_df.empty')
        user_edu_exp_df = user_edu_exp_df.append(
            {'stu_id': query_user_id, 'education': "null", 'major': "null"}, ignore_index=True)

    # user_edu_exp_df = user_edu_exp_df[['stu_id', 'education', 'major']]
    # None处理
    user_edu_exp_df.fillna({'education': "null", 'major': "null"}, inplace=True)
    # 特征合并
    user_edu_exp_df = user_edu_exp_df.groupby(by='stu_id').agg(lambda x: ",".join(list(x.astype(str)))).reset_index()

    # 3.user工作经历
    if user_work_exp_df.empty:
        print('user_work_exp_df.empty')
        user_work_exp_df = user_work_exp_df.append(
            {'stu_id': query_user_id, 'position_name': 'null', 'job_content': 'null'}, ignore_index=True)

    # user_work_exp_df = user_work_exp_df[['stu_id', 'position_name', 'job_content']]
    # None处理
    user_work_exp_df.fillna({'position_name': 'null', 'job_content': 'null'}, inplace=True)
    # 按照id合并
    user_work_exp_df = user_work_exp_df.groupby(by='stu_id').agg(lambda x: ",".join(list(x.astype(str)))).reset_index()

    # 4.user_df 合并所有user信息
    user_df = user_intention_df.merge(user_edu_exp_df, on='stu_id', how='left')
    user_df = user_df.merge(user_work_exp_df, on='stu_id', how='left')
    # 此处position_name 重复，变为position_name_x, position_name_y
    user_df['job_content'] = user_df['job_content'].astype(str) + "," + \
                               user_df['position_name_x'].astype(str) + "," + \
                               user_df['position_name_y'].astype(str) + "," +\
                               user_df['major'].astype(str) + "," + \
                               user_df['industry'].astype(str)
    # user feats
    user_df = user_df[['stu_id', 'base_code',
                      'start_salary', 'end_salary', 'education',
                      'industry_code', 'position_code',
                      'job_content']]
    rename_user_dict = {'stu_id': 'user_id',
                        'job_content': 'experience',
                        'industry_code': 'user_industry_code'}
    user_df = user_df.rename(columns=rename_user_dict)

    return jd_df, user_df








# 测试使用
if __name__ == '__main__':
    pass












































