from sklearn.metrics import precision_score, recall_score, f1_score
import pandas as pd
import numpy as np
from scipy import stats
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import MinMaxScaler
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm

# 指标类工具
"""
    评价指标：
    返回一个字典，字典中包含精确率，召回率，f1-score
"""
def res_metrics(actual,predicted):
    # 精确率
    p = precision_score(actual, predicted, average='binary',pos_label=0)
    # 召回率
    r = recall_score(actual, predicted, average='binary',pos_label=0)
    # f1-score
    f1score = f1_score(actual, predicted, average='binary',pos_label=0)
    res = {"precision":p,"recall":r,"f1":f1score}
    return res

"""
    皮尔逊相关系数
"""
def pierxun(jibing,jibing_res):
    pass

# 数据处理类工具

"""
    kmp 算法
    寻找匹配的模式
    用于数据清洗
"""
def Get_next(p,next):
    nums = len(p)
    for m in range(1,nums):
        k = 1
        for i in range(0,m):

            if i + 1>=m - i:
                break
            if p[0:i + 1] == p[m - i - 1:m]:
                k = i + 2
        next.append(k)
def kmp_match(s,p,next):
    nums1 = len(s)
    nums2 = len(p)
    i=j=0
    while i != nums1 and j != nums2:

        if s[i] != p[j]:
            j = next[j]
            i += 1
        else :
            i += 1
            j += 1
    if j != nums2 :
        return -1
    else:
        return 0

"""
    3σ 去除极值
"""
def three_sigema(jibing):
    # 处理前数据有多少行
    qian = jibing.shape[0]
    col = jibing.columns.tolist()
    three_sigema_col = []
    # 先查看哪些特征是基本符合正态分布的
    for col_ in col:
        m = jibing[col_].mean()
        std = jibing[col_].std()
        k = stats.kstest(jibing[col_], 'norm', (m, std))
        # pvalue > 0.05 说明符合条件
        # 筛选出这些列存储到列表中
        if k[1] > 0.05:
            three_sigema_col.append(col_)
    for col_ in three_sigema_col:
        # 取一列
        df = jibing[col_]
        # 获取这一列中满足 3σ 条件的下标
        way = (df.mean() - 3 * df.std() < df) & (df.mean() + 3 * df.std() > df)
        index = np.arange(df.shape[0])[way]
        # 更新 jibing 留下这些下标对应的行
        jibing = jibing.iloc[index]
    # 处理后数据剩下多少行
    hou = jibing.shape[0]
    print("操作前有{}行\n操作后有{}行\n删去了{}行".format(qian,hou,qian-hou))
    # 重置索引
    jibing = jibing.reset_index(drop=True)
    # 返回 3σ 处理后的数据
    return jibing

"""
    归一化操作
"""
def guiyihua(jibing):
    col = jibing.columns.tolist()
    col = col[9:59]
    col.append("年龄")
    # 1. 实例化转换器（feature_range是归一化的范围，即最小值-最大值）
    transfer = MinMaxScaler(feature_range=(0, 1))
    # 2. 调用fit_transform （只需要处理特征）
    jibing[col] = transfer.fit_transform(jibing[col])
    return jibing

"""
    标准化操作
"""
def biaozhunhua(jibing):
    # 获取所有列的名称
    col = jibing.columns.tolist()
    # 获取需要归一化的列名
    col = col[9:59]
    col.append("年龄")
    # 标准化
    ss = StandardScaler()
    ss = ss.fit(jibing.loc[:, col])
    jibing.loc[:, col] = ss.transform(jibing.loc[:, col])
    return jibing

"""
    分箱操作：
    用于决策树和贝叶斯算法中，将连续的数据映射到几个区间中
    便于决策树做出决策
    便于贝叶斯计算概率
"""
def fenxiang(jibing):
    # 获取连续型变量的列名
    col = jibing.columns.tolist()
    col = col[9:59]
    col.append("年龄")
    # 数据总行数
    total = jibing.shape[0]
    # 对与每列进行操作
    for j in range(len(col)):
        # 获取最大最小值，划分区间
        # 分为7个等级
        min_ = jibing.loc[:, col[j]].min()
        max_ = jibing.loc[:, col[j]].max()
        qujian = np.linspace(min_, max_, 8)
        # 将数据划分到对应的等级中
        for i in range(total):
            if jibing.loc[i, col[j]] <= qujian[1]:
                jibing.loc[i, col[j]] = 1
            if jibing.loc[i, col[j]] <= qujian[2] and jibing.loc[i, col[j]] > qujian[1]:
                jibing.loc[i, col[j]] = 2
            if jibing.loc[i, col[j]] <= qujian[3] and jibing.loc[i, col[j]] > qujian[2]:
                jibing.loc[i, col[j]] = 3
            if jibing.loc[i, col[j]] <= qujian[4] and jibing.loc[i, col[j]] > qujian[3]:
                jibing.loc[i, col[j]] = 4
            if jibing.loc[i, col[j]] <= qujian[5] and jibing.loc[i, col[j]] > qujian[4]:
                jibing.loc[i, col[j]] = 5
            if jibing.loc[i, col[j]] <= qujian[6] and jibing.loc[i, col[j]] > qujian[5]:
                jibing.loc[i, col[j]] = 6
            if jibing.loc[i, col[j]] <= qujian[7] and jibing.loc[i, col[j]] > qujian[6]:
                jibing.loc[i, col[j]] = 7
    return jibing


# 画图类工具
"""
    设置字体
"""
def set_font():
    fm.FontProperties(fname='./myfont/Font.ttf')
    matplotlib.rc("font",family='Microsoft YaHei',weight="bold")

"""
    折线图
"""
def zhexiantu(index,y,title):
    plt.plot(index,y)
    plt.title(title)
    plt.show()