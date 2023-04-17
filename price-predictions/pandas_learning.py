import pandas as pd
import numpy as np

data = {
    'name':['Alice','Bob','Charles','David','Eric'],
    'year':[2017,2016,2015,2017,2017],
    'salary':[40000,24000,31000,20000,30000]
}
data = pd.DataFrame(data)

"""
查找行 .query(condition)
"""
#print(data.query('salary>20000'))
#print(data.salary)
# print(data[data.name=='Eric'])
# print(data.query('name=="Eric"'))
# print(data.query("name=='Eric' and salary>20000"))

"""
特定列选择
"""
#print(data.filter(items=['name', 'salary']))
#print(data['name'])

"""
模糊查找
"""
#print(data.filter(like='2',axis=0))   ##axis=0代表行，=1代表列
#print(data.filter(like='n',axis=1))

"""
分组
"""
#print(data.groupby(['year']).count())

data_2 = pd.DataFrame({
    'key1':['a','a','b','b','a'],
    'key2':['one','two','one','two','one'],
    'data1':np.random.randn(5),
    'data2':np.random.randn(5)
})
# print(data_2.groupby(['key1']).mean())
# print(data_2.groupby(['key1','key2']).mean())

"""
统计函数,mean,count,std,min,max
"""
#统计列不同的个数
# print(data_2)
# print(data_2.groupby(['key1']).count())
# print(data_2.groupby(['key1','key2']).count())

"""
排序
"""
# print(data_2.data2.sort_values())
# print(data_2.sort_values(by='key1'))
#优先级
# print(data_2.sort_values(by=['key1','key2','data1']))
# print(data_2.sort_values(by='key1',ascending=False))
"""
增删改查
"""
#选择一行，iloc,loc
# print(data_2)
# print(data_2.iloc[2])#单纯的数字index
# print(data_2.loc[1])#索引可以是string
# print(data_2.groupby(['key1']).count().iloc[0])
# print(data_2.groupby(['key1']).count().loc['a'])
#选择一列
# print(data_2.iloc[:,0])#第一列
# print(data_2.iloc[0,2])#某行某列
# print(data_2.iloc[0:2])#到第两行
# print(data_2.iloc[1:2])#从第几行开始
# print(data_2.iloc[0:-1])#-1全部
# print(data_2.iloc[0:2,2:])#后两列
#修改， .at,iat
# data_2.at[1,'data1'] = 2
# data_2.iat[2,3] = 3
# print(data_2)
#增加行append，loc
# print(data_2.append({'data1': 1.2, 'data2': 1.4, 'key1': 'b', 'key2': 'two'},ignore_index=True))
# data_2.loc[5]=['a','one',2,3]
#增加一列，assign，loc
# print(data_2.assign(yyy=[1,2,3,4,5]))
# data_2.loc[:,'yyy']=[1,2,3,4,5]
# 统计空值
print(data_2.isnull())


