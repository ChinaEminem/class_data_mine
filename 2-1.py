import matplotlib.pyplot as plt
from collections import Counter
import numpy as np
import seaborn as sns

#  问题（1），均值和中位数
print("问题（1）")
age = [13, 16, 52, 16, 20, 20, 21, 22, 25, 35, 25, 25, 30, 33, 33, 19, 35, 22, 35, 35, 15, 35, 36, 40, 45, 25, 46, 70]
print("该组年龄数据的平均值为：{}".format(sum(age)/len(age)))

age.sort()  # 先将列表排序
length = len(age)
if length % 2 == 0:  # 判断列表长度是奇数还是偶数
    median = (age[length//2 - 1] + age[length//2]) / 2
else:
    median = age[length//2]
print("中位数为：{}\n".format(median))


# 问题（2），众数,模态
print("问题（2）")
count_dict = Counter(age)  # 使用Counter函数统计每个元素出现的次数
max_count = max(count_dict.values())  # 找出出现次数最多的元素的出现次数
mode = [k for k, v in count_dict.items() if v == max_count]  # 找出所有出现次数等于最大次数的元素
print("众数为：{}".format(mode))

sorted_dict = sorted(count_dict.items(), key=lambda x: x[1], reverse=True)  # 对字典按value进行降序排序
print("各年龄出现次数为：{}".format(sorted_dict))
print("所以是单模态\n")


# 问题（3），中列数
print("问题（3）")
midrange = (max(age) + min(age)) / 2
print("中列数为：{}\n".format(midrange))


# 问题（4），第一个四分位数和第三个四分位数
print("问题（4）")
q1 = np.percentile(age, 25)
q3 = np.percentile(age, 75)
print("第一个四分位数Q1为：{}".format(q1))
print("第三个四分位数Q3为：{}\n".format(q3))


# 问题（5），五数概括
print("问题（5）")
minimum = np.min(age)
maximum = np.max(age)
print("最小值为：{}".format(minimum))
print("第一个四分位数Q1为：{}".format(q1))
print("中位数为：{}".format(median))
print("第三个四分位数Q3为：{}".format(q3))
print("最大值为：{}\n".format(maximum))


# 问题（6），盒图
print("问题（6）")
plt.boxplot(age)
plt.show()


# 问题（7），分数位图
print("分数位-分数位图是一种将数据映射到二维平面上的图形，其中 x 轴表示一个分位数（比如 25%、50%、75%），y 轴表示另一个分位数。每个数据点被映射到这个平面上的一个位置，它的横坐标是它在第一个分位数上的排名，纵坐标是它在第二个分位数上的排名。分数位-分数位图通常用于比较两个数据集之间的关系，例如两个时间序列之间的相关性。"\

"相比之下，分位图则是一种将数据映射到一维坐标轴上的图形，其中 x 轴表示数据的值，y 轴表示数据的密度或频率。分位图通常用于探索数据的分布情况，可以显示出数据的中心位置、离散程度和异常值等信息。常见的分位图包括箱线图和密度图等。")