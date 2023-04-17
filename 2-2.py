import statistics
import matplotlib.pyplot as plt
import scipy.stats as stats

age = [23, 23, 27, 27, 39, 41, 47, 49, 50, 52, 54, 54, 56, 57, 58, 58, 60, 61]
fat_rate = [9.5, 26.5, 7.8, 17.8, 31.4, 25.9, 27.4, 27.2, 31.2, 34.6, 42.5, 28.8, 33.4, 30.2, 34.1, 32.9, 41.2, 35.7]
# 问题（1）
print("问题（1）")
age_mean = statistics.mean(age)
fat_rate_mean = statistics.mean(fat_rate)
print("年龄的均值为：{}，体脂率的均值为：{}".format(age_mean, fat_rate_mean))
age_median = statistics.median(age)
fat_rate_median = statistics.median(fat_rate)
print("年龄的中位数为：{}，体脂率的中位数为：{}".format(age_median, fat_rate_median))
age_stdev = statistics.stdev(age)
fat_rate_stdev = statistics.stdev(fat_rate)
print("年龄的标准差为：{}，体脂率的标准差为：{}".format(age_stdev, fat_rate_stdev))

# 问题（2），盒图
print("问题（2）")
plt.boxplot([age, fat_rate])
plt.title("Two Datasets Boxplot")
plt.xlabel("Dataset")
plt.ylabel("Value")
plt.show()

# 问题（3），散点图和q-q图
print("问题（3）")
plt.scatter(age, fat_rate)
plt.title("Scatter Plot")
plt.xlabel("age")
plt.ylabel("fat_rate")
plt.show()
# 绘制 Q-Q 图
# 计算正态分布概率图
qqplot = stats.probplot(fat_rate, dist="norm", plot=plt)
plt.title("Q-Q Plot")
plt.xlabel("Theoretical Quantiles")
plt.ylabel("Ordered Values")
plt.show()



