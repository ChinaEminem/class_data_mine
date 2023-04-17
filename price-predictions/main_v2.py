import warnings
warnings.filterwarnings("ignore")     #忽略警告信息
import pandas_profiling as ppf        #EDA探索性数据分析
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns                 #绘图包
from scipy.stats import norm,skew     #获取统计信息
from scipy import stats
from scipy.special import boxcox1p
from sklearn.preprocessing import LabelEncoder
#加载数据集
train = pd.read_csv('./house-prices-advanced-regression-techniques/train.csv')
test = pd.read_csv('./house-prices-advanced-regression-techniques/test.csv')
#保存Id列
train_Id = train['Id']
test_Id = test['Id']
#删除原数据集的Id列
train.drop('Id',axis=1,inplace=True)
test.drop('Id',axis=1,inplace=True)

#经图像分析，SalePrice的分布呈现正偏态，对其做对数变换，让数据接近正态分布
train['SalePrice'] = np.log1p(train['SalePrice'])   #ln(1+x)
# #查看转换后的数据分布
# fix,ax = plt.subplots(nrows=2,figsize=(6,10))
# sns.distplot(train['SalePrice'],fit=norm,ax=ax[0])
# (mu,sigma) = norm.fit(train['SalePrice'])
# ax[0].legend(['Normal dist. ($\mu=$ {:.2f} and $\sigma=$ {:.2f}'.format(mu,sigma)],loc='best')
# stats.probplot(train['SalePrice'],plot=ax[1])
# plt.show()

#缺失值处理，训练集和测试集一并处理
all_data = pd.concat((train,test)).reset_index(drop=True)
#删除目标变量
all_data.drop(['SalePrice'],axis=1,inplace=True)
#统计缺失率
all_data_na = (all_data.isnull().sum()/len(all_data))*100
#删除缺失率为0的列，并降序排序
all_data_na = all_data_na.drop(all_data_na[all_data_na == 0].index).sort_values(ascending=False)
miss_data = pd.DataFrame({'Missing':all_data_na})
# #绘制特征缺失率的条形图
# fig,ax = plt.subplots(figsize=(15,8))
# sns.barplot(x=all_data_na.index,y=all_data_na)
# plt.xticks(rotation='90')
# plt.show()

#填补缺失值
for col in ('GarageType','GarageFinish','GarageQual','GarageCond','PoolQC','MiscFeature','Alley','Fence',
            'FireplaceQu','MasVnrType','BsmtQual','BsmtCond','BsmtExposure','BsmtFinType2','BsmtFinType1'):
    all_data[col] = all_data[col].fillna('None')
for col in ('GarageYrBlt','GarageArea','GarageCars','MasVnrArea','BsmtFinSF1','BsmtFinSF2','BsmtUnfSF',
            'TotalBsmtSF','BsmtFullBath','BsmtHalfBath'):
    all_data[col] = all_data[col].fillna(0)
#众数填补缺失较少的离散型特征值
for col in ('MSZoning','Electrical','KitchenQual','Exterior1st','Exterior2nd','SaleType'):
    all_data[col] = all_data[col].fillna(all_data[col].mode()[0])
#同一地区的到街道距离中位数填充LotFrontage
all_data['LotFrontage'] = all_data.groupby('Neighborhood')['LotFrontage'].transform(lambda x: x.fillna(x.median()))
#其他
all_data['Functional'] = all_data['Functional'].fillna('Typ')
#对于Utillties，所有记录只有一个NoSeWa和两个NA，其余全是Allpub,方差极小且NoSeWa没有出现在测试集中，故该特征对建模没有影响，删去
all_data.drop(columns='Utilities',inplace=True)
# #检查缺失值是否处理完毕
# print(all_data.isnull().sum().max())
#特征相关性
# report = ppf.ProfileReport(train)
# report.to_file('report.html')

#有一些特征虽然是数值型的，但其表征的意义只是不同的类别，其数值大小没有实际意义，将其转化为分类特征
for col in ('MSSubClass','YrSold','MoSold'):
    all_data[col] = all_data[col].astype(str)
#有一些特征类别实际上有高低好坏之分，将其映射为有大小的数字
cols = ('FireplaceQu','BsmtQual','BsmtCond','GarageQual','GarageCond','ExterQual','ExterCond',
        'HeatingQC','PoolQC','KitchenQual','BsmtFinType1','BsmtFinType2','Functional','Fence','BsmtExposure',
        'GarageFinish','LandSlope','LotShape','PavedDrive','Street','Alley','CentralAir','MSSubClass','OverallCond','YrSold','MoSold')
for col in cols:
    lbl = LabelEncoder()
    lbl.fit(list(all_data[col].values))
    all_data[col] = lbl.transform(list(all_data[col].values))

#利用与价格相关性强的项构建交互项，即添加新特征
#房屋总面积
all_data['TotalSF'] = all_data['TotalBsmtSF'] + all_data['1stFlrSF'] + all_data['2ndFlrSF']
#整体质量与房屋总面积交互项
all_data['OverallQual_TotalSF'] = all_data['OverallQual'] * all_data['TotalSF']
#整体质量与地上房间数交互项
all_data['OverallQual_GrLivArea'] = all_data['OverallQual'] * all_data['GrLivArea']
#整体质量与地上生活面积交互项
all_data['OverallQual_TotRmsAbvGrd'] = all_data['OverallQual'] * all_data['TotRmsAbvGrd']
#整体质量与车库面积交互项
all_data['OverallQual_GarageArea'] = all_data['OverallQual'] * all_data['GarageArea']
#建造时间与总面积交互项
all_data['YearBuilt_TotalSF'] = all_data['YearBuilt'] * all_data['TotalSF']

#对特征进行Box-Cox变换，尽量服从正态分布
#筛选出数值型特征
numeric_feature = all_data.dtypes[all_data.dtypes != 'object'].index
#计算特征的偏度
numeric_data = all_data[numeric_feature]
skew_features_compute = numeric_data.apply(lambda x: skew(x.dropna())).sort_values(ascending=False)
skewness = pd.DataFrame(skew_features_compute)
#对高偏度的特征进行Box_Cox变换(绝对值大于0.75)
new_skewness = skewness[skewness.abs()>0.75]
skew_features_boxcox = new_skewness.index
lam = 0.15
for feature in skew_features_boxcox:
    all_data[feature] = boxcox1p(all_data[feature],lam)
#one-hot编码
all_data = pd.get_dummies(all_data)
#建立模型
from sklearn.linear_model import ElasticNet,Lasso
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.kernel_ridge import KernelRidge
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import RobustScaler
from sklearn.base import BaseEstimator,TransformerMixin,RegressorMixin,clone
from sklearn.model_selection import KFold,cross_val_score,train_test_split
from sklearn.metrics import mean_squared_error
import xgboost as xgb
import lightgbm as lgb
#训练集目标变量
y_train = train.SalePrice.values
#训练集特征
train = all_data[:train.shape[0]]
#测试集特征
test = all_data[train.shape[0]:]
#评价函数。采用5折交叉验证，使用RMSE来为模型打分
n_folds = 5
def rmsle_cv(model):
    kf = KFold(n_folds,shuffle=True,random_state=42).get_n_splits(train.values)
    rmse = np.sqrt(-cross_val_score(model,train.values,y_train,scoring="neg_mean_squared_error",cv=kf))
    return (rmse)
#基本模型，LASSO Regression回归，该模型对异常值敏感，由于数据集存在一定的离群点，所有使用RobustScale对数据进行标准化处理
lasso = make_pipeline(RobustScaler(),Lasso(alpha=0.0005,random_state=1))
#岭回归
KRR = KernelRidge(alpha=0.6,kernel='polynomial',degree=2,coef0=2.5)
#弹性网络
ENet = make_pipeline(RobustScaler(),ElasticNet(alpha=0.0005,l1_ratio=0.9,random_state=3))
#梯度提升回归
GBoost = GradientBoostingRegressor(n_estimators=3000,learning_rate=0.05,max_depth=4,max_features='sqrt',
                                   min_samples_leaf=15,min_samples_split=10,loss='huber',random_state=5)
#XGBoost
model_xgb = xgb.XGBRegressor(colsample_bytree=0.5,gamma=0.05,learning_rate=0.05,max_depth=3,min_child=1.8,
                             n_estimators=2200,reg_alpha=0.5,reg_lambda=0.8,subsample=0.5,random_state=7,nthread=-1)
#LightGBM
model_lgb = lgb.LGBMRegressor(objective='regression',num_leaves=5,learning_rate=0.05,n_estimators=720,max_bin=55,
                              bagging_fraction=0.8,bagging_freq=5,feature_fraction=0.2,feature_fraction_seed=9,
                              bagging_seed=9,min_data_in_leaf=6,min_sum_hessian_in_leaf=11,verbose=-1)

#模型效果评价
models = {'Lasso':lasso,'ElasticNet':ENet,'Kernel Ridge':KRR,'Gradient Boosting':GBoost,'XGBoost':model_xgb,'LightGBm':model_lgb}
for model_name, model in models.items():
    score = rmsle_cv(model)
    print('{}:{:.4f} ({:.4f})\n'.format(model_name,score.mean(),score.std()))

# 堆叠方法
class StackingAverageModels(BaseEstimator,RegressorMixin,TransformerMixin):
    def __init__(self,base_models,meta_model,n_fold=5):
        self.base_models = base_models#第一层模型
        self.meta_model = meta_model#第二层模型
        self.n_fold = n_fold
#运用克隆的基本模型拟合数据
    def fit(self, X, y):
        self.base_models_ = [list() for x in self.base_models]
        self.meta_model_ = clone(self.meta_model)
        kfold = KFold(n_splits=self.n_fold, shuffle=True, random_state=156)
        #训练克隆的第一层模型
        out_of_fold_predictions = np.zeros((X.shape[0], len(self.base_models)))
        for i, model in enumerate(self.base_models):
            for train_index, holdout_index in kfold.split(X, y):
                instance = clone(model)
                self.base_models_[i].append(instance)
                instance.fit(X[train_index],y[train_index])
                y_pred = instance.predict(X[holdout_index])
                out_of_fold_predictions[holdout_index,i] = y_pred
        #使用交叉验证预测的结果作为新特征，来训练克隆的第二层模型
        self.meta_model_.fit(out_of_fold_predictions,y)
        return self

    def predict(self,X):
        meta_features = np.column_stack([np.column_stack([model.predict(X) for model in base_models]).mean(axis=1)
                                         for base_models in self.base_models_])
        return self.meta_model_.predict(meta_features)

#用ENet\KRR\GBoost作为第一层学习器，用Lasso作为第二层学习器，查看stacking的交叉验证评分
stacked_averaged_models = StackingAverageModels(base_models=(ENet,GBoost,KRR), meta_model=lasso)
# score = rmsle_cv(stacked_averaged_models)
# print(score.mean(),score.std())
#预测值与实际值之间的RMSE
def rmsle(y,y_pred):
    return np.sqrt(mean_squared_error(y,y_pred))
#StackedRegressor
#用整个训练集训练数据,预测测试集的房价,给出模型在训练集上的评分
stacked_averaged_models.fit(train.values,y_train)
stacked_train_pred = stacked_averaged_models.predict(train.values)
stacked_pred = np.expm1(stacked_averaged_models.predict(test.values))
# print(rmsle(y_train, stacked_train_pred))
#XGBoost
model_xgb.fit(train,y_train)
xgb_train_pred = model_xgb.predict(train)
xgb_pred = np.expm1(model_xgb.predict(test))
# print(rmsle(y_train,xgb_train_pred))
#LightGBM
model_lgb.fit(train,y_train)
lgb_train_pred = model_lgb.predict(train)
lgb_pred = np.expm1(model_lgb.predict(test))
print(rmsle(y_train,lgb_train_pred))

#生成预测结果
ensemble = stacked_pred*0.7 + xgb_pred*0.15 + lgb_pred*0.15
sub = pd.DataFrame()
sub['Id'] = test_Id
sub['SalePrice'] = ensemble
sub.to_csv('submission.csv',index=False)


