#%%
import pandas as pd
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import MinMaxScaler
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
import lightgbm
import xgboost
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import learning_curve
# %%
players = pd.read_csv(r'.\data\Players.csv')
player_df = pd.read_csv(r'.\data\player_data.csv')
stat_df = pd.read_csv(r'.\data\Seasons_Stats.csv')
salary = pd.read_csv(r'.\data\NBA_season1718_salary.csv')
# %%
# 컬럼명 소문자 변환, player이름 컬럼 player로 통일
# unnamed 컬럼 삭제
cols = players.columns
players.columns = cols.str.lower()
players = players.drop(['unnamed: 0'], axis=1)

cols = player_df.columns
player_df.columns = cols.str.lower()
player_df = player_df.rename(columns={'name':'player'})

cols = stat_df.columns
stat_df.columns = cols.str.lower()
stat_df = stat_df.drop(['unnamed: 0'], axis=1)

cols = salary.columns
salary.columns = cols.str.lower()
salary = salary.drop(['unnamed: 0'], axis=1)
# %%
# 스탯이 연도별로 나오다보니 중복값이 있어서 merge에 문제가 생기므로 player로 groupby
# 중복이 있는 선수들은 스탯의 평균값을 사용함
stat_avg = stat_df.groupby(by='player').mean()
stat_avg = stat_avg.drop(['year'], axis=1)
stat_avg
# %%
# 연봉과 스탯 데이터를 merge함
stat_sal = pd.merge(stat_avg, salary, on='player', how='inner')
stat_sal
# %%
stat_sal.columns
# %%
stat_sal_pos = pd.merge(stat_sal, player_df, on = 'player', how='inner')
stat_sal_pos = stat_sal_pos.drop(['year_start', 'year_end', 'height','weight','birth_date','college'], axis=1)
stat_sal_pos
# %%
final_df = pd.merge(stat_sal_pos, players, on='player', how='inner')
final_df = final_df.drop(['collage', 'born','birth_city','birth_state'], axis=1)
final_df
# %%
# 결측값이 있는 컬럼 제거, salary값 있는 컬럼명 salary로 변경
final_df = final_df.drop(['blanl', 'blank2','3p%','2p%'], axis=1)
final_df = final_df.rename(columns={'season17_18':'salary'})
final_df
# %%
final_df.isna().sum()
# %%
# 타겟값인 salary값을 맨 마지막에 위치하게 함
Final_df = final_df.drop(['salary'], axis=1)
f_df = pd.concat([Final_df, final_df[['salary']]], axis=1)
print(f'최종 데이터 프레임 shape: {f_df.shape}')
#%%
# object 컬럼 레이블링
le = LabelEncoder()
a = f_df.select_dtypes('object')
a.columns
f_df['tm'] = le.fit_transform(f_df['tm'])
f_df['position'] = le.fit_transform(f_df['position'])
# %%
# 상관관계 그래프 그리기
cor = f_df.corr()
mask = np.zeros_like(cor, dtype=np.bool_)
mask[np.triu_indices_from(mask,1)] = True
fig = plt.figure(figsize=(30,30))
sns.heatmap(cor, annot=True, mask=mask,vmin=-1, vmax=1)
plt.title('salary correlation', size=30)
# %%
cor_df = pd.DataFrame(cor, columns=f_df.columns[1:])
# %%
# 상관관계 지수가 0.5보다 높은 컬럼만 추출
cor_list = list(cor_df[cor_df['salary']>=0.5].index)
cor_list
# %%
# 최종 데이터프레임
fin_df = f_df[cor_list]
# %%
# 스케일링
mmscale = MinMaxScaler()
fin_array = mmscale.fit_transform(fin_df)
fin_df = pd.DataFrame(fin_array, columns = cor_list)
fin_df
# %%
# PCA
pca = PCA(n_components=18)
pca_array = pca.fit_transform(fin_df.iloc[:,:-1])
pca_df = pd.DataFrame(pca_array)
print(sum(pca.explained_variance_ratio_))

pca = PCA(n_components=10)
pca_array = pca.fit_transform(fin_df.iloc[:,:-1])
pca_df = pd.DataFrame(pca_array)
print(sum(pca.explained_variance_ratio_))

pca = PCA(n_components=4)
pca_array = pca.fit_transform(fin_df.iloc[:,:-1])
pca_df = pd.DataFrame(pca_array)
print(sum(pca.explained_variance_ratio_))

pca = PCA(n_components=3)
pca_array = pca.fit_transform(fin_df.iloc[:,:-1])
pca_df = pd.DataFrame(pca_array)
print(sum(pca.explained_variance_ratio_))
# 주성분이 3개일 때 모형 설명 91% 가능한 것을 확인함
# %%
pca_df
# %%
X_data = pca_df
y_target = fin_df['salary']
# %%
# 데이터 분리
X_train, X_test, y_train, y_test = train_test_split(X_data, y_target, test_size=0.2, random_state=0)
print(f'X_train shape:{X_train.shape}')
print(f'y_train shape:{y_train.shape}')
print(f'X_test shape:{X_test.shape}')
print(f'y_test shape:{y_test.shape}')
# %%
# 모델 학습 및 평가
rf_reg = RandomForestRegressor()
lr_reg = LinearRegression()
lgbm_reg = lightgbm.LGBMRegressor()
xgb_reg = xgboost.XGBRegressor()

def trainmodel(model, X_train, y_train, X_test, y_test):
    model.fit(X_train, y_train)
    pred = model.predict(X_test)
    mse = mean_squared_error(y_test, pred)
    
    print(f'{model.__class__.__name__}')
    print(f'mean squared error : {mse}')
    
# %%
models = [rf_reg, lr_reg, lgbm_reg, xgb_reg]

for model in models:
    trainmodel(model, X_train, y_train, X_test, y_test)
# %%
