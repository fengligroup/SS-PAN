import numpy as np
from sklearn import neighbors
from sklearn.neural_network import MLPRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn import svm
import xgboost as xgb
from sklearn.ensemble import AdaBoostRegressor
from sklearn.linear_model import SGDRegressor
from sklearn.linear_model import Ridge
from sklearn.metrics import r2_score
import pandas as pd
import joblib

#the confirguration of pandas
pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows',None)

# The purpose of introducing this function to partition the dataset is to avoid the uneven distribution caused by random partitioning in 5-fold cross-validation.
def iid(x,y,iid_flag):
    x.insert(loc=len(x.columns),column='η',value=y)
    x.sort_values(by="η",inplace=True,ascending=False)
    x.reset_index(inplace=True,drop=True)
    x.reset_index(inplace=True,drop=False)
    data_train=x[x['index']%5!=iid_flag]#iid_flag is between 0 to 4
    data_test=x._append(data_train)
    data_test = data_test.drop_duplicates(keep=False)
    data_train = data_train.drop(['index'], axis=1)
    data_test = data_test.drop(['index'], axis=1)
    return data_train,data_test

#data
data = pd.read_excel("data/dataset.xlsx", sheet_name="import2Python")
y = data.loc[:, ["η"]]
x = data.drop(["η"], axis=1)

#model
KNN = neighbors.KNeighborsRegressor(n_neighbors=7, weights="uniform", algorithm="ball_tree", p=2)
MLP = MLPRegressor(solver='lbfgs', activation="logistic", hidden_layer_sizes=(6, 4), tol=0.0001)
SVM = svm.SVR(C=1, gamma=0.05, epsilon=0.02)
RF = RandomForestRegressor(n_estimators=55)
XGB = xgb.XGBRegressor(max_depth=3, learning_rate=0.03, n_estimators=110, gamma=0,
                       min_child_weight=7, subsample=0.9)
ADB = AdaBoostRegressor(n_estimators=130)
model_list = [KNN, MLP, SVM, RF, XGB, ADB]
model_name_list = ['KNN','MLP', 'SVM', 'RF', 'XGB', 'ADB']
feature_list=[['电子亲和力', '容许因子', 'B位电子亲和力(KJ/mol)', 'B位价电子距离(Å)'],
              ['电子亲和力', '容许因子', 'B位电子亲和力(KJ/mol)', 'A位第一电离能(KJ/mol)'],
              ['容许因子', '相对分子质量之比', 'B位价电子距离(Å)'],
              ['容许因子', 'B位价电子距离(Å)'],
              ['容许因子', 'B位电子亲和力(KJ/mol)', '电负性'],
              ['容许因子', 'B位电子亲和力(KJ/mol)']]

#baselearner training
sdata = pd.DataFrame()
for i in range(len(model_list)):
    y_predict = pd.DataFrame()
    model = model_list[i]
    selected_feature = data.loc[:, feature_list[i]]
    model.fit(selected_feature, y)
    #The training results of models such as MLP are influenced by the random seed, therefore, so the trained models are dumped, create a folder named "saved_model" if FileNotFoundError is reported here.
    joblib.dump(model, f'saved_model/{model_name_list[i]}.m')
    for j in range(0,5):
        selected_feature = data.loc[:, feature_list[i]]
        data_train, data_test = iid(selected_feature, y, j)
        x_train = data_train.drop(["η"], axis=1)
        y_train = data_train.loc[:, ["η"]]
        x_test = data_test.drop(["η"], axis=1)
        y_test = data_test.loc[:, ["η"]]
        model.fit(x_train,y_train)
        y_predict_tmp = pd.concat([y_test.reset_index(drop=True),pd.DataFrame(model.predict(x_test))],axis=1)
        y_predict = pd.concat([y_predict, y_predict_tmp],axis=0)
    if i==0:
        sdata.insert(0, 'True_η', y_predict.iloc[:,0])
    sdata.insert(i+1, model_name_list[i], y_predict.iloc[:,1])

sdata.reset_index(inplace=True,drop=True)
sx=sdata.drop(['True_η'],axis=1)
sy=sdata.loc[:,'True_η']
model_1 = svm.LinearSVR(
    epsilon=0,
    tol=0.0001,
    C=1,
    loss='epsilon_insensitive',
    fit_intercept=True,
    intercept_scaling=1,
    dual=True,
    verbose=0,
    random_state=None,
    max_iter=10000)
model_2=SGDRegressor(loss="epsilon_insensitive", fit_intercept=True, learning_rate='invscaling', eta0=0.01,epsilon=0.0)
model_3=Ridge();
model_1.fit(sx,sy)
joblib.dump(model_1,f"saved_model/stacking_SVC.m")
model_2.fit(sx,sy)
joblib.dump(model_1,f"saved_model/stacking_SGD.m")
model_3.fit(sx,sy)
joblib.dump(model_1,f"saved_model/stacking_Ridge.m")
y_predict_1=pd.DataFrame()
y_predict_2=pd.DataFrame()
y_predict_3=pd.DataFrame()
for j in range(0, 5):
    sx_tmp=sx.copy()
    sdata_train, sdata_test = iid(sx_tmp, sy, j)
    sx_train = sdata_train.drop(['η'], axis=1)
    sy_train = sdata_train.loc[:, ['η']]
    sx_test = sdata_test.drop(['η'], axis=1)
    sy_test = sdata_test.loc[:, ['η']]
    model_1.fit(sx_train, np.ravel(sy_train))
    model_2.fit(sx_train, np.ravel(sy_train))
    model_3.fit(sx_train, np.ravel(sy_train))
    y_predict_1_tmp = pd.concat([sy_test.reset_index(drop=True), pd.DataFrame(model_1.predict(sx_test))], axis=1)
    y_predict_1 = pd.concat([y_predict_1, y_predict_1_tmp], axis=0)
    y_predict_2_tmp = pd.concat([sy_test.reset_index(drop=True), pd.DataFrame(model_2.predict(sx_test))], axis=1)
    y_predict_2 = pd.concat([y_predict_2, y_predict_2_tmp], axis=0)
    y_predict_3_tmp = pd.concat([sy_test.reset_index(drop=True), pd.DataFrame(model_3.predict(sx_test))], axis=1)
    y_predict_3 = pd.concat([y_predict_3, y_predict_3_tmp], axis=0)
score_svr = r2_score(y_predict_1.iloc[:, 0], y_predict_1.iloc[:, 1])
score_lr = r2_score(y_predict_2.iloc[:, 0], y_predict_2.iloc[:, 1])
score_ridge = r2_score(y_predict_3.iloc[:, 0], y_predict_3.iloc[:, 1])
print(f'SVR:{y_predict_1}')
print(f'r2score:{score_svr}')
print(f'SGD:{y_predict_2}')
print(f'r2score:{score_lr}')
print(f'ridge:{y_predict_3}')
print(f'r2score:{score_ridge}')
