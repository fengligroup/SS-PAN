from sklearn import neighbors
from sklearn.neural_network import MLPRegressor
from sklearn import svm
from sklearn.ensemble import RandomForestRegressor
import xgboost as xgb
from sklearn.ensemble import AdaBoostRegressor
from sklearn.model_selection import cross_val_score
import pandas as pd
def dfs(feature_number,dep,vis=-1,feature_subset=[]):
    if(dep==0):
        vis=-1
    if(dep==feature_number):
        x=t.loc[:,feature_subset]
        rmse=-1 * sum(cross_val_score(model, x, target, cv=5, scoring='neg_mean_squared_error'))/5
        print(f"特征子集：{feature_subset},rmse：{rmse}")
        return
    for i in range(10):
        if(9 - i < feature_number - dep - 1):#no more suffcient features to be selected
            continue
        if(i<=vis):
            continue
        feature_subset_tmp=feature_subset.copy()
        feature_subset_tmp.append(col_name_list[i])
        dfs(feature_number,dep+1,i,feature_subset=feature_subset_tmp)


t=pd.read_excel("data/dataset.xlsx", sheet_name="import2Python")
col_name_list=["原子序数之比",'电子亲和力','容许因子','相对分子质量之比','B位电子亲和力(KJ/mol)','电负性','B位价电子距离(Å)',
               'A位第一电离能(KJ/mol)','赝势核半径(a.u.)','A位相对分子质量']
x=t.loc[:,col_name_list]
target=t.loc[:,"η"]
KNN = neighbors.KNeighborsRegressor(n_neighbors=6, weights="uniform", algorithm="ball_tree", p=1)
MLP = MLPRegressor(solver='lbfgs', activation="logistic", alpha=0.01, hidden_layer_sizes=(6, 2), tol=0.0001,shuffle=False)
SVM = svm.SVR(C=1, gamma=0.1, epsilon=0.03)
RF = RandomForestRegressor(n_estimators=180, random_state=169)
XGB = xgb.XGBRegressor(max_depth=5, learning_rate=0.05, n_estimators=100, colsample_bytree=0.7, gamma=0.1,
                       min_child_weight=1, subsample=0.9)
ADB = AdaBoostRegressor(n_estimators=120,random_state=99)
model_list = [KNN, MLP, SVM, RF, XGB, ADB]
model_name_list = ['KNN', 'MLP', 'SVM', 'RF', 'XGB', 'ADB']
for i in range(0,6):
    print(model_name_list[i])
    model = model_list[i]
    for j in range(1,11):
        print(f"feature number:{j}")
        dfs(j,0)

