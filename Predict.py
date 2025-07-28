import pandas as pd
import numpy as np
import joblib
x=pd.read_excel('data/latentSpace.xlsx',sheet_name="import2Python")
model_name_list = ['KNN','MLP','SVM', 'RF', 'XGB', 'ADB']
feature_list=[['电子亲和力', '容许因子', 'B位电子亲和力(KJ/mol)', 'B位价电子距离(Å)'],
              ['电子亲和力', '容许因子', 'B位电子亲和力(KJ/mol)', 'A位第一电离能(KJ/mol)'],
              ['容许因子', '相对分子质量之比', 'B位价电子距离(Å)'],
              ['容许因子', 'B位价电子距离(Å)'],
              ['容许因子', 'B位电子亲和力(KJ/mol)', '电负性'],
              ['容许因子', 'B位电子亲和力(KJ/mol)']]
y=np.zeros((811,6))

model=joblib.load(f"saved_model/stacking_SVC.m")
z=model.predict(y)
z=pd.DataFrame(z,columns=['SVR'])
z.to_excel('predict_meta.xlsx')