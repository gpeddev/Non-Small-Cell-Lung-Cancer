# Imports
from statistics import mean

from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score

from Pooling.Pooling import pooling
from SupportCode.Paths import CropTumor, CroppedWindow
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split, GridSearchCV

np.random.seed(0)


vae_1_latent_space=256
vae_2_latent_space=256
# Load clinical data
data = pd.read_csv("ClinicalDataAnalysis/data.csv")
data.reset_index()
# create new column names
column_names_vae1=[]
for i in range(vae_1_latent_space*2):
    column_names_vae1.append("VAE1-"+str(i))
column_names_vae2=[]
for i in range(vae_2_latent_space*2):
    column_names_vae2.append("VAE2-"+str(i))

# load vectors from 5 kfold models
print("Pooling VAE 1")
pool_rslt_1=pooling("./BestResults/VAE_1/Model_1/19_2022-08-10_01_53_19/Models", vae_1_latent_space, CropTumor)
print("Pooling VAE 2")
pool_rslt_2=pooling("./BestResults/VAE_2/Model_1/17_2022-08-15_19_47_42/Models", vae_2_latent_space, CroppedWindow)

f1_score_result=[]
number_of_variables_result=[]
dataset_result=[]

for vae_counter in range(5):
    # create pandas from dictionary and create a columns with the features
    kfold_list_features1=[]
    kfold_list_features2=[]
    for i in range(5):
        temp1 = pd.DataFrame.from_dict(pool_rslt_1[i], orient='index', columns=column_names_vae1)
        temp1 = temp1.reset_index()
        temp1.rename(columns={"index": "Case ID"}, inplace=True)
        temp2 = pd.DataFrame.from_dict(pool_rslt_2[i], orient='index', columns=column_names_vae2)
        temp2 = temp2.reset_index()
        temp2.rename(columns={"index": "Case ID"}, inplace=True)
        kfold_list_features1.append(temp1)
        kfold_list_features2.append(temp2)

    new_data=data.loc[:,["Case ID","Recurrence"]]

    ################################ work with one to begin with
    new_data=pd.merge(new_data, kfold_list_features1[vae_counter])
    new_data=pd.merge(new_data,kfold_list_features2[vae_counter])

    new_data["Recurrence"]=new_data["Recurrence"].replace("yes",1)
    new_data["Recurrence"]=new_data["Recurrence"].replace("no",0)

    new_data.drop("Case ID",inplace=True,axis=1)

    Y_true=new_data["Recurrence"]
    X=new_data.drop("Recurrence", axis=1)

    ######################################## feature selection of VAE1 and VAE2 using LogisticRegression (L1 regularization)
    X_train, X_test, y_train, y_test = train_test_split(X, Y_true, test_size=0.3, random_state=1)

    grid_search_pipeline = Pipeline([
                         ('scaler',StandardScaler()),
                         ('model',LogisticRegression(penalty='l1', solver='liblinear'))
    ])

    search = GridSearchCV(grid_search_pipeline,
                          {'model__C':np.arange(0.01,10,0.01)},
                          cv = 5, scoring="f1_weighted", verbose=3
                          )

    search.fit(X_train,y_train)

    search.best_params_

    optimalC=search.best_params_["model__C"]
    coefficients = search.best_estimator_.named_steps['model'].coef_

    importance = np.abs(coefficients)[0]

    features=X.columns
    importance!=0
    survived_columns=features[importance!=0]
    print(f"Selected number of variables: {len(survived_columns)}")
    ################################### final score of optimal value

    # create pipeline for final score
    test_pipeline = Pipeline([
                         ('scaler',StandardScaler()),
                         ('model',LogisticRegression(penalty='l1', solver='liblinear',C=optimalC))
    ])
    test_pipeline.fit(X_train,y_train)
    y_pred=test_pipeline.predict(X_test)

    f1score = f1_score(y_test, y_pred,average="weighted")
    acc= (y_pred == y_test).mean()
    # Print performance metrics
    print(f"F1 score: {f1score}")

    # create final X and concat with Y_true
    final_X=X.loc[:,survived_columns]
    final_X["Recurrence"]=Y_true

    # store result
    f1_score_result.append(f1score)
    number_of_variables_result.append(len(survived_columns))
    dataset_result.append(final_X)
    print(f"Finish feature selection for fold :{vae_counter}")

for i in range(5):
    print(f"F1_weighted score: {f1_score_result[i]}")
    print(f"Number of variables: {number_of_variables_result[i]}")
print(f"Average f1_weighted score: {mean(f1_score_result)}")




