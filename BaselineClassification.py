# Imports
from sklearn.linear_model import LogisticRegression
import matplotlib.pyplot as plt
from sklearn.model_selection import cross_val_score
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from Pooling.Pooling import pooling
from SupportCode.Paths import CropTumor, CroppedWindow
from sklearn.metrics import accuracy_score
from sklearn.metrics import f1_score
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn.feature_selection import VarianceThreshold, SelectKBest, f_regression, f_classif
from group_lasso import LogisticGroupLasso
np.random.seed(0)
LogisticGroupLasso.LOG_LOSSES = True


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

# create pandas from dictionary
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
new_data=pd.merge(new_data, kfold_list_features1[0])
new_data=pd.merge(new_data,kfold_list_features2[0])

new_data["Recurrence"]=new_data["Recurrence"].replace("yes",1)
new_data["Recurrence"]=new_data["Recurrence"].replace("no",0)


new_data.drop("Case ID",inplace=True,axis=1)



Y_true=new_data["Recurrence"]
x=new_data.drop("Recurrence",axis=1)
# Normalization
numerics=x.select_dtypes(include=np.number).columns.tolist()
x[numerics]=(x[numerics]-x[numerics].min())/(x[numerics].max()-x[numerics].min())

counter=1
groups=list(range(1,1025,1))

grid_group_reg=0.0001
grid_group_reg_list=[]
accuracy=[]
while True:
    print(f"Trying with group_reg: {grid_group_reg}")
    pipe = Pipeline(
        memory=None,
        steps=[
            ("variable_selection",
             LogisticGroupLasso(
                 groups=groups,
                 group_reg=grid_group_reg,
                 l1_reg=0,
                 tol=1e-5,
                 supress_warning=True,
                 n_iter=100000)
             ),
            ("Classifier", SVC())
        ])
    if len(accuracy)==0:        # if first time just append accuracy and group_reg
        pipe.fit(x, Y_true)
        Y_pred = pipe.predict(x)
        sparsity_mask = pipe["variable_selection"].sparsity_mask_
        acc = (Y_pred == Y_true).mean()
        print(f"Number variables: {len(sparsity_mask)}")
        print(f"Number of chosen variables: {sparsity_mask.sum()}")
        print(f"Accuracy: {acc}")
        accuracy.append(acc)
        grid_group_reg_list.append(grid_group_reg)
        grid_group_reg = grid_group_reg / 10
    else:
        if accuracy[-1]<acc:        # stops when the accuracy stop increasing
            pipe.fit(x, Y_true)
            Y_pred = pipe.predict(x)
            sparsity_mask = pipe["variable_selection"].sparsity_mask_
            acc = (Y_pred == Y_true).mean()
            print(f"Number variables: {len(sparsity_mask)}")
            print(f"Number of chosen variables: {sparsity_mask.sum()}")
            print(f"Accuracy: {acc}")
            accuracy.append(acc)
            grid_group_reg_list.append(grid_group_reg)
            grid_group_reg=grid_group_reg/10
        else:                       # add last accuracy and grid_group_reg  and end searching.
            pipe.fit(x, Y_true)
            Y_pred = pipe.predict(x)
            sparsity_mask = pipe["variable_selection"].sparsity_mask_
            acc = (Y_pred == Y_true).mean()
            print(f"Number variables: {len(sparsity_mask)}")
            print(f"Number of chosen variables: {sparsity_mask.sum()}")
            print(f"Accuracy: {acc}")
            accuracy.append(acc)
            grid_group_reg_list.append(grid_group_reg)
            print("Finish group lasso optimizing")
            break
    group_lasso_results = x.loc[:, sparsity_mask]

# Set X
X=group_lasso_results

X_train, X_test, y_train, y_test = train_test_split(X, Y_true, test_size = 0.20)

svclassifier = SVC(kernel='linear')
svclassifier.fit(X_train, y_train)
y_pred = svclassifier.predict(X_test)
print(confusion_matrix(y_test, y_pred))
print(classification_report(y_test, y_pred))
scores = cross_val_score(svclassifier, X, Y_true, cv=5)
scores
print("%0.2f accuracy with a standard deviation of %0.2f" % (scores.mean(), scores.std()))

svclassifier = SVC(kernel='poly')
svclassifier.fit(X_train, y_train)
y_pred = svclassifier.predict(X_test)
print(confusion_matrix(y_test, y_pred))
print(classification_report(y_test, y_pred))
scores = cross_val_score(svclassifier, X, Y_true, cv=5)
scores
print("%0.2f accuracy with a standard deviation of %0.2f" % (scores.mean(), scores.std()))

svclassifier = SVC(kernel='rbf',degree=5)
svclassifier.fit(X_train, y_train)
y_pred = svclassifier.predict(X_test)
print(confusion_matrix(y_test, y_pred))
print(classification_report(y_test, y_pred))
scores = cross_val_score(svclassifier, X, Y_true, cv=5)
scores
print("%0.2f accuracy with a standard deviation of %0.2f" % (scores.mean(), scores.std()))


svclassifier = SVC(kernel='sigmoid')
svclassifier.fit(X_train, y_train)
y_pred = svclassifier.predict(X_test)
print(confusion_matrix(y_test, y_pred))
print(classification_report(y_test, y_pred))
scores = cross_val_score(svclassifier, X, Y_true, cv=5)
scores
print("%0.2f accuracy with a standard deviation of %0.2f" % (scores.mean(), scores.std()))