
# ############################################################################################################## IMPORTS
from statistics import mean
import matplotlib.pyplot as plt
# from imblearn.over_sampling import SMOTE, RandomOverSampler, SVMSMOTE, ADASYN
from imblearn.over_sampling import RandomOverSampler, SMOTE, KMeansSMOTE, SVMSMOTE
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score, classification_report, ConfusionMatrixDisplay, RocCurveDisplay, auc
from sklearn.svm import SVC
from sklearn.metrics import confusion_matrix
from Pooling.Pooling import pooling
from SupportCode.Paths import CropTumor, CroppedWindow
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.model_selection import train_test_split, GridSearchCV
from imblearn.pipeline import Pipeline

# ################################################################################## SELECT VARIABLES FOR REVIEWING CODE
latent_space = 256
baseline = False
model_path = "./BestResults/VAE_2/Model_1/20_2022-08-15_19_51_41"

# Use CropTumor for the first VAE(VAE-tumor), Use CroppedWindow for the second VAE(VAE-window)
#image_source = CropTumor
image_source = CroppedWindow


# ################################################################################################ LOAD AND PREPARE DATA
np.random.seed(0)
metric = "balanced_accuracy"

# Load clinical data
data = pd.read_csv("ClinicalDataAnalysis/data.csv")
data.reset_index()

# create new column names
column_names_vae1 = []
for i in range(latent_space * 2):
    column_names_vae1.append("VAE1-"+str(i))

# load vectors from 5 kfold models
print("Pooling VAE")
pool_rslt_1 = pooling(model_path+"/Models", latent_space, image_source)

# ############################################################################################ PERFORM FEATURE SELECTION
f1_score_result = []
number_of_variables_result = []
dataset_result = []

for vae_counter in range(5):
    # create pandas from dictionary and create a columns with the features
    kfold_list_features = []
    for i in range(5):
        temp1 = pd.DataFrame.from_dict(pool_rslt_1[i], orient='index', columns=column_names_vae1)
        temp1 = temp1.reset_index()
        temp1.rename(columns={"index": "Case ID"}, inplace=True)
        kfold_list_features.append(temp1)

    new_data = data.loc[:, ["Case ID", "Recurrence"]]
    new_data = pd.merge(new_data, kfold_list_features[vae_counter])
    new_data["Recurrence"] = new_data["Recurrence"].replace("yes", 1)
    new_data["Recurrence"] = new_data["Recurrence"].replace("no", 0)
    new_data.drop("Case ID", inplace=True, axis=1)
    Y_true = new_data["Recurrence"]
    X = new_data.drop("Recurrence", axis=1)

    # ################################################### feature selection using LogisticRegression (L1 regularization)
    X_train, X_test, y_train, y_test = train_test_split(X, Y_true, test_size=0.3, random_state=1)

    grid_search_pipeline = Pipeline([
                         ('scaler', StandardScaler()),
                         ('model', LogisticRegression(penalty='l1', solver='liblinear', max_iter=1000))
    ])

    search = GridSearchCV(grid_search_pipeline,
                          {'model__C': np.arange(0.01, 10, 0.01)},
                          cv=5, scoring="balanced_accuracy", verbose=1, n_jobs=-1
                          )

    search.fit(X_train, y_train)
    optimalC = search.best_params_["model__C"]
    coefficients = search.best_estimator_.named_steps['model'].coef_
    importance = np.abs(coefficients)[0]
    features = X.columns
    survived_columns = features[importance != 0]
    print(f"Selected number of variables: {len(survived_columns)}")
    # ################################## final score of optimal value

    # create pipeline for final score. train on train dataset (including validation) and test on test dataset
    test_pipeline = Pipeline([
                         ('scaler', StandardScaler()),
                         ('model', LogisticRegression(penalty='l1', solver='liblinear', C=optimalC))
    ])
    test_pipeline.fit(X_train, y_train)
    y_pred = test_pipeline.predict(X_test)
    f1score = f1_score(y_test, y_pred, average="weighted")
    # Print performance metrics
    print(f"F1 score: {f1score}")

    # create final X and concat with Y_true
    final_X = X.loc[:, survived_columns]
    final_X["Recurrence"] = Y_true
    # store result
    f1_score_result.append(f1score)
    number_of_variables_result.append(len(survived_columns))
    dataset_result.append(final_X)
    print(f"Finish feature selection for fold :{vae_counter}")

for i in range(5):
    print(f"F1_weighted score: {f1_score_result[i]}")
    print(f"Number of variables: {number_of_variables_result[i]}")
print(f"Average f1_weighted score: {mean(f1_score_result)}")

# ################################################################################################################## SVM
grid_models = []
testing_dataset = []
for i in range(5):
    temp = dataset_result[i]

    dataset_path = model_path + "/DatasetSplits"
    patient_names_test_dataset = np.load(dataset_path + "/test_dataset_fold_" + str(i+1) + ".npy")
    # removes file extension
    patient_names_test_dataset = [patient.rsplit(".")[0] for patient in patient_names_test_dataset]

    temp["Case ID"] = data["Case ID"]

    # ################################################################################## Start Clinical Data Integration

    if baseline is False:
        temp = pd.merge(data, temp.drop("Recurrence", axis=1))
        temp["%GG"] = temp["%GG"].replace('0%', 0)
        temp["%GG"] = temp["%GG"].replace('>0 - 25%', 1)
        temp["%GG"] = temp["%GG"].replace('50 - 75%', 2)
        temp["%GG"] = temp["%GG"].replace('25 - 50%', 3)
        temp["%GG"] = temp["%GG"].replace('75 - < 100%', 4)
        temp["%GG"] = temp["%GG"].replace('100%', 5)
        temp["Recurrence"] = temp["Recurrence"].replace("yes", 1)
        temp["Recurrence"] = temp["Recurrence"].replace("no", 0)

        dummies = ["Gender", "Ethnicity", "Smoking status", "Tumor Location (choice=RUL)", "Tumor Location (choice=RML)",
                   "Tumor Location (choice=RLL)", "Tumor Location (choice=LUL)", "Tumor Location (choice=LLL)",
                   "Tumor Location (choice=L Lingula)", "Histology", "Pathological T stage", "Pathological N stage",
                   "Pathological M stage", "Histopathological Grade", "Pleural invasion (elastic, visceral, or parietal)",
                   "Adjuvant Treatment", "Chemotherapy", "Radiation"]
        temp = pd.get_dummies(temp, columns=dummies)
    # #################################################################################### End Clinical Data Integration

    # select rows whos Case ID is in test_dataset_1
    t_dataset = temp[temp['Case ID'].isin(patient_names_test_dataset)]
    # select rows whos Case ID is not in test_dataset_1
    tr_dataset = temp[~temp['Case ID'].isin(patient_names_test_dataset)]

    param_grid = {'model__C': [0.1, 1, 10, 100, 1000],
                  'model__gamma': [1, 0.1, 0.01, 0.001, 0.0001],
                  'model__kernel': ['rbf', "sigmoid", "poly", "linear"],
                  'model__degree': [3, 4, 5, 6, 7, 8, 9, 10, 11, 12]}

    new_model = Pipeline([('scaler', StandardScaler()),
                          # Only one oversampling per time
                          # uncomment to include RandomOverSampler 0.5
                          #("oversampling", RandomOverSampler(sampling_strategy=0.5)),
                          # uncomment to include RandomOverSampler 0.7
#                          ("oversampling", RandomOverSampler(sampling_strategy=0.7)),
                          # uncomment to include SMOTE 0.5
#                          ("oversampling", SMOTE(sampling_strategy=0.5)),
                          # uncomment to include SMOTE 0.7
                          ("oversampling", SMOTE(sampling_strategy=0.7)),
                          # uncomment to include SMOTE 1.0
                          #("oversampling", SMOTE(sampling_strategy=1.0)),
                          # uncomment to include SVMSMOTE 0.7
                          #("oversampling", SVMSMOTE(sampling_strategy=1)),
                          ('model', SVC(probability=True, class_weight="balanced",tol=1e-3))])

    grid = GridSearchCV(new_model, param_grid, refit=True, scoring=metric, verbose=1, n_jobs=-1)
    # fitting the model for grid search
    grid.fit(tr_dataset.drop("Case ID", axis=1).drop("Recurrence", axis=1), tr_dataset["Recurrence"])
    grid_models.append(grid)
    testing_dataset.append(t_dataset)
    # print best parameter after tuning
    print(grid.best_params_)
    # print how our model looks after hyper-parameter tuning
    print(grid.best_estimator_)

# ###################################################################### MODEL EVALUATION AND PRINT STATISTICS AND PLOTS

sens_rslts = []
spec_rslts = []

for i in range(5):
    g_prediction = grid_models[i].predict(testing_dataset[i].drop("Case ID", axis=1).drop("Recurrence", axis=1))

    # print classification report
    cl_report = classification_report(testing_dataset[i]["Recurrence"], g_prediction, zero_division=0)
    print(cl_report)
    cl_report = classification_report(testing_dataset[i]["Recurrence"], g_prediction, output_dict=True, zero_division=0)
    # in binary classification, recall of the positive class is also known as “sensitivity”;
    # recall of the negative class is “specificity”.
    sensitivity = cl_report["0"]["recall"]
    sens_rslts.append(sensitivity)
    specificity = cl_report["1"]["recall"]
    spec_rslts.append(specificity)

    # Confusion matrix
    comf_matrix = confusion_matrix(testing_dataset[i]["Recurrence"], g_prediction)
    ConfusionMatrixDisplay(comf_matrix, display_labels=['no recurrence', 'recurrence']).plot()
    plt.show()


print(f"Mean sensitivity of our model is {mean(sens_rslts):.4f} with std {np.std(sens_rslts):.4f}")
print(f"Mean specificity of our model is {mean(spec_rslts):.4f} with std {np.std(spec_rslts):.4f}")
g_prediction = grid_models[0].predict(testing_dataset[0].drop("Case ID", axis=1).drop("Recurrence", axis=1))
grid_models[0].predict_proba(testing_dataset[0].drop("Case ID", axis=1).drop("Recurrence", axis=1))

# ################################################################################################################ GRAPH
tprs = []
aucs = []
mean_fpr = np.linspace(0, 1, 100)
fig, ax = plt.subplots()

for i in range(5):
    viz = RocCurveDisplay.from_estimator(
        grid_models[i],
        testing_dataset[i].drop("Case ID", axis=1).drop("Recurrence", axis=1),
        testing_dataset[i]["Recurrence"],
        name="ROC {}".format(i),
        alpha=0.3,
        lw=1,
        ax=ax,
    )
    interp_tpr = np.interp(mean_fpr, viz.fpr, viz.tpr)
    interp_tpr[0] = 0.0
    tprs.append(interp_tpr)
    aucs.append(viz.roc_auc)

ax.plot([0, 1], [0, 1], linestyle="--", lw=2, color="r", label="Chance", alpha=0.8)

mean_tpr = np.mean(tprs, axis=0)
mean_tpr[-1] = 1.0
mean_auc = auc(mean_fpr, mean_tpr)
std_auc = np.std(aucs)
ax.plot(
    mean_fpr,
    mean_tpr,
    color="b",
    label=r"Mean ROC (AUC = %0.2f $\pm$ %0.2f)" % (mean_auc, std_auc),
    lw=2,
    alpha=0.8,
)

std_tpr = np.std(tprs, axis=0)
tprs_upper = np.minimum(mean_tpr + std_tpr, 1)
tprs_lower = np.maximum(mean_tpr - std_tpr, 0)
ax.fill_between(
    mean_fpr,
    tprs_lower,
    tprs_upper,
    color="grey",
    alpha=0.2,
    label=r"$\pm$ 1 std. dev.",
)

ax.set(
    xlim=[-0.05, 1.05],
    ylim=[-0.05, 1.05],
    title="Receiver operating characteristic example",
)
ax.legend(loc="lower right")
plt.show()
