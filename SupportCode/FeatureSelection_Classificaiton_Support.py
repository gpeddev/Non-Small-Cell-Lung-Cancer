from group_lasso import LogisticGroupLasso
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score
from sklearn.model_selection import GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from Pooling.Pooling import pooling
import pandas as pd
import numpy as np
import asgl
from sklearn.model_selection import train_test_split, GridSearchCV


def _prepare_VAE_features(latent_space, model_path, image_source):
    # create new column names for VAE features based on latent space
    VAE_column_names = []
    for i in range(latent_space * 2):
        VAE_column_names.append("VAE-" + str(i))

    # load feature vectors from 5 kfold models at pool_rslt
    print("Pooling VAE")
    pool_rslt = pooling(model_path + "/Models", latent_space, image_source)

    # Create dataframe with VAE features, Case ID and Recurrence
    kfold_list_features = []
    for i in range(5):
        temp1 = pd.DataFrame.from_dict(pool_rslt[i], orient='index', columns=VAE_column_names)
        temp1 = temp1.reset_index()
        temp1.rename(columns={"index": "Case ID"}, inplace=True)

        # new_data => ready dataformat for use in machine learning algorithms
        data = pd.read_csv("ClinicalDataAnalysis/data.csv")
        data.reset_index()
        new_data = data.loc[:, ["Case ID", "Recurrence"]]
        new_data = pd.merge(new_data, temp1)
        new_data["Recurrence"] = new_data["Recurrence"].replace("yes", 1)
        new_data["Recurrence"] = new_data["Recurrence"].replace("no", 0)
        kfold_list_features.append(new_data)
    # kfold_list_features => CASE ID, Recurrence, VAE feature vectors
    return kfold_list_features


def _prepare_clinical_data():
    data = pd.read_csv("ClinicalDataAnalysis/data.csv")
    data.reset_index()
    temp = data
    temp["%GG"] = temp["%GG"].replace('0%', 0)
    temp["%GG"] = temp["%GG"].replace('>0 - 25%', 1)
    temp["%GG"] = temp["%GG"].replace('50 - 75%', 2)
    temp["%GG"] = temp["%GG"].replace('25 - 50%', 3)
    temp["%GG"] = temp["%GG"].replace('75 - < 100%', 4)
    temp["%GG"] = temp["%GG"].replace('100%', 5)
    temp.drop("Recurrence", inplace=True, axis=1)

    dummies = ["Gender", "Ethnicity", "Smoking status", "Tumor Location (choice=RUL)",
               "Tumor Location (choice=RML)",
               "Tumor Location (choice=RLL)", "Tumor Location (choice=LUL)", "Tumor Location (choice=LLL)",
               "Tumor Location (choice=L Lingula)", "Histology", "Pathological T stage", "Pathological N stage",
               "Pathological M stage", "Histopathological Grade",
               "Pleural invasion (elastic, visceral, or parietal)",
               "Adjuvant Treatment", "Chemotherapy", "Radiation"]
    clinical_data = pd.get_dummies(temp, columns=dummies)
    return clinical_data

def get_data(baseline,latent_space,model_path,image_source):
    final_data = []
    kfold_list_features=_prepare_VAE_features(latent_space, model_path, image_source)
    for i in range(5):
        if baseline is False:
            final_data.append(pd.merge(_prepare_clinical_data(),
                                       kfold_list_features[i]))
        else:
            final_data.append(kfold_list_features[i])
    return final_data
    # final_data contains the result.
    # final data when we have baseline contains
    # Case ID, Recurrence, VAE-0 .. VAE-511
    # final_data when not at baseline contains
    # Case ID, clinical data.... ,Recurrence, VAE-0 ... VAE-511

def apply_group_lasso(model_path, baseline, final_data):
    reduced_columns = []
    for i in range(5):
        dataset_path = model_path + "/DatasetSplits"
        patient_names_test_dataset = np.load(dataset_path + "/test_dataset_fold_" + str(i + 1) + ".npy")
        # removes file extension
        patient_names_test_dataset = [patient.rsplit(".")[0] for patient in patient_names_test_dataset]

        # select rows whos Case ID is in test_dataset_1
        test_dataset = final_data[i][final_data[i]['Case ID'].isin(patient_names_test_dataset)]
        # select rows whos Case ID is not in test_dataset_1
        train_dataset = final_data[i][~final_data[i]['Case ID'].isin(patient_names_test_dataset)]

        tempX = train_dataset.drop("Case ID", axis=1)
        tempY = tempX["Recurrence"]
        tempX = tempX.drop("Recurrence", axis=1)

        if baseline is True:

            pipe = Pipeline([
                ("scaler",
                 StandardScaler()),
                ("variable_selection",
                 LogisticGroupLasso(
                     #                 groups=list(range(1, 513)),
                     groups=[number for number in range(1, 513)],
#                     group_reg=0,
                     l1_reg=0,
                     n_iter=10000,
                     # scale_reg="inverse_group_size",
                     supress_warning=True, ),
                 ),
            ],
            )
            search = GridSearchCV(pipe,
                                  {'variable_selection__group_reg': [0.05]},
                                  cv=5, scoring="balanced_accuracy", verbose=3, n_jobs=-1
                                  )

            search.fit(tempX, tempY)

            optimalC = search.best_params_["variable_selection__group_reg"]
        else:
            ######  Create groups for columns
            col_group = [1, 2, 3, 4, 5, 6, 6, 7, 7, 7,
                         7, 7, 8, 8, 8, 9, 9, 10, 10, 11,
                         11, 12, 12, 13, 13, 14, 14, 15, 15, 15,
                         16, 16, 16, 16, 16, 16, 16, 18, 18, 18,
                         19, 19, 19, 20, 20, 20, 20, 20, 21, 21,
                         22, 22, 23, 23, 24, 24] + [number for number in range(25, 537)]

            pipe = Pipeline([
                ("scaler",
                 StandardScaler()),
                ("variable_selection",
                 LogisticGroupLasso(
                     groups=col_group,
                     l1_reg=0,
                     n_iter=100000,
                     scale_reg="inverse_group_size",
                     supress_warning=True, ),
                 ),
            ],
            )
            search = GridSearchCV(pipe,
                                  {'variable_selection__group_reg': [0.05, 0.005]},
                                  cv=5, scoring="f1_weighted", verbose=3, n_jobs=-1
                                  )
            search.fit(tempX, tempY)

            optimalC = search.best_params_["variable_selection__group_reg"]

        coefficients = search.best_estimator_.named_steps['variable_selection'].coef_
        importance = np.abs(coefficients)[:, 0]
        features = tempX.columns
        survived_columns = features[importance != 0]
        print(f"Selected number of variables: {len(survived_columns)}")
        reduced_columns.append(survived_columns)

    return reduced_columns


def apply_SVM(model_path,final_data,reduced_columns):
    grid_models = []
    testing_dataset = []
    for i in range(5):
        dataset_path = model_path + "/DatasetSplits"
        patient_names_test_dataset = np.load(dataset_path + "/test_dataset_fold_" + str(i + 1) + ".npy")
        # removes file extension
        patient_names_test_dataset = [patient.rsplit(".")[0] for patient in patient_names_test_dataset]

        # select rows whos Case ID is in test_dataset_1
        test_dataset = final_data[i][final_data[i]['Case ID'].isin(patient_names_test_dataset)]
        temp_columns = list(reduced_columns[i])
        temp_columns.append("Case ID")
        temp_columns.append("Recurrence")
        test_dataset = test_dataset[temp_columns]
        # select rows whos Case ID is not in test_dataset_1
        train_dataset = final_data[i][~final_data[i]['Case ID'].isin(patient_names_test_dataset)]
        train_dataset = train_dataset[temp_columns]

        param_grid = {'model__C': [0.1, 1, 10, 100, 1000],
                      'model__gamma': [1, 0.1, 0.01, 0.001, 0.0001],
                      'model__kernel': ['rbf', "sigmoid", "poly", "linear"],
                      'model__degree': [3, 4, 5, 6, 7, 8, 9, 10, 11, 12]}

        new_model = Pipeline([('scaler', StandardScaler()),
                              ("oversampling", SMOTE(sampling_strategy=0.7)),
                              ('model', SVC(probability=True, class_weight="balanced", tol=1e-3))])

        grid = GridSearchCV(new_model, param_grid, refit=True, scoring="balanced_accuracy", verbose=1, n_jobs=-1)
        # fitting the model for grid search
        grid.fit(train_dataset.drop("Case ID", axis=1).drop("Recurrence", axis=1), train_dataset["Recurrence"])
        grid_models.append(grid)
        testing_dataset.append(test_dataset)
        # print best parameter after tuning
        print(grid.best_params_)
        # print how our model looks after hyper-parameter tuning
        print(grid.best_estimator_)
    return grid_models,testing_dataset


def feature_selection_lasso(kfold_list_features):
    dataset_result=[]
    for vae_counter in range(5):

        # new_data => ready dataformat for use in machine learning algorithms
        new_data = kfold_list_features[vae_counter]
        new_data.drop("Case ID", inplace=True, axis=1)

        # Split to Y_true containing the true values
        # Split to X containing the data values
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
#        dataset_result.append(final_X)
        x=list(survived_columns)
        dataset_result.append(x.append("Case ID"))
    return dataset_result