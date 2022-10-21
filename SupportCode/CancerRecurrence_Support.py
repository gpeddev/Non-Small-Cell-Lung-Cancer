from imblearn.pipeline import Pipeline
from sklearn.feature_selection import VarianceThreshold
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score
from sklearn.model_selection import GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Lasso

from Pooling.Pooling import pooling
import pandas as pd
import numpy as np


def get_clinical_data():
    data = pd.read_csv("ClinicalDataAnalysis/data.csv")
    data.reset_index()
    return data


def get_features_vectors(latent_space,model_path,image_source):
    print("Pooling VAE...")
    pool_rslt = pooling(model_path + "/Models", latent_space, image_source)
    print("Pooling VAE completed")

    kfold_list_features = []
    column_names_vae = []
    for i in range(latent_space * 2):
        column_names_vae.append("VAE1-" + str(i))
    for i in range(5):
        temp1 = pd.DataFrame.from_dict(pool_rslt[i], orient='index', columns=column_names_vae)
        selector = VarianceThreshold()
        temp1 = pd.DataFrame(selector.fit_transform(temp1), temp1.index, temp1.columns)
        temp1 = temp1.reset_index()
        temp1.rename(columns={"index": "Case ID"}, inplace=True)
        kfold_list_features.append(temp1)

    return kfold_list_features

def feature_selection_lasso(data, final_dataframes_list, model_path, baseline):

    dataset_result=[]
    testing_dataset_result=[]
    training_dataframe,testing_dataframe = split_dataframe(model_path,final_dataframes_list)

    for vae_counter in range(5):

        Y_train = training_dataframe[vae_counter]["Recurrence"]
        X_train = training_dataframe[vae_counter].drop("Recurrence", axis=1).drop("Case ID", axis=1)

        Y_test = testing_dataframe[vae_counter]["Recurrence"]
        X_test = testing_dataframe[vae_counter].drop("Case ID", axis=1).drop("Recurrence", axis=1)


        # ################################################### feature selection using LogisticRegression (L1 regularization)


        grid_search_pipeline = Pipeline([
            ('scaler', StandardScaler()),
            ('model', LogisticRegression(penalty='l1', solver='liblinear', max_iter=10000))
        ])

        search = GridSearchCV(grid_search_pipeline,
                              {'model__C': np.arange(0.01, 10, 0.01)},
                              cv=5, scoring="neg_mean_squared_error", verbose=1, n_jobs=-1
                              )

        search.fit(X_train, Y_train)
        optimalC = search.best_params_["model__C"]
        coefficients = search.best_estimator_.named_steps['model'].coef_
        importance = np.abs(coefficients)[0]
        features = X_train.columns
        survived_columns = features[importance != 0]
        print(f"Selected number of variables: {len(survived_columns)}")

        survived_columns=list(survived_columns)
        survived_columns.append("Case ID")
        survived_columns.append("Recurrence")
        train_dataset_recuced=training_dataframe[survived_columns]
        testing_dataset_reduced=testing_dataframe[survived_columns]

        testing_dataset_result.append(test_dataset_recuced)


        # ################################## final score of optimal value

        # create pipeline for final score. train on train dataset (including validation) and test on test dataset
        test_pipeline = Pipeline([
            ('scaler', StandardScaler()),
            ('model', LogisticRegression(penalty='l1', solver='liblinear', C=optimalC))
        ])
        test_pipeline.fit(X_train, Y_train)
        y_pred = test_pipeline.predict(X_test)
        f1score = f1_score(Y_test, y_pred, average="weighted")
        # Print performance metrics
        print(f"F1 score: {f1score}")

        # create final X_train and concat with Y_train

        final_X = X_train
        final_X["Recurrence"] = Y_train
        final_X["Case ID"] = train_dataset["Case ID"]
        final_X=final_X[survived_columns]
        # store result
        dataset_result.append(final_X)
    return dataset_result, testing_dataset_result




def merge_dataframes(baseline,clinical_data,vectors_features_list):
    final_data_list=[]
    if baseline is True:
        for vae_counter in range(5):
            feature_vector_id_recurrence = clinical_data.loc[:, ["Case ID", "Recurrence"]]
            feature_vector_id_recurrence = pd.merge(feature_vector_id_recurrence, vectors_features_list[vae_counter])
            feature_vector_id_recurrence["Recurrence"] = feature_vector_id_recurrence["Recurrence"].replace("yes", 1)
            feature_vector_id_recurrence["Recurrence"] = feature_vector_id_recurrence["Recurrence"].replace("no", 0)
            final_data_list.append(feature_vector_id_recurrence)
        return final_data_list
    else:
        for vae_counter in range(5):
            feature_vector_id_recurrence = clinical_data.loc[:, ["Case ID"]]
            feature_vector_id_recurrence = pd.merge(feature_vector_id_recurrence, vectors_features_list[vae_counter])

            final_data_list = pd.merge(clinical_data, feature_vector_id_recurrence)
            final_data_list["%GG"] = final_data_list["%GG"].replace('0%', 0)
            final_data_list["%GG"] = final_data_list["%GG"].replace('>0 - 25%', 1)
            final_data_list["%GG"] = final_data_list["%GG"].replace('50 - 75%', 2)
            final_data_list["%GG"] = final_data_list["%GG"].replace('25 - 50%', 3)
            final_data_list["%GG"] = final_data_list["%GG"].replace('75 - < 100%', 4)
            final_data_list["%GG"] = final_data_list["%GG"].replace('100%', 5)
            final_data_list["Recurrence"] = final_data_list["Recurrence"].replace("yes", 1)
            final_data_list["Recurrence"] = final_data_list["Recurrence"].replace("no", 0)

            dummies = ["Gender", "Ethnicity", "Smoking status", "Tumor Location (choice=RUL)",
                       "Tumor Location (choice=RML)",
                       "Tumor Location (choice=RLL)", "Tumor Location (choice=LUL)", "Tumor Location (choice=LLL)",
                       "Tumor Location (choice=L Lingula)", "Histology", "Pathological T stage", "Pathological N stage",
                       "Pathological M stage", "Histopathological Grade",
                       "Pleural invasion (elastic, visceral, or parietal)",
                       "Adjuvant Treatment", "Chemotherapy", "Radiation"]
            final_data_list = pd.get_dummies(final_data_list, columns=dummies)
        return final_data_list

def split_dataframe(model_path,final_dataframes_list):
    training_dataframe = []
    testing_dataframe = []
    for vae_counter in range(5):
        dataset_path = model_path + "/DatasetSplits"
        patient_names_test_dataset = np.load(dataset_path + "/test_dataset_fold_" + str(vae_counter + 1) + ".npy")
        # removes file extension
        patient_names_test_dataset = [patient.rsplit(".")[0] for patient in patient_names_test_dataset]

        # select rows whos Case ID is in test_dataset_1
        test_dataset = final_dataframes_list[
            final_dataframes_list['Case ID'].isin(patient_names_test_dataset)]
        # select rows whos Case ID is not in test_dataset_1
        train_dataset = final_dataframes_list[
            ~final_dataframes_list['Case ID'].isin(patient_names_test_dataset)]

        training_dataframe.append(train_dataset)
        testing_dataframe.append(test_dataset)

    return training_dataframe, testing_dataframe