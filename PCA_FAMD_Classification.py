# ############################################################################################################## IMPORTS
import pandas as pd
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline
from sklearn.model_selection import GridSearchCV
from sklearn.svm import SVC

from SupportCode.Classification_Support import print_graph2
from SupportCode.FAMD_Classification_Support import get_data
from SupportCode.Paths import CropTumor, CroppedWindow
import numpy as np
import prince
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt


# ################################################################################## SELECT VARIABLES FOR REVIEWING CODE
latent_space = 256
baseline = True
model_path = "./BestResults/VAE_2/Model_1/20_2022-08-15_19_51_41"

# Use CropTumor for the first VAE(VAE-tumor), Use CroppedWindow for the second VAE(VAE-window)
#image_source = CropTumor
image_source = CroppedWindow


# ################################################################################################ LOAD AND PREPARE DATA
np.random.seed(0)
metric = "balanced_accuracy"

data=get_data(baseline,latent_space,model_path,image_source)
# baseline=True, data => Case ID, Recurrence, VAE-0... VAE-511
# baseline=False, data => Case ID, Clinical Data... Clinical Data ,Recurrence, VAE-0... VAE-511
grid_models=[]
testing_dataset=[]
for i in range(5):
    temp=data[i].drop("Case ID",axis=1)
    ################### load training and testing dataset

    dataset_path = model_path + "/DatasetSplits"
    patient_names_test_dataset = np.load(dataset_path + "/test_dataset_fold_" + str(i + 1) + ".npy")
    # removes file extension
    patient_names_test_dataset = [patient.rsplit(".")[0] for patient in patient_names_test_dataset]

    # select rows whos Case ID is in test_dataset_1
    test_dataset = data[i][data[i]['Case ID'].isin(patient_names_test_dataset)]
    # select rows whos Case ID is not in test_dataset_1
    train_dataset = data[i][~data[i]['Case ID'].isin(patient_names_test_dataset)]


    pca_test_dataset = test_dataset.drop("Case ID", axis=1)
    pca_train_dataset = train_dataset.drop("Case ID", axis=1)

########################################################################################################## Scaling
    # train scaler on the training dataset
    scaler = StandardScaler()
    scaler.fit(pca_train_dataset.drop("Recurrence", axis=1))

    # Training dataset
    # save Recurrence column
    temp_recurrence=pca_train_dataset[["Recurrence"]]
    # save column names and index names
    temp_columns = pca_train_dataset.drop("Recurrence", axis=1).columns
    temp_index = pca_train_dataset.index
    # scale data and return dataframe with the right columns and indexes
    pca_train_dataset = pd.DataFrame(scaler.transform(pca_train_dataset.drop("Recurrence", axis=1)),
                                     index=temp_index,
                                     columns=temp_columns)
    # restore Recurrence
    pca_train_dataset["Recurrence"] = temp_recurrence
    # Testing dataset
    temp_recurrence = pca_test_dataset[["Recurrence"]]
    # save column names and index names
    temp_columns = pca_test_dataset.drop("Recurrence", axis=1).columns
    temp_index = pca_test_dataset.index
    # scale data and return dataframe with the right columns and indexes
    pca_test_dataset = pd.DataFrame(scaler.transform(pca_test_dataset.drop("Recurrence", axis=1)),
                                     index=temp_index,
                                     columns=temp_columns)
    # restore Recurrence
    pca_test_dataset["Recurrence"] = temp_recurrence


    pca = prince.PCA(
        n_components=30,
        n_iter=3,
        rescale_with_mean=True,
        rescale_with_std=True,
        copy=True,
        check_input=True,
        engine='auto',
        random_state=42)

    pca = pca.fit(pca_train_dataset.drop("Recurrence",axis=1))

    pca_features = pca.transform(pca_train_dataset.drop("Recurrence",axis=1))
    pca_train_transformed = pca_features
    pca_train_transformed.columns = ["PCA-" + str(number) for number in range(30)]
    pca_train_transformed["Recurrence"]=pca_train_dataset[["Recurrence"]]


    pca_features = pca.transform(pca_test_dataset.drop("Recurrence", axis=1))
    pca_test_transformed = pca_features
    pca_test_transformed.columns = ["PCA-" + str(number) for number in range(30)]
    pca_test_transformed["Recurrence"] = pca_test_dataset[["Recurrence"]]


    # plt.bar(
    #     range(1, len(pca.explained_inertia_) + 1),
    #     pca.explained_inertia_
    # )
    #
    # plt.xlabel('PCA Feature')
    # plt.ylabel('Explained variance')
    # plt.title('Feature Explained Variance')
    # plt.show()
##################################################################################### SVM
    param_grid = {'model__C': [0.1, 1, 10, 100, 1000],
                  'model__gamma': [1, 0.1, 0.01, 0.001, 0.0001],
                  'model__kernel': ['rbf', "sigmoid", "poly", "linear"],
                  'model__degree': [3, 4, 5, 6, 7, 8, 9, 10]}
#                'model__degree': [3, 4, 5, 6, 7, 8, 9, 10, 11, 12]} # original try

    new_model = Pipeline([
                          ("oversampling", SMOTE(sampling_strategy=0.7)),
                          ('model', SVC(probability=True, class_weight="balanced", tol=1e-3))])

    grid = GridSearchCV(new_model, param_grid, refit=True, scoring="balanced_accuracy", verbose=2, n_jobs=12)
    # fitting the model for grid search
    grid.fit(pca_train_transformed.drop("Recurrence", axis=1), pca_train_transformed["Recurrence"])
    grid_models.append(grid)
    testing_dataset.append(pca_test_transformed)
    y_pred = grid.predict(pca_test_transformed.drop("Recurrence", axis=1))
    y_true = pca_test_transformed[["Recurrence"]]
    print("TRST")

y_pred=grid_models[0].predict(testing_dataset[0].drop("Recurrence", axis=1))
y_true=testing_dataset[0][["Recurrence"]]
print_graph2(grid_models,testing_dataset)