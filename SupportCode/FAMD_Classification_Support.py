import pandas as pd

from SupportCode.FeatureSelection_Classificaiton_Support import _prepare_VAE_features


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
    clinical_data = temp
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