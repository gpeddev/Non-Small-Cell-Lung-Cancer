from sklearn.feature_selection import VarianceThreshold


def prepare_VAE_features():
    kfold_list_features = []
    for i in range(5):
        temp1 = pd.DataFrame.from_dict(pool_rslt[i], orient='index', columns=column_names_vae1)
        selector = VarianceThreshold()
        temp1 = pd.DataFrame(selector.fit_transform(temp1), temp1.index, temp1.columns)
        temp1 = temp1.reset_index()
        temp1.rename(columns={"index": "Case ID"}, inplace=True)
        # new_data => ready dataformat for use in machine learning algorithms
        new_data = data.loc[:, ["Case ID", "Recurrence"]]
        new_data = pd.merge(new_data, temp1)
        new_data["Recurrence"] = new_data["Recurrence"].replace("yes", 1)
        new_data["Recurrence"] = new_data["Recurrence"].replace("no", 0)
        # new_data.drop("Case ID", inplace=True, axis=1)
        kfold_list_features.append(new_data)