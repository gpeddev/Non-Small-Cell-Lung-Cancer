from random import random

import SimpleITK as Sitk
import numpy as np



# Helpers functions
# Returns an array with all the slices from the file_list CTs
def get_all_slices(filepath, file_list):
    results = []
    for filename in file_list:
        image_data = Sitk.GetArrayFromImage(Sitk.ReadImage(filepath+filename, Sitk.sitkUInt8))
        for i in range(image_data.shape[0]):
            results.append(image_data[i, :, :])
    return np.array(results)

def preprocess_data(path,filenames):
    selected_slices = get_all_slices(path, filenames)
#    random.shuffle(selected_slices)
    selected_slices = selected_slices.astype("float32") / 255
    selected_slices = np.reshape(selected_slices,
                              newshape=(selected_slices.shape[0],
                                        selected_slices.shape[1],
                                        selected_slices.shape[2],
                                        1))

    return selected_slices


# Make initial split to training and testing datasets(holdout)


