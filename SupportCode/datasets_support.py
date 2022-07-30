import random
import SimpleITK as Sitk
import numpy as np


# _get_all_slices
# returns an array containing all slices of all the files in the file_list
# returned array shape => (number of all slices, slice_width, slice_height)
def _get_all_slices(filepath, file_list):
    results = []
    for filename in file_list:
        image_data = Sitk.GetArrayFromImage(Sitk.ReadImage(filepath+filename, Sitk.sitkUInt8))
        for i in range(image_data.shape[0]):
            results.append(image_data[i, :, :])
    random.shuffle(results)
    return np.array(results)


# preprocess_data
# 1. Converts all data to 0 - 1 range
# 2. Returns the data ready for our models shape=> (number of slides, slide width, slide height, 1)
def preprocess_data(path, filenames):
    selected_slices = _get_all_slices(path, filenames)
    selected_slices = selected_slices.astype("float32") / 255
    selected_slices = np.reshape(selected_slices,
                                 newshape=(selected_slices.shape[0],
                                           selected_slices.shape[1],
                                           selected_slices.shape[2],
                                           1)
                                 )
    return selected_slices
