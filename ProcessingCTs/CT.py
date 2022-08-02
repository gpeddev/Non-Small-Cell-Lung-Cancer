import SimpleITK as Sitk
import numpy as np
from ProcessingCTs.SliceCalculations import SliceCalculations as Slcal


class CT:
    """ This class handles CT as an object and encapsulate basic functions on a CT level"""
    def __init__(self, filepath):
        filename = filepath.split("/")[-1]
        self.__image_data = Sitk.GetArrayFromImage(Sitk.ReadImage(filepath, Sitk.sitkUInt8))
        self.__filename = filename

    def get_image_data(self):
        return self.__image_data

    def set_image_data(self, data):
        self.__image_data = data

    def set_filename(self, new_name):
        self.__filename = new_name

    def get_filename(self):
        return self.__filename

    def __get_slide(self, number):
        return self.__image_data[number, :, :]

    def save_image(self, filepath):
        Sitk.WriteImage(Sitk.GetImageFromArray(self.get_image_data()), filepath)

    def find_max_widths_per_ct(self):
        """Scans all slice of the CT and gives the max allowed width for both row and column"""
        number_of_slices = self.__image_data.shape[0]

        widths = []
        for slice_number in range(number_of_slices):
            row1, row2, col1, col2 = Slcal.find_min_max_per_slice(self.__get_slide(slice_number))
            row_width = row2 - row1
            col_width = col2 - col1
            if row_width % 2 == 0:
                row_width = int(row_width / 2)
                row_width += 1
            else:
                row_width = int(row_width / 2)

            if col_width % 2 == 0:
                col_width = int(col_width / 2)
                col_width += 1
            else:
                col_width = int(col_width / 2)

            widths.append(row_width)
            widths.append(col_width)

        return max(widths)

    @staticmethod
    def crop_per_ct(image, width):
        """Given a CT and the crop width, crops each slide of the CT based on the row, col center and width """
        data = Sitk.GetArrayFromImage(image)
        number_of_slices = data.shape[0]
        result_list = []

        for slice_number in range(number_of_slices):
            ct_slice = data[slice_number, :, :]
            row_center, col_center = Slcal.find_center_per_slice(ct_slice)
            if Slcal.problem_with_boundaries(ct_slice, row_center, col_center, width):
                ct_slice = Slcal.adjust_padding_center(ct_slice, row_center, col_center, width)
                row_center, col_center = Slcal.find_center_per_slice(ct_slice)
            temp_array = Slcal.crop_slice(ct_slice, row_center, col_center, width)
            result_list.append(temp_array)

        result = np.array(result_list)
        return result
