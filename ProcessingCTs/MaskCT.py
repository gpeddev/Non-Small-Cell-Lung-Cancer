from ProcessingCTs.CT import CT
from ProcessingCTs.SliceCalculations import SliceCalculations as Slcal
import numpy as np


class MaskCT(CT):
    """Class to encapsulate the functionality for the masks of the CTs"""

    @staticmethod
    def __new_mask_slide(ct_slice, row_center, col_center, width):
        if width > ct_slice.shape[0] or width > ct_slice.shape[1]:
            raise IndexError("Width out of image bounds")
        row_low = row_center - width
        row_upper = row_center + width
        col_low = col_center - width
        col_upper = col_center + width
        if row_center - width < 0:
            row_low = 0
            row_upper = width * 2
        if row_center + width > ct_slice.shape[0] - 1:
            row_low = ct_slice.shape[0] - 1 - width * 2
            row_upper = ct_slice.shape[0] - 1
        if col_center - width < 0:
            col_low = 0
            col_upper = width * 2
        if col_center + width > ct_slice.shape[1] - 1:
            col_low = ct_slice.shape[1] - 1 - width * 2
            col_upper = ct_slice.shape[1] - 1
        ct_slice[row_low: row_upper, col_low: col_upper] = 1
        return ct_slice

    def window_mask(self, width):
        mask_data = self.get_image_data()
        number_of_slices = mask_data.shape[0]
        result_list = []
        for item in range(number_of_slices):
            row_center, col_center = Slcal.find_center_per_slice(mask_data[item, :, :])
            result_list.append(self.__new_mask_slide(mask_data[item, :, :], row_center, col_center, width))

        result = np.array(result_list)
        self.set_image_data(result)
