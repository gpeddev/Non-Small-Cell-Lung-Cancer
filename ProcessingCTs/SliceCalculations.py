# Contains the operation needed on a slice level

import numpy as np
from statistics import mean


class SliceCalculations:
    """ Helper class to encapsulate the functions on a slice level"""

    @staticmethod
    def find_min_max_per_slice(ct_slice):
        """Find the min max index on a row and column level of each slice"""
        col_sum = np.where(np.sum(ct_slice, axis=0) > 0)
        row_sum = np.where(np.sum(ct_slice, axis=1) > 0)

        # row1 row minimum
        # row2 row maximum
        row1, row2 = row_sum[0][0], row_sum[0][-1]

        # col1 col minimum
        # col2 col maximum
        col1, col2 = col_sum[0][0], col_sum[0][-1]

        return row1, row2, col1, col2

    @staticmethod
    def find_center_per_slice(ct_slice):
        """Find the center on a row and column level."""
        row1, row2, col1, col2 = SliceCalculations.find_min_max_per_slice(ct_slice)
        row_center = int(mean([row1, row2]))
        col_center = int(mean([col1, col2]))
        return row_center, col_center

    @staticmethod
    def crop_slice(ct_slice, row_center, col_center, width):
        """Crop the slice based on the center and a window width arround it,"""
        return ct_slice[row_center - width: row_center + width, col_center - width: col_center + width]

    @staticmethod
    def problem_with_boundaries(ct_slice, row_center, col_center, width):
        """Check if based on row, column center and width if there is an out of bounds case"""
        if row_center-width < 0:
            return True
        if row_center+width > ct_slice.shape[0]-1:
            return True
        if col_center-width < 0:
            return True
        if col_center+width > ct_slice.shape[1]-1:
            return True
        return False

    @staticmethod
    def adjust_padding_center(ct_slice, row_center, col_center, width):
        """in case a window based on the center row and column isnt possible adds padding"""
        if row_center-width < 0:
            temp_pad = abs(row_center - width)
            ct_slice = np.pad(ct_slice, ((temp_pad, 0), (0, 0)), "constant")
            ct_slice = ct_slice[:-temp_pad, :]
            row_center = row_center + abs(row_center - width)
        if row_center+width > ct_slice.shape[0]-1:
            temp_pad = abs(row_center + width - ct_slice.shape[0])
            ct_slice = np.pad(ct_slice, ((0, temp_pad), (0, 0)), "constant")
            ct_slice = ct_slice[temp_pad:, :]
        if col_center-width < 0:
            temp_pad = abs(col_center - width)
            ct_slice = np.pad(ct_slice, ((0, 0), (temp_pad, 0)), "constant")
            ct_slice = ct_slice[:, :-temp_pad]
        if col_center+width > ct_slice.shape[1]-1:
            temp_pad = abs(col_center + width - ct_slice.shape[1])
            ct_slice = np.pad(ct_slice, ((0, 0), (0, temp_pad)), "constant")
            ct_slice = ct_slice[:, temp_pad:]
        return ct_slice
