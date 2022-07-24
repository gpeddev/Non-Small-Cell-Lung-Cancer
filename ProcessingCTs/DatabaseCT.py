import os
import numpy as np
import SimpleITK as Sitk
from ProcessingCTs.ImageCT import ImageCT
from ProcessingCTs.MaskCT import MaskCT
from ProcessingCTs.CT import CT
import matplotlib.pyplot as plt


class DatabaseCT:

    @staticmethod
    def windowing(dir_in, dir_out, wl=-600, ww=1500, slope=1, intercept=-1024):
        for item in os.listdir(dir_in):
            image = Sitk.ReadImage(dir_in + item)
            casted_image = Sitk.Cast(image, Sitk.sitkInt32)
            image_data = Sitk.GetArrayFromImage(casted_image)
            image_data = image_data * slope + intercept  # slope * data + intercept
            image_data[image_data <= (wl - ww / 2)] = wl - ww / 2
            image_data[image_data > (wl + ww / 2)] = wl + ww / 2
            new_image = Sitk.GetImageFromArray(image_data)
            Sitk.WriteImage(new_image, dir_out + item)

    @staticmethod
    def grayscale(dir_in, dir_out):
        for item in os.listdir(dir_in):
            image = Sitk.ReadImage(dir_in + item)
            image_data = Sitk.GetArrayFromImage(image)
            min_image_value = np.min(image_data)
            max_image_value = np.max(image_data)
            image_data = (image_data - min_image_value) * ((255 - 0) / (max_image_value - min_image_value)) + 0
            new_image = Sitk.GetImageFromArray(image_data)
            new_image = Sitk.Cast(new_image, Sitk.sitkUInt8)
            Sitk.WriteImage(new_image, dir_out + item)

    @staticmethod
    def masking(dir_in, dir_out, mask_dir):
        for filename in os.listdir(dir_in):
            image = Sitk.ReadImage(dir_in + filename)
            mask_image = Sitk.ReadImage(mask_dir + filename.replace(".nii", "_roi.nii"))
            mask_image = Sitk.Cast(mask_image, Sitk.sitkUInt8)
            new_mask = Sitk.GetImageFromArray(Sitk.GetArrayFromImage(mask_image))
            result = Sitk.Mask(image, new_mask, outsideValue=0, maskingValue=0)
            Sitk.WriteImage(result, dir_out + filename)

    @staticmethod
    def cropping_minimum(dir_in, dir_out):
        print(dir_in)
        print(dir_out)
        max_width = DatabaseCT.__find_max_width_per_database(dir_in)
        for item in os.listdir(dir_in):
            image = Sitk.ReadImage(dir_in+item, Sitk.sitkUInt8)
            cropped_image_data = CT.crop_per_ct(image, max_width)
            Sitk.WriteImage(Sitk.GetImageFromArray(cropped_image_data), dir_out+item)

    @staticmethod
    def create_windowed_masks(dir_in, dir_out, window_width):
        for filename in os.listdir(dir_in):
            mask_image = MaskCT(dir_in+filename)
            try:
                mask_image.window_mask(window_width)
            except IndexError:
                print("Width is to big for file: ", filename)
                print("File mask is omitted.")
            else:
                mask_image.save_image(dir_out+filename)

    @staticmethod
    def __ct_file_list(filepath):
        file_list = os.listdir(filepath)
        return file_list

    @staticmethod
    def __find_max_width_per_database(dir_in):
        print("dir_in"+dir_in)
        widths = []
        for item in os.listdir(dir_in):
            image = ImageCT(dir_in + item)
            widths.append(image.find_max_widths_per_ct())
        return max(widths)

    @staticmethod
    def slice_statistics(dir_in):
        total = 0
        counter = 0
        min_slides = 1000000
        max_slides = -1000000
        lst = []
        for item in os.listdir(dir_in):
            image = ImageCT(dir_in + item)
            slide_number = image.get_image_data().shape[0]
            print("CT filename: ", image.get_filename())
            print("Number of slides: ", slide_number)
            min_slides = min(min_slides, slide_number)
            max_slides = max(max_slides, slide_number)
            total = total + slide_number
            counter = counter + 1
            lst.append(slide_number)

        print("\n\nOverall statistics")
        print("Total number of CTs: ", len(os.listdir(dir_in)))
        print("Total number of slides: ", total)
        print("Average number of slides: ", total/counter)
        print("Minimum number of slides", min_slides)
        print("Max number of slides", max_slides)
        plt.hist(lst, bins=50)
        plt.show()
