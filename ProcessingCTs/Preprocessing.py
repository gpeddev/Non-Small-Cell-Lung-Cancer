from ProcessingCTs.DatabaseCT import DatabaseCT as dbCT
from Paths import *

########################################################################################################################
#                                                       VAE 1                                                          #
########################################################################################################################
dbCT.windowing(dir_in=Original_Images, dir_out=WindowCT, wl=-600, ww=1500, slope=1, intercept=1024)
print("WindowCT completed")

dbCT.grayscale(WindowCT, GrayscaleCT)
print("GrayscaleCT completed")

dbCT.masking(GrayscaleCT, MaskCT, Original_Masks)
print("MaskCT completed")

dbCT.cropping_minimum(MaskCT, CropTumor)
print("CropTumor completed")

########################################################################################################################
#                                                       VAE 2                                                          #
########################################################################################################################
dbCT.create_windowed_masks(Original_Masks, NewMaskWindow, 104)
print("NewMaskWindow completed")

dbCT.masking(GrayscaleCT, MaskedCT, NewMaskWindow)
print("MaskedCT completed")

dbCT.cropping_minimum(MaskedCT, CroppedWindow)
print("CroppedWindow completed")

dbCT.slice_statistics(CropTumor)

