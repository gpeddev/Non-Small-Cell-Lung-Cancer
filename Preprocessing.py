# Fills the folder structure of our Data with the appropriate processed CTs and their masks

from ProcessingCTs.DatabaseCT import DatabaseCT as dbCT
from SupportCode.Paths import *

########################################################################################################################
#                                                       VAE 1                                                          #
########################################################################################################################
print("VAE 1")
dbCT.windowing(dir_in=Original_Images, dir_out=WindowCT, wl=-600, ww=1500, slope=1, intercept=-1024)
print("WindowCT ready")

dbCT.grayscale(WindowCT, GrayscaleCT)
print("GrayscaleCT ready")

dbCT.masking(GrayscaleCT, MaskedCT_VAE1, Original_Masks)
print("MaskedCT_VAE1 ready")

dbCT.cropping_minimum(MaskedCT_VAE1, CropTumor)
print("CropTumor ready")

print("\n")
########################################################################################################################
#                                                       VAE 2                                                          #
########################################################################################################################
print("VAE 2")
dbCT.create_windowed_masks(Original_Masks, NewMaskWindow, 104)
print("NewMaskWindow ready")

dbCT.masking(GrayscaleCT, MaskedCT_VAE2, NewMaskWindow)
print("MaskedCT_VAE2 ready")

dbCT.cropping_minimum(MaskedCT_VAE2, CroppedWindow)
print("CroppedWindow ready")

print("\nFinish Preprocessing Data")
