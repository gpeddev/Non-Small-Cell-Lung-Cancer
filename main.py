from ProcessingCTs.DatabaseCT import DatabaseCT as dbCT

Original_Images = "./Data/00_Images/"
Original_Masks = "./Data/01_Masks/"
WindowCT = "./Data/02_WindowCT_VAE1_results/"
GrayscaleCT = "./Data/03_GrayscaleCT_VAE1_results/"
MaskCT = "./Data/04_MaskCT_VAE1_results/"
CropTumor = "./Data/05_CropTumor_VAE1_results/"
NewMaskWindow = "./Data/06_NewMaskWindow_VAE2_results/"
MaskedCT = "./Data/07_MaskedCT_VAE2_results/"
CroppedWindow = "./Data/08_CroppedWindow_VAE2_results/"

########################################################################################################################
#                                                       VAE 1                                                          #
########################################################################################################################
# dbCT.windowing(dir_in=Original_Images, dir_out=WindowCT, wl=-600, ww=1500, slope=1, intercept=1024)
# print("WindowCT completed")
#
# dbCT.grayscale(WindowCT, GrayscaleCT)
# print("GrayscaleCT completed")
#
# dbCT.masking(GrayscaleCT, MaskCT, Original_Masks)
# print("MaskCT completed")
#
# dbCT.cropping_minimum(MaskCT, CropTumor)
# print("CropTumor completed")

########################################################################################################################
#                                                       VAE 2                                                          #
########################################################################################################################
# dbCT.create_windowed_masks(Original_Masks, NewMaskWindow, 104)
# print("NewMaskWindow completed")
#
# dbCT.masking(GrayscaleCT, MaskedCT, NewMaskWindow)
# print("MaskedCT completed")
#
# dbCT.cropping_minimum(MaskedCT, CroppedWindow)
# print("CroppedWindow completed")

########################################################################################################################
#                                                   Statistics                                                         #
########################################################################################################################

dbCT.slice_statistics(CropTumor)

