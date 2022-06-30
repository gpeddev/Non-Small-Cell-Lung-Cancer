from ProcessingCTs.DatabaseCT import DatabaseCT

Original_Images = "/home/gped/Workspace/Thesis/01_Data/003_Data/00_Images/"
Original_Masks = "/home/gped/Workspace/Thesis/01_Data/003_Data/01_Masks/"
Windowed_Images = "/home/gped/Workspace/Thesis/01_Data/003_Data/02_Windowing/"
Grayscale_Images = "/home/gped/Workspace/Thesis/01_Data/003_Data/03_Grayscale/"
Masked_Images_out = "/home/gped/Workspace/Thesis/01_Data/003_Data/04_Masked/"
Cropped_Images = "/home/gped/Workspace/Thesis/01_Data/003_Data/05_Tumor_Cropped/"
Create_Window_Masks = "/home/gped/Workspace/Thesis/01_Data/003_Data/06_Window_Masks/"
Window_Masked_Images = "/home/gped/Workspace/Thesis/01_Data/003_Data/07_Window_Masked_Images/"
Window_Cropped_Images = "/home/gped/Workspace/Thesis/01_Data/003_Data/08_Window_Cropped_Images/"

img_db = DatabaseCT()
########################################################################################################################
#                                                       VAE 1                                                          #
########################################################################################################################
img_db.windowing(dir_in=Original_Images, dir_out=Windowed_Images, wl=-600, ww=1500, slope=1, intercept=1024)
print("Windowing completed")

img_db.grayscale(Windowed_Images, Grayscale_Images)
print("Grayscale completed")

img_db.masking(Grayscale_Images, Masked_Images_out, Original_Masks)
print("Masking completed")

img_db.cropping_minimum(Masked_Images_out, Cropped_Images)
print("Cropping tumor completed")

########################################################################################################################
#                                                       VAE 2                                                          #
########################################################################################################################
img_db.create_windowed_masks(Original_Masks, Create_Window_Masks, 104)
print("Create windowed masks completed")

img_db.masking(Grayscale_Images, Window_Masked_Images, Create_Window_Masks)
print("Masking completed")

img_db.cropping_minimum(Window_Masked_Images, Window_Cropped_Images)
print("Cropping Window tumor completed")
