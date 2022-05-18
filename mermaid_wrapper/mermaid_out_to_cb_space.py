import SimpleITK as sitk
import numpy as np
import os,subprocess,sys,time,pydicom
from glob import glob
from rt_utils import RTStructBuilder

subj = sys.argv[1]

#if len(sys.argv) >= 2:
#    data_home = sys.argv[2]
#else:
data_home = '/software/data'

model_out = os.path.join(data_home,'test_result_best_eval.pth.tar')

with open('/hash_id.txt') as f:
    hash_id = f.read()[:-1]

processed_dir = glob(os.path.join(data_home,subj,'Processed','CB*'))[0]
cb = processed_dir.split('/')[-1]
original_dir = os.path.join(data_home,subj,'Original',cb)
network_out = glob(os.path.join(model_out,subj + '*' + cb + '*'))[0]
cb_dicom_dir = glob(os.path.join(data_home,'dicoms',subj,cb,'*CT_*'))[0]

#load cropped original CB image for reference space
cropped_itk = sitk.ReadImage(os.path.join(processed_dir,'cropped_original_image.nii.gz'))
qOffset = cropped_itk.GetOrigin()

#load outputted images
ct_warped_itk = sitk.ReadImage(os.path.join(network_out,'warped_moving_image.nii.gz'))
label_itk = sitk.ReadImage(os.path.join(network_out,'warped_moving_labels.nii.gz'))

#correct the origin in the isotropic outputted images
ct_warped_itk.SetOrigin(qOffset)
label_itk.SetOrigin(qOffset)

#initialize resampling filter
resamp_filter = sitk.ResampleImageFilter()
resamp_filter.SetReferenceImage(cropped_itk)

#resample iso cropped CT
resamp_filter.SetInterpolator(sitk.sitkLinear)
rct = resamp_filter.Execute(ct_warped_itk)

#resample iso outputted cropped labels
#resamp_filter.SetInterpolator(sitk.sitkNearestNeighbor)
resamp_filter.SetInterpolator(sitk.sitkLabelGaussian)
rlabels = resamp_filter.Execute(label_itk)

#save resampled images
sitk.WriteImage(rct,os.path.join(network_out,'cbspace_warped_CT_image.nii.gz'))
sitk.WriteImage(rlabels,os.path.join(network_out,'cbspace_warped_moving_labels.nii.gz'))

#To put cropped resampled images to full FOV array, load bounding box saved in preprocessing crop operation
bbox = np.load(os.path.join(processed_dir,'bbox_xu.npy'))

#Load full FOV original CB image
orig_itk = sitk.ReadImage(os.path.join(original_dir,'original_image.nii.gz'))
qOffsetOrig = orig_itk.GetOrigin()
voxSpacingOrig = orig_itk.GetSpacing()

#Create blank full FOV image, insert labels mask and save
fov_label_mask = sitk.Image(orig_itk.GetSize(),sitk.sitkUInt32)
fov_label_mask.SetOrigin(qOffsetOrig)
fov_label_mask.SetSpacing(voxSpacingOrig)
fov_label_mask[bbox[0]:bbox[0] + bbox[3], bbox[1]:bbox[1] + bbox[4], bbox[2]:bbox[2] + bbox[5]] = sitk.Cast(rlabels,sitk.sitkUInt32)
fullFOVLabelFile = os.path.join(network_out,'fullFOV_cbspace_warped_moving_labels.nii.gz')
sitk.WriteImage(fov_label_mask,fullFOVLabelFile)

#write cropped CB image to full FOV image
orig_img_arr = sitk.GetArrayFromImage(orig_itk)
bg_arr = np.full(orig_img_arr.shape,-1000)
bg_img = sitk.Cast(sitk.GetImageFromArray(bg_arr),sitk.sitkFloat32)
bg_img[bbox[0]:bbox[0] + bbox[3], bbox[1]:bbox[1] + bbox[4], bbox[2]:bbox[2] + bbox[5]] = orig_itk[bbox[0]:bbox[0] + bbox[3], bbox[1]:bbox[1] + bbox[4], bbox[2]:bbox[2] + bbox[5]]
bg_img.SetOrigin(qOffsetOrig)
bg_img.SetSpacing(voxSpacingOrig)
fullFOVCroppedCBFile = os.path.join(network_out,'fullFOV_cropped_cb_image.nii.gz')
sitk.WriteImage(bg_img,fullFOVCroppedCBFile)

#write cropped warped CT image to full FOV image
wct_arr = np.full(orig_img_arr.shape,-1)
wct_img = sitk.Cast(sitk.GetImageFromArray(wct_arr),sitk.sitkFloat32)
wct_img[bbox[0]:bbox[0] + bbox[3], bbox[1]:bbox[1] + bbox[4], bbox[2]:bbox[2] + bbox[5]] = rct
wct_img.SetOrigin(qOffsetOrig)
wct_img.SetSpacing(voxSpacingOrig)
fullFOVCroppedWarpedCTFile = os.path.join(network_out,'fullFOV_cropped_warped_ct_image.nii.gz')
sitk.WriteImage(wct_img,fullFOVCroppedWarpedCTFile)

#Create output DICOM directories
#dcm_out = os.path.join(network_out,'DICOM')
dcm_out = os.path.join(data_home,'output')
os.makedirs(dcm_out,exist_ok=True)

#convert RT struct
label_img = sitk.ReadImage(fullFOVLabelFile)
sb_arr = sitk.GetArrayFromImage(label_img) == 1
sd_arr = sitk.GetArrayFromImage(label_img) == 2

rtstructout = os.path.join(dcm_out,'RTSTRUCT.dcm')
rtstruct = RTStructBuilder.create_new(dicom_series_path = cb_dicom_dir)
rtstruct.add_roi(mask = np.transpose(sb_arr,(1,2,0)), color = [255,0,0], name="Bowel_sm")
rtstruct.add_roi(mask = np.transpose(sd_arr,(1,2,0)), color = [255,165,38], name="Stomach_duo")
rtstruct.save(rtstructout)

#load first cbct dicom to check FORUID in RTSTRUCT
cbhdr = pydicom.dcmread(glob(os.path.join(cb_dicom_dir,'*.dcm'))[0])
forUID = cbhdr['FrameOfReferenceUID'].value
rt = pydicom.dcmread(os.path.join(dcm_out,'RTSTRUCT.dcm'))
mermaid_id_str = 'MERMAID_FASTCBNET_v1 ' + hash_id
rt.add_new([0x3006,0x4],'LO',mermaid_id_str)
rt[0x3006,0x20][0][0x3006,0x28].value = mermaid_id_str
rt[0x3006,0x20][1][0x3006,0x28].value = mermaid_id_str
rt[0x8,0x103e].value = mermaid_id_str
if rt[0x3006,0x10][0][0x20,0x52].value != forUID:
	rt[0x3006,0x10][0][0x20,0x52].value = forUID
	rt[0x3006,0x20][0][0x3006,0x24].value = forUID
	rt[0x3006,0x20][1][0x3006,0x24].value = forUID
rt.save_as(rtstructout)


#Convert cropped full FOV CB nii to DICOM
series_IDs = sitk.ImageSeriesReader.GetGDCMSeriesIDs(cb_dicom_dir)
series_file_names = sitk.ImageSeriesReader.GetGDCMSeriesFileNames(cb_dicom_dir, series_IDs[0])

series_reader = sitk.ImageSeriesReader()
series_reader.SetFileNames(series_file_names)
series_reader.MetaDataDictionaryArrayUpdateOn()
series_reader.LoadPrivateTagsOn()
cbct_img = series_reader.Execute()

#load the full fov image
#fullFOVCroppedCBFile = os.path.join(network_out,'fullFOV_cropped_cb_image.nii.gz')
#fov_img = sitk.ReadImage(fullFOVCroppedCBFile)
fov_img = sitk.Cast(bg_img,sitk.sitkInt32)
warp_img = sitk.Cast(wct_img*1000,sitk.sitkInt32)

# Write the 3D image as a series
# IMPORTANT: There are many DICOM tags that need to be updated when you modify an
#            original image. This is a delicate opration and requires knowlege of
#            the DICOM standard. This example only modifies some. For a more complete
#            list of tags that need to be modified see:
#                           http://gdcm.sourceforge.net/wiki/index.php/Writing_DICOM

writer = sitk.ImageFileWriter()
# Use the study/series/frame of reference information given in the meta-data
# dictionary and not the automatically generated information from the file IO
writer.KeepOriginalImageUIDOn()

# Copy relevant tags from the original meta-data dictionary (private tags are also
# accessible).
tags_to_copy = ["0010|0010", # Patient Name
                "0010|0020", # Patient ID
                "0010|0030", # Patient Birth Date
                "0020|000D", # Study Instance UID, for machine consumption
                "0020|0010", # Study ID, for human consumption
                "0008|0020", # Study Date
                "0008|0030", # Study Time
                "0008|0050", # Accession Number
                "0008|0060"  # Modality
]

#output cropped CBCT
modification_time = time.strftime("%H%M%S")
modification_date = time.strftime("%Y%m%d")

# Copy some of the tags and add the relevant tags indicating the change.
# For the series instance UID (0020|000e), each of the components is a number, cannot start
# with zero, and separated by a '.' We create a unique series ID using the date and time.
# tags of interest:
direction = fov_img.GetDirection()
series_tag_values = [(k, series_reader.GetMetaData(0,k)) for k in tags_to_copy if series_reader.HasMetaDataKey(0,k)] + \
                 [("0008|0031",modification_time), # Series Time
                  ("0008|0021",modification_date), # Series Date
                  ("0008|0008","DERIVED\\SECONDARY"), # Image Type
                  ("0020|000e", "1.2.826.0.1.3680043.2.1125."+modification_date+".1"+modification_time), # Series Instance UID
                  ("0020|0037", '\\'.join(map(str, (direction[0], direction[3], direction[6],# Image Orientation (Patient)
                                                    direction[1],direction[4],direction[7])))),
                  ("0008|103e", "Cropped CBCT Processed-Mermaid_FastCBNet")] # Series Description

for i in range(fov_img.GetDepth()):
    image_slice = fov_img[:,:,i]
    # Tags shared by the series.
    for tag, value in series_tag_values:
        image_slice.SetMetaData(tag, value)
    # Slice specific tags.
    image_slice.SetMetaData("0008|0012", time.strftime("%Y%m%d")) # Instance Creation Date
    image_slice.SetMetaData("0008|0013", time.strftime("%H%M%S")) # Instance Creation Time
    image_slice.SetMetaData("0020|0032", '\\'.join(map(str,fov_img.TransformIndexToPhysicalPoint((0,0,i))))) # Image Position (Patient)
    image_slice.SetMetaData("0020,0013", str(i)) # Instance Number
    # Write to the output directory and add the extension dcm, to force writing in DICOM format.
    writer.SetFileName(os.path.join(dcm_out,'CT_1.2.826.0.1.3680043.2.1125.' + modification_date + '.1' + modification_time + '_' + str(i) + '.dcm'))
    writer.Execute(image_slice)



#output warped planCT
modification_time = time.strftime("%H%M%S")
modification_date = time.strftime("%Y%m%d")

direction = warp_img.GetDirection()
series_tag_values = [(k, series_reader.GetMetaData(0,k)) for k in tags_to_copy if series_reader.HasMetaDataKey(0,k)] + \
                 [("0008|0031",modification_time), # Series Time
                  ("0008|0021",modification_date), # Series Date
                  ("0008|0008","DERIVED\\SECONDARY"), # Image Type
                  ("0020|000e", "1.2.826.0.1.3680043.2.1125."+modification_date+".1"+modification_time), # Series Instance UID
                  ("0020|0037", '\\'.join(map(str, (direction[0], direction[3], direction[6],# Image Orientation (Patient)
                                                    direction[1],direction[4],direction[7])))),
                  ("0008|103e", "Warped planCT Processed-Mermaid_FastCBNet")] # Series Description

for i in range(warp_img.GetDepth()):
    image_slice = warp_img[:,:,i]
    # Tags shared by the series.
    for tag, value in series_tag_values:
        image_slice.SetMetaData(tag, value)
    # Slice specific tags.
    image_slice.SetMetaData("0008|0012", time.strftime("%Y%m%d")) # Instance Creation Date
    image_slice.SetMetaData("0008|0013", time.strftime("%H%M%S")) # Instance Creation Time
    image_slice.SetMetaData("0020|0032", '\\'.join(map(str,warp_img.TransformIndexToPhysicalPoint((0,0,i))))) # Image Position (Patient)
    image_slice.SetMetaData("0020,0013", str(i)) # Instance Number
    # Write to the output directory and add the extension dcm, to force writing in DICOM format.
    writer.SetFileName(os.path.join(dcm_out,'CT_1.2.826.0.1.3680043.2.1125.' + modification_date + '.1' + modification_time + '_' + str(i) + '.dcm'))
    writer.Execute(image_slice)

