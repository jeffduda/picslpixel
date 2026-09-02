import logging
import json
import SimpleITK as sitk
import argparse
import pydicom
import numpy as np
import os

from picslpixel.label_overlay import LabelOverlay
from picslpixel.dicom_utils import load_and_sort_dicom_directory


def dicom_rgb(args):
    img = sitk.ReadImage(args.input_image)
    dcms = load_and_sort_dicom_directory(args.input_dicom_directory)

    img = sitk.DICOMOrient(img, "LPS")

    rgbArr = sitk.GetArrayFromImage(img)
    #rgbArr = np.flip(rgbArr, 1)

    seriesUID=pydicom.uid.generate_uid()
    seriesNumber=args.series_number
    seriesDescription=args.series_description
    accessionNumber=args.accession_number
    
    for i, dcm in enumerate(dcms):

        instanceUID = pydicom.uid.generate_uid()

        idx = [0,0,i]
        #print("Physical position from index:", img.TransformIndexToPhysicalPoint(idx))
        #print(dcm.ImagePositionPatient)

        diff = abs(img.TransformIndexToPhysicalPoint(idx)[2] - dcm.ImagePositionPatient[2])
        #print("Difference in Z position:", diff)
        if diff > args.tolerance:
            print("Warning: Difference in Z position exceeds tolerance.")

        # Update the pixel data of the DICOM file with the new image data
        dcm.PixelData = sitk.GetArrayFromImage(img).tobytes()
        dcm.SOPInstanceUID = instanceUID
        dcm.SeriesNumber = seriesNumber
        dcm.SeriesDescription = seriesDescription
        if accessionNumber is not None:
            dcm.AccessionNumber = accessionNumber

        itype = dcm.ImageType
        itype[0]="DERIVED"
        itype[1]="SECONDARY"
        dcm.ImageType = itype

        dcm.RescaleIntercept = 0
        dcm.RescaleSlope = 1
        dcm.PixelRepresentation = 0
        dcm.HighBit = 7
        dcm.BitsAllocated = 8
        dcm.SamplesPerPixel = 3
        dcm.BitsStored = 8
        dcm.PhotometricInterpretation = "RGB"
        dcm.AcquisitionNumber = 1

        if dcm.file_meta.TransferSyntaxUID == pydicom.uid.JPEG2000Lossless:
            dcm.file_meta.TransferSyntaxUID = pydicom.uid.ExplicitVRLittleEndian

        colorDat = rgbArr[i,:,:,:].astype(np.uint8)
        print("Color data shape:", colorDat.shape)
        dcm.PixelData = colorDat.tobytes()

        # Save the modified DICOM file to the output directory
        output_path = os.path.join(args.output_directory, instanceUID + ".dcm")
        dcm.save_as(output_path)




if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="Create dicom rgbs from a color overlay nifti file.")
    parser.add_argument("--input_image", '-i', type=str, default=None, required=True, help="Input NIfTI image")
    parser.add_argument("--input_dicom_directory", '-d', type=str, default=None, required=True, help="Directory of dicom files")
    parser.add_argument("--output_directory", '-o', type=str, default=None, required=True, help="Path to the output nifti file")
    parser.add_argument("--tolerance", '-t', type=float, default=1e-3, required=False, help="Tolerance for position matching (default: 1e-3)")
    parser.add_argument("--series_number", '-s', type=int, default=104, required=False, help="Series number for the output DICOM files (default: 104)")
    parser.add_argument("--series_description", '-sd', type=str, default="Overlay RGB", required=False, help="Series description for the output DICOM files (default: 'Overlay RGB')")
    parser.add_argument("--accession_number", '-an', type=str, required=False, help="Accession number for the output DICOM files (default: 'HTX12345')")
    args = parser.parse_args()

    dicom_rgb(args)