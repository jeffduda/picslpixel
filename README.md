<p align="center"><img src="docs/picslpixel_logo.png" alt="PMBB Brick" width="128" /></p>

# picslpixel

Various helper classes and functions for working with medical imaging data using SimpleITK


# Installation
You can install the package from source or eventually via PyPI

## Get the repo
```bash
git clone https://github.com/jeffduda/picslpixel.git 
pip install -e picslpixel
```

## Examples

### Create a color overlay from a scalar volume and a set of labels
Define the colors in a json file (see data/label_colors.json for an example)

The overlay example has the following parameters

* -i input_image.nii.gz (required)
* -s input_labels.nii.gz (required)
* -c label_colors.json (required)
* -w window width (int) [default=400]
* -l window level (int) [default=50]
* -b border width (int) [default=1]
* -o output_volume.nii.gz (required)

```bash
python /path/to/picslpixel/picslpixel/examples/volume_overlay.py -i scalar_volume.nii.gz -s label_volume.nii.gz -c label_colors.json  -w 400 -l 50 -b 1 -o output_overlay.nii.gz
```

### Generate RGB Dicom files from an RGB Nifti and the original dicom files
The original dicom files need to be the only files in the input directory, and the output directory should already exist. A new series UID will be randomly generated.

* -i overlay.nii.gz 
* -d original_dicom_files/ 
* -o rgb_dicom_files/
* -t tolerance (allowed diff in slice positions between dicom and nifti)
* -s series number to use in new dicom
* -sd series desscription to use in new dicom
* -an accession number to use in new dicom

```bash
python /path/to/picslpixel/picslpixel/examples/dicom_rgb.py -i overlay.nii.gz -d original_dicom_dir/ -o output_dicom_dir -s 101 -sd "Overlay Series" -an HTX0001
```





