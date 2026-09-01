<p align="center"><img src="docs/picslpixel_logo.png" alt="PMBB Brick" width="128" /></p>

# picslpixel

Various helper classes and functions for working with medical imaging data using SimpleITK


# Installation
You can install the package from source or eventually via PyPI

## Get the repo
git clone https://github.com/jeffduda/picslpixel.git 
pip install -e picslpixel

## Examples

### Create a color overlay from a scalar volume and a set of labels
Define the colors in a json file (see data/label_colors.json for an example)

python /path/to/picslpixel/picslpixel/examples/volume_overlay.py -i scalar_volume.nii.gz -s label_volume.nii.gz -c label_colors.json  -w 400 -l 50 -b 1 -o output_overlay.nii.gz
