import SimpleITK as sitk

# From https://github.com/InsightSoftwareConsortium/SimpleITK-Notebooks/blob/main/Python/05_Results_Visualization.ipynb
def  mask_image_multiply(mask, image):
    components_per_pixel = image.GetNumberOfComponentsPerPixel()
    if components_per_pixel == 1:
        return mask * image
    else:
        return sitk.Compose(
            [
                mask * sitk.VectorIndexSelectionCast(image, channel)
                for channel in range(components_per_pixel)
            ]
        )