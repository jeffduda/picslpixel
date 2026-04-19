import SimpleITK as sitk
from .utilities import mask_image_multiply

# From https://github.com/InsightSoftwareConsortium/SimpleITK-Notebooks/blob/main/Python/05_Results_Visualization.ipynb

def alpha_blend(image1, image2, alpha=0.5, mask1=None, mask2=None):
        """
        Alaph blend two images, pixels can be scalars or vectors.
        The alpha blending factor can be either a scalar or an image whose
        pixel type is sitkFloat32 and values are in [0,1].
        The region that is alpha blended is controled by the given masks.
        """

        if not mask1:
            mask1 = sitk.Image(image1.GetSize(), sitk.sitkFloat32) + 1.0
            mask1.CopyInformation(image1)
        else:
            mask1 = sitk.Cast(mask1, sitk.sitkFloat32)


        if not mask2:
            mask2 = sitk.Image(image2.GetSize(), sitk.sitkFloat32) + 1
            mask2.CopyInformation(image2)
        else:
            mask2 = sitk.Cast(mask2, sitk.sitkFloat32)

        # if we received a scalar, convert it to an image
        if type(alpha) != sitk.SimpleITK.Image:
            alpha = sitk.Image(image1.GetSize(), sitk.sitkFloat32) + alpha
            alpha.CopyInformation(image1)
        components_per_pixel = image1.GetNumberOfComponentsPerPixel()
        if components_per_pixel > 1:
            img1 = sitk.Cast(image1, sitk.sitkVectorFloat32)
            img2 = sitk.Cast(image2, sitk.sitkVectorFloat32)
        else:
            img1 = sitk.Cast(image1, sitk.sitkFloat32)
            img2 = sitk.Cast(image2, sitk.sitkFloat32)

        intersection_mask = mask1 * mask2

        intersection_image = mask_image_multiply(alpha * intersection_mask, img1) + mask_image_multiply((1 - alpha) * intersection_mask, img2)
        
        out_image = intersection_image + mask_image_multiply(mask2 - intersection_mask, img2) + mask_image_multiply(mask1 - intersection_mask, img1)
        return out_image