import SimpleITK as sitk
import numpy as np
import json

from .alpha_blend import alpha_blend
from .utilities import mask_image_multiply  

class LabelOverlay:
    def __init__(self, image, label_image, label_colors=None, borders_colors=None, width=1):
        self.image = image
        self.label_image = label_image
        self.label_colors = label_colors if label_colors is not None else {}
        self.border_colors = borders_colors if borders_colors is not None else {}
        self._window=None
        self._level=None
        self._width=width

    @property
    def window(self):
        return self._window
    @window.setter
    def window(self, value):
        self._window = value

    @property
    def width(self):
        return self._width
    @width.setter
    def width(self, value):
        self._width = value



    @property
    def level(self):
        return self._level
    @level.setter
    def level(self, value):
        self._level = value

    #def _mask_image_multiply(self, mask, image):
    #    components_per_pixel = image.GetNumberOfComponentsPerPixel()
    #    if components_per_pixel == 1:
    #        return mask * image
    #    else:
    #        return sitk.Compose(
    #            [
    #                mask * sitk.VectorIndexSelectionCast(image, channel)
    #                for channel in range(components_per_pixel)
    #            ]
    #        )

    def _alpha_blend(self, image1, image2, alpha=0.5, mask1=None, mask2=None):
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

    

    def apply_window_level(self):
        if self.window is not None and self.level is not None:
            lower_bound = self.level - self.window / 2
            upper_bound = self.level + self.window / 2
            wl_img = sitk.IntensityWindowing(self.image, windowMinimum=lower_bound, windowMaximum=upper_bound,
                                             outputMinimum=0, outputMaximum=255)
            #wl_img = sitk.Cast(wl_img, sitk.sitkUInt8)
            
            return wl_img
        else:
            return self.image

    def load_label_colors(self, label_colors_file):
        with open(label_colors_file, 'r') as f:
            dat = json.load(f)
            if 'labels' in dat:
                for label in dat['labels']:
                    self.label_colors[label['index']] = label.get('color')
                    self.border_colors[label['index']] = label.get('border')


    def create_overlay(self):
        wl_img = self.apply_window_level()

        # Get labels that are in both images and have defined colors
        labels = np.unique(sitk.GetArrayViewFromImage(self.label_image))
        labels = [l for l in labels if l in self.label_colors]

        # Create an RGB image for the overlay
        overlay = sitk.Compose([wl_img]*3)  

        for label in labels:
            color = self.label_colors[label][:3]  # Get RGB color for the label
            border_color = self.border_colors.get(label, color)

            # Create a binary mask for the current label
            label_mask = self.label_image == label
            #label_mask = sitk.Cast(label_mask, sitk.sitkFloat32)

            # Create a border mask using morphological operations
            #dilated_mask = sitk.BinaryDilate(label_mask, [1]*self.label_image.GetDimension())
            #border_mask = sitk.Subtract(dilated_mask, label_mask)

            # Color the label and border in the overlay
            r_img = sitk.Image(label_mask.GetSize(), sitk.sitkFloat32) + color[0]
            g_img = sitk.Image(label_mask.GetSize(), sitk.sitkFloat32) + color[1]
            b_img = sitk.Image(label_mask.GetSize(), sitk.sitkFloat32) + color[2]
            r_img.CopyInformation(label_mask)
            g_img.CopyInformation(label_mask)
            b_img.CopyInformation(label_mask)

            roi_border = None
            if border_color is not None and border_color != color:
                r_border = sitk.Image(label_mask.GetSize(), sitk.sitkFloat32) + border_color[0]
                g_border = sitk.Image(label_mask.GetSize(), sitk.sitkFloat32) + border_color[1]
                b_border = sitk.Image(label_mask.GetSize(), sitk.sitkFloat32) + border_color[2]
                r_border.CopyInformation(label_mask)
                g_border.CopyInformation(label_mask)
                b_border.CopyInformation(label_mask)
                roi_border = sitk.Compose([r_border, g_border, b_border])


                #border_mask = sitk.LabelContour(sitk.Cast(label_mask, sitk.sitkUInt8), fullyConnected=True, backgroundValue=0)
                #label_mask = label_mask - border_mask

                roi_core = sitk.BinaryErode(label_mask, kernelRadius=[self.width]*3, kernelType=sitk.sitkBall, foregroundValue=1)
                roi_core = sitk.Cast(roi_core, sitk.sitkFloat32)
                label_mask = sitk.Cast(label_mask, sitk.sitkFloat32)
                border_mask = label_mask - roi_core
                label_mask = roi_core       

                border_mask =sitk.Cast(border_mask, sitk.sitkFloat32)
                label_mask = sitk.Cast(label_mask, sitk.sitkFloat32)

            roi_color = sitk.Compose([r_img, g_img, b_img])
            roi_color.CopyInformation(label_mask)
            #roi_color = self._mask_image_multiply(label_mask, roi_color)

            overlay = self._alpha_blend(roi_color, overlay, mask1=label_mask, alpha=self.label_colors[label][3]/255.0)
            if roi_border is not None:
                overlay = self._alpha_blend(roi_border, overlay, mask1=border_mask, alpha=self.border_colors[label][3]/255.0)



        return(overlay)
    
