import logging
import json
import SimpleITK as sitk
import argparse

from picslpixel.label_overlay import LabelOverlay


def volume_overlay(args):
    img = sitk.ReadImage(args.input_image)
    seg = sitk.ReadImage(args.input_segmentation)

    overlay = LabelOverlay(img, seg, width=args.border_width)
    overlay.window = args.window
    overlay.level = args.level
    overlay.load_label_colors(args.colors_file)
    overlay_img = overlay.create_overlay()

    sitk.WriteImage(overlay_img, args.output)



if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="Example script using argparse.")
    parser.add_argument("--input_image", '-i', type=str, default=None, required=True, help="Input NIfTI image")
    parser.add_argument("--input_segmentation", '-s', type=str, default=None, required=True, help="Input NIfTI of labels")
    parser.add_argument("--border_width", '-b', type=str, default=1, required=False, help="Border width for label overlay (default: 1)")
    parser.add_argument("--window", '-w', type=int, default=300, required=False, help="Window width for intensity windowing (default: 300)")
    parser.add_argument("--level", '-l', type=int, default=50, required=False, help="Window level for intensity windowing (default: 50)")
    parser.add_argument("--colors_file", '-c', type=str, default=None, required=True, help="Path to the JSON file with colors for each label")
    parser.add_argument("--output", '-o', type=str, default=None, required=True, help="Path to the output nifti file")
    args = parser.parse_args()

    volume_overlay(args)