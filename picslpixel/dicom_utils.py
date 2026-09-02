import os
import numpy as np
import pydicom

def load_and_sort_dicom_directory(directory_path):
    """
    Loads all DICOM files from a directory and sorts them spatially along
    the slice-normal vector to form a proper 3D volume.
    """
    slices = []
    
    # 1. Read all valid DICOM files in the directory
    for filename in os.listdir(directory_path):
        filepath = os.path.join(directory_path, filename)
        if os.path.isfile(filepath):
            try:
                ds = pydicom.dcmread(filepath)
                # Ensure the file contains spatial metadata
                if "ImagePositionPatient" in ds and "ImageOrientationPatient" in ds:
                    slices.append(ds)
            except pydicom.errors.InvalidDicomError:
                continue  # Skip non-DICOM files

    if not slices:
        raise ValueError("No valid DICOM files with spatial metadata found.")

    # 2. Extract direction cosines from the first slice
    # [rx, ry, rz] = row direction vector, [cx, cy, cz] = column direction vector
    orientation = slices[0].ImageOrientationPatient
    row_cosines = np.array(orientation[:3])
    col_cosines = np.array(orientation[3:])
    
    # 3. Compute the slice-normal vector (cross product of row and column)
    slice_normal = np.cross(row_cosines, col_cosines)

        # 4. Project Image Position Patient onto the slice-normal vector and sort
    # We zip the projection distance with the slice object to sort them together
    sorted_dcm = sorted(
        slices,
        key=lambda s: np.dot(np.array(s.ImagePositionPatient), slice_normal)
    )

    # 5. Stack the pixel arrays into a single 3D numpy volume
    # Resulting shape format: (Z_slices, Y_rows, X_columns)
    #volume_3d = np.stack([s.pixel_array for s in sorted_slices])

    return sorted_dcm

# Example Usage:
# dicom_dir = "path/to/your/dicom/series"
# sorted_datasets = load_and_sort_dicom_directory(dicom_dir)
# print(f"Number of slices: {len(sorted_datasets)}")