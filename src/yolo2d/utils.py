import cv2
import numpy as np
import os

def create_yolo_segmentation_labels_from_directory(mask_dir, output_dir, class_id=0):
    """
    Creates YOLO segmentation labels for all mask images in a directory.
    
    Args:
        mask_dir (str): Path to the directory containing mask images.
        output_dir (str): Path to the directory where label files will be saved.
        class_id (int): Class ID to assign to all objects.
    """
    # Ensure the output directory exists
    os.makedirs(output_dir, exist_ok=True)

    # Process each mask image in the directory
    for mask_filename in os.listdir(mask_dir):
        # Construct full file path
        mask_path = os.path.join(mask_dir, mask_filename)
        
        # Check if the file is an image
        if not mask_path.lower().endswith(('.png', '.jpg', '.jpeg')):
            print(f"Skipping non-image file: {mask_filename}")
            continue
        
        # Load the mask image (assume black regions represent the object)
        mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
        if mask is None:
            print(f"Could not read mask: {mask_filename}")
            continue
        
        height, width = mask.shape

        # Invert the mask (black -> white, white -> black)
        inverted_mask = cv2.bitwise_not(mask)

        # Find contours in the inverted mask
        contours, _ = cv2.findContours(inverted_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        # Prepare the label data
        label_data = []
        for contour in contours:
            # Normalize polygon points
            polygon_points = []
            for point in contour:
                px, py = point[0]
                polygon_points.append(f"{px / width} {py / height}")  # Use whitespace instead of comma

            # Combine into one line
            label_line = f"{class_id} " + " ".join(polygon_points)
            label_data.append(label_line)

        # Write to output file
        label_filename = os.path.splitext(mask_filename)[0] + ".txt"
        label_filepath = os.path.join(output_dir, label_filename)
        with open(label_filepath, "w") as f:
            f.write("\n".join(label_data))

        print(f"Label created for: {mask_filename} -> {label_filepath}")