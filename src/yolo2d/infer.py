from ultralytics import YOLO
import cv2
import numpy as np
import torch


model_path = "yolov11-100epochs.pt"
model = YOLO(model_path) 

# Perform prediction
def infer(input_image, post_process=True):
    results = model.predict(source=input_image, save=False, conf=0.05)
    # Check if masks are available
    for result in results:
        if result.masks is not None:
            # Extract mask data from the Masks object
            masks = result.masks.data.cpu().numpy()  # Convert to NumPy array
            # Initialize an empty mask to combine all instances
            combined_mask = np.zeros_like(input_image)

            # Iterate over each mask
            for i, mask in enumerate(masks):
                if mask.ndim != 2:
                    raise ValueError(f"Expected 2D mask, got shape: {mask.shape}")
                # Convert mask to binary and ensure it's uint8
                mask = (mask * 255).astype(np.uint8)
                # Resize the mask to match the input image size if necessary
                mask_resized = cv2.resize(mask, (input_image.shape[1], input_image.shape[0]))
                # Convert the mask to 3-channel
                mask_3ch = np.repeat(mask_resized[:, :, np.newaxis], 3, axis=2)
                # Apply the mask to the combined mask
                combined_mask = cv2.addWeighted(combined_mask, 1, mask_3ch, 0.5, 0)
            pred_gray_image = cv2.cvtColor(combined_mask, cv2.COLOR_BGR2GRAY)
            _, pred_binary_mask = cv2.threshold(pred_gray_image, 1, 255, cv2.THRESH_BINARY_INV)
            if post_process:
                # Invert the binary mask to make black areas white
                inverted_mask = cv2.bitwise_not(pred_binary_mask // 255)
                # Apply dilation to expand the white regions
                kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (25, 25))  # Adjust kernel size for connectivity
                dilated_mask = cv2.dilate(inverted_mask, kernel, iterations=1)
                # Apply morphological closing to fill gaps
                closed_mask = cv2.morphologyEx(dilated_mask, cv2.MORPH_CLOSE, kernel)
                # Invert the mask back to its original format
                pred_mask = cv2.bitwise_not(closed_mask)
            else:
                pred_mask = pred_binary_mask // 255
            # Invert the fault mask
            inverted_fault_mask = cv2.bitwise_not(pred_mask * 255)
            # Convert binary image to a 3-channel image for overlay (BGR format)
            fault_mask_bgr = cv2.cvtColor(inverted_fault_mask, cv2.COLOR_GRAY2BGR)

            # # --- black annotation ---
            # # Create a black mask where the fault exists
            # black_overlay = np.zeros_like(input_image)
            # # Combine input_image and black_overlay using the mask
            # overlay = np.where(fault_mask_bgr == 255, black_overlay, input_image)

            # --- white annotation ---
            # Overlay the fault mask onto the raw seismic image
            overlay = cv2.addWeighted(input_image, 0.8, fault_mask_bgr, 0.5, 0)

            # cv2.imwrite('input_overlay.jpg', overlay)
            return overlay
        else:
            print("No masks found in results.")
