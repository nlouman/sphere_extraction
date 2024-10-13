""" Description: This script extracts sphere masks from images using GroundingDINO for bounding box extraction and SAM2 for mask extraction. 
    It creates a preview of the extracted bounding boxes as well as the extracted masks overlaid on the original image.
    It saves the parameters of the extracted spheres as a pickle file.
"""
# Basic imports
from time import time
time_start = time()
import argparse
import os
import tqdm
import numpy as np
from scipy.optimize import minimize
import pickle as pkl

# Disable warnings
import warnings
warnings.filterwarnings("ignore")

# GroundingDINO imports
from groundingdino.util.inference import load_model, predict, annotate
import cv2
# SAM2 imports
from sam2.build_sam import build_sam2
from sam2.sam2_image_predictor import SAM2ImagePredictor

# Helper imports
from helper import *

# Parse input arguments
parser = argparse.ArgumentParser()
parser.add_argument('--bbox-model-config-path', type=str, required=False, default='../GroundingDINO/groundingdino/config/GroundingDINO_SwinT_OGC.py', help='Path to GroundingDINO model config file')
parser.add_argument('--bbox-model-weights-path', type=str, required=False, default='../GroundingDINO/weights/groundingdino_swint_ogc.pth', help='Path to GroundingDINO model weights file')
parser.add_argument('--mask-model-config-path', type=str, required=False, default='sam2_hiera_l.yaml', help='Path to SAM2 model config file')
parser.add_argument('--mask-model-weights-path', type=str, required=False, default='../segment-anything-2/checkpoints/sam2_hiera_large.pt', help='Path to SAM2 model weights file')
parser.add_argument('--image-folder', type=str, required=True, help='Path to folder with image file(s).')
parser.add_argument('--output-folder', type=str, required=True, help='Folder to save annotated image(s).')
parser.add_argument('--box-threshold', type=float, required=False, default=0.4, help='Bounding box threshold')
parser.add_argument('--text-threshold', type=float, required=False, default=0.25, help='Text threshold')
parser.add_argument('--compression-scale', type=float, required=False, default=0.5, help='Scale to compress image')
parser.add_argument('--num-spheres', type=int, required=True, help='Number of spheres to extract')
parser.add_argument('--padding', type=int, required=False, default=0, help='Padding to add to bounding box during mask extraction')
parser.add_argument('--order-spheres', type=bool, required=False, default=True, help='Order spheres by angle')

args = parser.parse_args()

# Args for image loading
IMAGE_FOLDER = args.image_folder
COMPRESSION_SCALE = args.compression_scale

# Args for output
OUTPUT_FOLDER = args.output_folder
MASK_OUTPUT_FOLDER = OUTPUT_FOLDER + '/masks'
BBOX_OUTPUT_FOLDER = OUTPUT_FOLDER + '/bounding_boxes'
ORDER_SPHERES = args.order_spheres

# Args for bounding box extraction
BBOX_CONFIG_PATH = args.bbox_model_config_path
BBOX_WEIGHTS_PATH = args.bbox_model_weights_path
BOX_TRESHOLD = args.box_threshold
TEXT_TRESHOLD = args.text_threshold
TEXT_PROMPT = "spheres"
INITIAL_CONFIDENCE_ADJUSTMENT = 0.05
MAX_CONFIDENCE_ADJUSTMENT = 0.2

# Args for mask extraction
MASK_CONFIG_PATH = args.mask_model_config_path
MASK_WEIGHTS_PATH = args.mask_model_weights_path
PADDING = args.padding
NUM_SPHERES = args.num_spheres

# Create folder structure
Helper.create_folder_structure(OUTPUT_FOLDER, MASK_OUTPUT_FOLDER, BBOX_OUTPUT_FOLDER)

###################################
# IMAGE PREPROCESSING AND LOADING #
###################################
print("Preprocessing images...")

if os.path.isdir(IMAGE_FOLDER):
    # Get all (uncompressed) images in the folder
    images = [f for f in os.listdir(IMAGE_FOLDER) if f.endswith('.jpg') and not f.endswith('_compressed.jpg')]
    compressed_images = []
    for image in images:
        image_path = os.path.join(IMAGE_FOLDER, image)
        # Compress image if there is no compressed version in the same directory
        # Will not overwrite existing compressed images when you change the compression factor
        compressed_image = BboxHelper.get_compressed(image_path, compression_factor=COMPRESSION_SCALE)
        compressed_image_source, compressed_image_transformed = compressed_image
        compressed_images.append((image_path, image.strip('.jpg'), compressed_image_source, compressed_image_transformed))
else:
    raise FileNotFoundError(f"Image folder {IMAGE_FOLDER} is not a valid directory.")

# Now we have "compressed_images", a list of tuples of the form (image_name: str, image_source: np.array, image: torch.Tensor)

##################################
# SPHERE BOUNDING BOX EXTRACTION #
##################################
print("Extracting sphere bounding boxes...")

# Load GroundingDINO model
model = load_model(model_config_path=BBOX_CONFIG_PATH, model_checkpoint_path=BBOX_WEIGHTS_PATH)

# Predict bounding boxes
boxes_cxxywh = []
for path, name, image_source, image in tqdm.tqdm(compressed_images):
    # Do initial prediction
    boxes, logits, phrases = predict(
            model=model,
            image=image,
            caption=TEXT_PROMPT,
            box_threshold=BOX_TRESHOLD,
            text_threshold=TEXT_TRESHOLD
        )
    
    confidence_adjustment = INITIAL_CONFIDENCE_ADJUSTMENT

    # Adjust confidence threshold until the number of spheres is at least equal to the desired number, maxing out at a certain threshold
    while len(boxes) < NUM_SPHERES:
        boxes, logits, phrases = predict(
            model=model,
            image=image,
            caption=TEXT_PROMPT,
            box_threshold=BOX_TRESHOLD-confidence_adjustment,
            text_threshold=TEXT_TRESHOLD
        )
        confidence_adjustment += INITIAL_CONFIDENCE_ADJUSTMENT
        if confidence_adjustment > MAX_CONFIDENCE_ADJUSTMENT:
            raise ValueError(f"Could not find enough spheres in image {name}.")
    
    # Save bounding boxes
    boxes_cxxywh.append((path, name, boxes))
        
    annotated_frame = annotate(image_source=image_source, boxes=boxes, logits=logits, phrases=phrases)
    cv2.imwrite(BBOX_OUTPUT_FOLDER + f'/{name}_annotated.jpg', annotated_frame)

# Convert bounding boxes to xyxy format and add the original image to the list
boxes_xyxy = []
for path, name, boxes in boxes_cxxywh:
    # Load (uncompressed) image
    image = cv2.imread(path)
    # Convert bounding boxes to xyxy format
    box_xyxy = BboxHelper.box_conversion_to_xyxy(image, boxes)
    # Save bounding boxes
    boxes_xyxy.append((path, name, image, box_xyxy))

##################################
#    SPHERE PARAMS EXTRACTION    #
##################################
print("Extracting sphere masks...")

# Load SAM2 model
predictor = SAM2ImagePredictor(build_sam2(MASK_CONFIG_PATH, MASK_WEIGHTS_PATH))

# Loop over all images
for path, name, image, bboxes in tqdm.tqdm(boxes_xyxy):
    # Add a black border around the original image to prevent indexing errors
    img_padded = cv2.copyMakeBorder(
        image, PADDING, PADDING, PADDING, PADDING, cv2.BORDER_CONSTANT, value=[0, 0, 0]
    )
    # Create an empty mask for the entire padded image
    full_mask_padded = np.zeros((img_padded.shape[0], img_padded.shape[1]), dtype=np.uint8)

    # Initialize list to store the cutouts, and one to store the masks and losses
    cutouts = []
    masks_with_losses = []

    # Loop over the bounding boxes
    for (x_min, y_min, x_max, y_max) in bboxes:
        ##################################
        #     SPHERE MASK EXTRACTION     #
        ##################################
        # Ensure bounding box coordinates are integers
        x_min, y_min, x_max, y_max = int(x_min), int(y_min), int(x_max), int(y_max)
        # Add padding to the bounding box
        x_min_padded = x_min - PADDING
        y_min_padded = y_min - PADDING
        x_max_padded = x_max + PADDING
        y_max_padded = y_max + PADDING
        # Extract the padded bbox from the padded image using numpy slicing
        cutout_padded = img_padded[y_min_padded:y_max_padded, x_min_padded:x_max_padded, :]
        # Use SAM2 on the small cutout to get the predicted mask
        # We assume the center of the cutout contains the sphere, so we set the mask there to 1 to initialize the prediction
        predictor.set_image(cutout_padded)
        mask, _, _ = predictor.predict(point_coords=[[(y_max_padded - y_min_padded) / 2, (x_max_padded - x_min_padded) / 2]], point_labels=[1])
        # We sum the masks over all colour channels and constrain values to [0,1] to get a single mask
        mask = np.clip(np.sum(mask, axis=0), 0, 1).astype(np.uint8)
        # The previous step results in artefacts, so we only keep the largest blob as a first post-processing step
        
        # Define an elliptical kernel (5x5 size can be adjusted)
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9))
        mask = cv2.erode(mask, kernel, iterations=4)
        filtered_mask = MaskHelper.filter_largest_blob(mask, [int((y_max_padded - y_min_padded) / 2), int((x_max_padded - x_min_padded) / 2)])
        filtered_mask = cv2.dilate(filtered_mask, kernel, iterations=4)

        ##################################
        #  ELLIPSE FITTING (REGRESSION)  #
        ##################################
        # Initial guess for the ellipse parameters: center, axes, and angle (in radians)
        x0 = [
            (x_max_padded - x_min_padded) // 2,  # x_c (center x)
            (y_max_padded - y_min_padded) // 2,  # y_c (center y)
            (x_max-x_min + y_max-y_min)//3,  # a (semi-major axis)
            (x_max-x_min + y_max-y_min)//5,  # b (semi-minor axis)
            np.pi/6  # theta (rotation angle in radians)
        ]
        # Bounds for [x_c, y_c, a, b, theta]
        x_dist = x_max_padded - x_min_padded
        y_dist = y_max_padded - y_min_padded
        r_max = max(x_dist, y_dist)/2
        bounds = [
            (3*x_dist/8, 5*x_dist/8),   # x_c bounds
            (3*y_dist/8, 5*y_dist/8),   # y_c bounds
            (0.75*r_max, r_max),                  # a bounds (semi-major axis, min 1 to max_a)
            (0.75*r_max, r_max),                  # b bounds (semi-minor axis, min 1 to max_b)
            (-np.pi, np.pi)          # theta bounds (angle in radians, range [-π/2, π/2])
        ]
        # Define the constraints for the optimization (ellipse ratio constraint)
        # constraints = [
        #     {'type': 'ineq', 'fun': lambda params: MaskHelper.ellipse_ratio_constraint(params, min_ratio=0.8, max_ratio=1.2)}
        # ]

        # # Minimize the loss function with bounds and constraints
        # # result = minimize(MaskHelper.loss_function, x0, args=(filtered_mask, 3, 1), bounds=bounds, constraints=constraints)
        # # ALTERNATIVE: Minimize the loss function with bounds using Powell method (no constraints)
        # result = minimize(MaskHelper.loss_function, x0, args=(filtered_mask, 1, 1), bounds=bounds, method='Powell')

        # # Extract the optimized ellipse parameters and compute the final loss proportional to the mask size
        # x_c_opt, y_c_opt, a_opt, b_opt, theta_opt = result.x
        # final_loss = MaskHelper.loss_function(result.x, filtered_mask) / ((x_max - x_min) * (y_max - y_min))

        # print(f"Initial guess: {x0}")
        # print(f"Optimized parameters: {result.x}")

        # # Convert to  parameters
        # ellipse_params = (x_c_opt+x_min_padded, y_c_opt+y_min_padded, a_opt, b_opt, theta_opt)

        # Fit the ellipse with restarts
        best_ellipse_params, best_loss = MaskHelper.fit_ellipse_with_restarts(filtered_mask, x0, bounds, num_restarts=5, seed=42, normalization_param=(x_max - x_min) * (y_max - y_min))

        # Extract the optimized ellipse parameters
        x_c_opt, y_c_opt, a_opt, b_opt, theta_opt = best_ellipse_params
        ellipse_params = (x_c_opt + x_min_padded, y_c_opt + y_min_padded, a_opt, b_opt, theta_opt)

        print(f"Best optimized ellipse parameters: {ellipse_params}")
        print(f"Best loss: {best_loss}")

        # masks_with_losses.append((fitted_ellipse_mask, final_loss, (y_min_padded, y_max_padded, x_min_padded, x_max_padded)))
        masks_with_losses.append((ellipse_params, best_loss, (filtered_mask, (y_min_padded, y_max_padded, x_min_padded, x_max_padded))))

    ##################################
    #  SPHERE PARAMS POST-PROCESSING #
    ##################################
    # After processing all masks, sort by loss and keep the best `NUM_SPHERES`
    # This upper-bounds the number of spheres extracted
    masks_with_losses.sort(key=lambda x: x[1])  # Sort by the loss (second element of the tuple)
    # Keep the n best masks with the lowest losses
    masks_with_losses = masks_with_losses[:min(NUM_SPHERES, len(masks_with_losses))]
    
    if ORDER_SPHERES:
        # Extract the sphere coordinates from your input list
        ellipses = [(x_c_padded, y_c_padded, a, b, theta) for ((x_c_padded, y_c_padded, a, b, theta), _, _) in masks_with_losses]
        # Step 1: Compute the global center
        global_center = MaskHelper.compute_global_center(ellipses)
        # Step 2 & 3: Calculate the angle for each sphere and sort by angle
        ordered_ellipses = MaskHelper.assign_numbers_by_counterclockwise_angle(ellipses, global_center)

        # Save circle params for current image
        # Format is a list of (number, x_c, y_c, radius, angle)
        ordered_ellipses_no_padding = [(number, x_c_padded - PADDING, y_c_padded - PADDING, a, b, theta, angle) for (number, x_c_padded, y_c_padded, a, b, theta, angle) in ordered_ellipses]
        pkl.dump(ordered_ellipses_no_padding, open(MASK_OUTPUT_FOLDER + f'/{name}_params.pkl', 'wb'))
    else:
        ellipses_no_padding = [(x_c_padded - PADDING, y_c_padded - PADDING, a, b, theta) for ((x_c_padded, y_c_padded, a, b, theta), _, _) in masks_with_losses]
        pkl.dump(ellipses_no_padding, open(MASK_OUTPUT_FOLDER + f'/{name}_params.pkl', 'wb'))
    

    ##################################
    #   SPHERE MASK VISUALIZATION    #
    ##################################
    # Create a mask with all the top masks for visualization purposes
    blended_mask = cv2.cvtColor(np.zeros_like(full_mask_padded), cv2.COLOR_GRAY2RGB)
    for ((x_c_padded, y_c_padded, a, b, theta), _, sam2_params) in masks_with_losses:
        cv2.ellipse(full_mask_padded, (int(x_c_padded), int(y_c_padded)), (int(a), int(b)), theta * 180 / np.pi, 0, 360, 1, thickness=-1)
        # cv2.circle(full_mask_padded, (int(x_c_padded), int(y_c_padded)), int(radius), 1, thickness=-1)
    
        sam2_mask_one_sphere, (y_min_padded, y_max_padded, x_min_padded, x_max_padded) = sam2_params
        # Add the SAM2 mask to the blended mask on the green channel
        blended_mask[y_min_padded:y_max_padded, x_min_padded:x_max_padded, 1] = sam2_mask_one_sphere * 255
    
    blended_mask[:, :, 0] = full_mask_padded * 255

    # Remove the padding from the final mask to match the original image size
    if PADDING > 0:
        full_mask = full_mask_padded[PADDING:-PADDING, PADDING:-PADDING]
        blended_mask = blended_mask[PADDING:-PADDING, PADDING:-PADDING]
    else:
        full_mask = full_mask_padded

    # Save the image with the mask overlay
    cv2.imwrite(MASK_OUTPUT_FOLDER + f'/{name}_blended_masks.jpg', cv2.cvtColor(blended_mask, cv2.COLOR_RGB2BGR))

    # Save the final mask as sparse array. PNG format is lossless and supports binary images.
    # cv2.imwrite(MASK_OUTPUT_FOLDER + f'/{name}_mask.png', full_mask * 255)

    # Convert image to RGB if it's in BGR format (as OpenCV reads in BGR)
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image_with_mask = image_rgb.copy().astype(np.float32)
    # Create an RGB version of the mask
    colored_mask = np.zeros_like(image_with_mask)
    colored_mask[full_mask == 1] = np.array([255, 0, 0])
    colored_mask = colored_mask.astype(np.float32)
    # Perform blending only on the masked areas
    alpha = 0.5  # Transparency factor for the mask
    image_with_mask[full_mask == 1] = cv2.addWeighted(
        colored_mask[full_mask == 1], alpha, image_with_mask[full_mask == 1], 1 - alpha, 0
    )

    # Convert back to uint8 for display or saving
    image_with_mask = image_with_mask.astype(np.uint8)
    # image_with_mask[full_mask == 1] = [255, 0, 0]

    if ORDER_SPHERES:
        # Add assigned sphere ordering to the visualization
        for number, x, y, a, b, theta, angle in ordered_ellipses:
            text = f"{number}"#, ({angle:.1f} deg)"

            # Calculate the size of the text (width, height)
            (text_width, text_height), baseline = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 2, thickness=3)
            # Calculate the new position to center the text
            x_centered = x - PADDING - text_width // 2
            y_centered = y - PADDING + text_height // 2  # To center correctly, add half the height
            cv2.putText(image_with_mask, text, (int(x_centered), int(y_centered)), cv2.FONT_HERSHEY_SIMPLEX, 1.5, 1, thickness=3, bottomLeftOrigin=False)

    # Save the image with the mask overlay
    cv2.imwrite(MASK_OUTPUT_FOLDER + f'/{name}_mask_overlay.jpg', cv2.cvtColor(image_with_mask, cv2.COLOR_RGB2BGR))

print("Sphere mask extraction complete. Time taken: {:.2f} seconds.".format(time() - time_start))