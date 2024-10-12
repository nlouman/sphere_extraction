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

args = parser.parse_args()

# Args for image loading
IMAGE_FOLDER = args.image_folder
COMPRESSION_SCALE = args.compression_scale

# Args for output
OUTPUT_FOLDER = args.output_folder
MASK_OUTPUT_FOLDER = OUTPUT_FOLDER + '/masks'
BBOX_OUTPUT_FOLDER = OUTPUT_FOLDER + '/bounding_boxes'

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
        filtered_mask = MaskHelper.filter_largest_blob(mask, [int((y_max_padded - y_min_padded) / 2), int((x_max_padded - x_min_padded) / 2)])

        ##################################
        #  ELLIPSE FITTING (REGRESSION)  #
        ##################################
        # Initial guess for the ellipse parameters: center, axes, and angle (in radians)
        x0 = [
            (x_max_padded - x_min_padded) // 2,  # x_c (center x)
            (y_max_padded - y_min_padded) // 2,  # y_c (center y)
            (x_max-x_min + y_max-y_min)//4,  # a (semi-major axis)
            (x_max-x_min + y_max-y_min)//4,  # b (semi-minor axis)
            np.pi/4  # theta (rotation angle in radians)
        ]
        # Bounds for [x_c, y_c, a, b, theta]
        bounds = [
            (0, x_max_padded-x_min_padded),   # x_c bounds
            (0, y_max_padded-y_min_padded),   # y_c bounds
            (1, (x_max-x_min + y_max-y_min)//2),                  # a bounds (semi-major axis, min 1 to max_a)
            (1, (x_max-x_min + y_max-y_min)//2),                  # b bounds (semi-minor axis, min 1 to max_b)
            (-np.pi/2, np.pi/2)          # theta bounds (angle in radians, range [-π/2, π/2])
        ]
        # Define the constraints for the optimization (ellipse ratio constraint)
        constraints = [
            {'type': 'ineq', 'fun': lambda params: MaskHelper.ellipse_ratio_constraint(params, min_ratio=0.8, max_ratio=1.2)}
        ]

        # Minimize the loss function with bounds and constraints
        result = minimize(MaskHelper.loss_function, x0, args=(filtered_mask, 1, 1), bounds=bounds, constraints=constraints)
        # ALTERNATIVE: Minimize the loss function with bounds using Powell method (no constraints)
        # result = minimize(MaskHelper.loss_function, x0, args=(filtered_mask, 1, 1), bounds=bounds)

        # Extract the optimized ellipse parameters and compute the final loss proportional to the mask size
        x_c_opt, y_c_opt, a_opt, b_opt, theta_opt = result.x
        final_loss = MaskHelper.loss_function(result.x, filtered_mask) / ((x_max - x_min) * (y_max - y_min))

        # Convert to circle parameters
        circle_params = (x_c_opt+x_min_padded, y_c_opt+y_min_padded, (a_opt + b_opt)/2)

        # masks_with_losses.append((fitted_ellipse_mask, final_loss, (y_min_padded, y_max_padded, x_min_padded, x_max_padded)))
        masks_with_losses.append((circle_params, final_loss))

    ##################################
    #  SPHERE PARAMS POST-PROCESSING #
    ##################################
    # After processing all masks, sort by loss and keep the best `NUM_SPHERES`
    # This upper-bounds the number of spheres extracted
    masks_with_losses.sort(key=lambda x: x[1])  # Sort by the loss (second element of the tuple)
    # Keep the n best masks with the lowest losses
    masks_with_losses = masks_with_losses[:min(NUM_SPHERES, len(masks_with_losses))]
    
    # Extract the sphere coordinates from your input list
    spheres = [(x_c_padded, y_c_padded, radius) for ((x_c_padded, y_c_padded, radius), _) in masks_with_losses]
    # Step 1: Compute the global center
    global_center = MaskHelper.compute_global_center(spheres)
    # Step 2 & 3: Calculate the angle for each sphere and sort by angle
    ordered_spheres = MaskHelper.assign_numbers_by_counterclockwise_angle(spheres, global_center)

    # Save circle params for current image
    # Format is a list of (number, x_c, y_c, radius, angle)
    ordered_spheres_no_padding = [(number, x_c_padded - PADDING, y_c_padded - PADDING, radius, angle) for (number, x_c_padded, y_c_padded, radius, angle) in ordered_spheres]
    pkl.dump(ordered_spheres_no_padding, open(MASK_OUTPUT_FOLDER + f'/{name}_params.pkl', 'wb'))

    ##################################
    #   SPHERE MASK VISUALIZATION    #
    ##################################
    # Create a mask with all the top masks for visualization purposes
    for ((x_c_padded, y_c_padded, radius), _) in masks_with_losses:
        cv2.circle(full_mask_padded, (int(x_c_padded), int(y_c_padded)), int(radius), 1, thickness=-1)

    # Remove the padding from the final mask to match the original image size
    if PADDING > 0:
        full_mask = full_mask_padded[PADDING:-PADDING, PADDING:-PADDING]
    else:
        full_mask = full_mask_padded

    # Save the final mask as sparse array. PNG format is lossless and supports binary images.
    # cv2.imwrite(MASK_OUTPUT_FOLDER + f'/{name}_mask.png', full_mask * 255)

    # Convert image to RGB if it's in BGR format (as OpenCV reads in BGR)
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image_with_mask = image_rgb.copy()
    image_with_mask[full_mask == 1] = [255, 0, 0]

    # Add assigned sphere ordering to the visualization
    for number, x, y, radius, angle in ordered_spheres:
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