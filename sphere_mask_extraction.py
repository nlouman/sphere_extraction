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
import gc
from skimage.measure import EllipseModel, ransac
from skimage import measure

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
parser.add_argument('--mask-model-config-path', type=str, required=False, default='configs/sam2.1/sam2.1_hiera_l', help='Path to SAM2 model config file')
parser.add_argument('--mask-model-weights-path', type=str, required=False, default='../sam2/checkpoints/sam2.1_hiera_large.pt', help='Path to SAM2 model weights file')
parser.add_argument('--image-folder', type=str, required=True, help='Path to folder with image file(s).')
parser.add_argument('--output-folder', type=str, required=True, help='Folder to save annotated image(s).')
parser.add_argument('--box-threshold', type=float, required=False, default=0.4, help='Bounding box threshold')
parser.add_argument('--text-threshold', type=float, required=False, default=0.25, help='Text threshold')
parser.add_argument('--compression-scale', type=float, required=False, default=0.5, help='Scale to compress image')
parser.add_argument('--num-spheres', type=int, required=True, help='Number of spheres to extract')
parser.add_argument('--padding', type=int, required=False, default=0, help='Padding to add to bounding box during mask extraction')
parser.add_argument('--order-spheres', type=bool, required=False, default=True, help='Order spheres by angle')
parser.add_argument('--restarts', type=int, required=False, default=1, help='Number of restarts for the regression ellipse fitting on the SAM2 mask')
parser.add_argument('--start_from_last_done_mask', action='store_true', help='Skip processing images that already have mask params saved.')

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
NUM_RESTARTS = args.restarts

# Create folder structure
Helper.create_folder_structure(OUTPUT_FOLDER, MASK_OUTPUT_FOLDER, BBOX_OUTPUT_FOLDER)

###################################
# IMAGE PREPROCESSING AND LOADING #
###################################
print("Preprocessing images...")

if os.path.isdir(IMAGE_FOLDER):
    # Get all (uncompressed) images in the folder
    images = [f for f in os.listdir(IMAGE_FOLDER) if f.endswith('.JPG') and not f.endswith('_compressed.jpg')]
    # compressed_images = []
    # for image in images:
    #     image_path = os.path.join(IMAGE_FOLDER, image)
    #     # Compress image if there is no compressed version in the same directory
    #     # Will not overwrite existing compressed images when you change the compression factor
    #     compressed_image = BboxHelper.get_compressed(image_path, compression_factor=COMPRESSION_SCALE)
    #     compressed_image_source, compressed_image_transformed = compressed_image
    #     compressed_images.append((image_path, image.strip('.jpg'), compressed_image_source, compressed_image_transformed))

else:
    raise FileNotFoundError(f"Image folder {IMAGE_FOLDER} is not a valid directory.")

# Now we have "compressed_images", a list of tuples of the form (image_name: str, image_source: np.array, image: torch.Tensor)

##################################
# SPHERE BOUNDING BOX EXTRACTION #
##################################

bbox_checkpoint_path = os.path.join(OUTPUT_FOLDER, 'bbox_checkpoints.pkl')
if os.path.exists(bbox_checkpoint_path):
    print("Loading bounding boxes from checkpoint...")
    with open(bbox_checkpoint_path, 'rb') as f:
        boxes_cxxywh = pkl.load(f)
else:
    print("Extracting sphere bounding boxes...")
    # Load GroundingDINO model
    model = load_model(model_config_path=BBOX_CONFIG_PATH, model_checkpoint_path=BBOX_WEIGHTS_PATH)

    # Predict bounding boxes
    boxes_cxxywh = []
    for image in tqdm.tqdm(images):
        image_path = os.path.join(IMAGE_FOLDER, image)
        name = image.strip('.jpg')
        compressed_image_source, compressed_image_transformed = BboxHelper.get_compressed(image_path, compression_factor=COMPRESSION_SCALE)

        boxes, logits, phrases = predict(
            model=model,
            image=compressed_image_transformed,
            caption=TEXT_PROMPT,
            box_threshold=BOX_TRESHOLD,
            text_threshold=TEXT_TRESHOLD
        )
    # for path, name, image_source, image in tqdm.tqdm(compressed_images):
    #     # Do initial prediction
    #     boxes, logits, phrases = predict(
    #             model=model,
    #             image=image,
    #             caption=TEXT_PROMPT,
    #             box_threshold=BOX_TRESHOLD,
    #             text_threshold=TEXT_TRESHOLD
    #         )
        
        confidence_adjustment = INITIAL_CONFIDENCE_ADJUSTMENT

        # Adjust confidence threshold until the number of spheres is at least equal to the desired number, maxing out at a certain threshold
        while len(boxes) < NUM_SPHERES:
            boxes, logits, phrases = predict(
                model=model,
                image=compressed_image_transformed,
                caption=TEXT_PROMPT,
                box_threshold=BOX_TRESHOLD-confidence_adjustment,
                text_threshold=TEXT_TRESHOLD
            )
            confidence_adjustment += INITIAL_CONFIDENCE_ADJUSTMENT
            if confidence_adjustment > MAX_CONFIDENCE_ADJUSTMENT:
                raise ValueError(f"Could not find enough spheres in image {name}.")
        
        # Save bounding boxes
        # boxes_cxxywh.append((path, name, boxes))
        boxes_cxxywh.append((image_path, name, boxes))
            
        annotated_frame = annotate(image_source=compressed_image_source, boxes=boxes, logits=logits, phrases=phrases)
        cv2.imwrite(BBOX_OUTPUT_FOLDER + f'/{name}_annotated.jpg', annotated_frame)

        with open(os.path.join(OUTPUT_FOLDER, 'bbox_checkpoints.pkl'), 'wb') as f:
            pkl.dump(boxes_cxxywh, f)
        
    del model  # unload GroundingDINO
    torch.cuda.empty_cache()
        

# Convert bounding boxes to xyxy format and add the original image to the list
# boxes_xyxy = []
# for path, name, boxes in boxes_cxxywh:
#     # Load (uncompressed) image
#     image = cv2.imread(path)
#     # Convert bounding boxes to xyxy format
#     box_xyxy = BboxHelper.box_conversion_to_xyxy(image, boxes)
#     # Save bounding boxes
#     boxes_xyxy.append((path, name, image, box_xyxy))



##################################
#    SPHERE PARAMS EXTRACTION    #
##################################
print("Extracting sphere masks...")

# Load SAM2 model
predictor = SAM2ImagePredictor(build_sam2(MASK_CONFIG_PATH, MASK_WEIGHTS_PATH))

# Loop over all images
pbar = tqdm.tqdm(boxes_cxxywh, desc="Processing images")
for path, name, bboxes in pbar:
    mask_param_path = os.path.join(MASK_OUTPUT_FOLDER, f"{name}_params.pkl")
    if args.start_from_last_done_mask and os.path.exists(mask_param_path):
        print(f"Skipping {name}, already processed.")
        continue

    pbar.set_description(f"Processing image ({name})")
    # Load the original image
    image = cv2.imread(path)

    # Convert bounding boxes to xyxy format
    box_xyxy = BboxHelper.box_conversion_to_xyxy(image, bboxes)

    # Add a black border around the original image to prevent indexing errors
    img_padded = cv2.copyMakeBorder(
        image, PADDING, PADDING, PADDING, PADDING, cv2.BORDER_CONSTANT, value=[0, 0, 0]
    )

    # print(image.shape, img_padded.shape, box_xyxy)
    # Create an empty mask for the entire padded image
    full_mask_padded = np.zeros((img_padded.shape[0], img_padded.shape[1]), dtype=np.uint8)

    # Initialize list to store the cutouts, and one to store the masks and losses
    cutouts = []
    masks_with_losses = []

    ##########################################
    #  FILTER OUT TOO-BIG-MASK OUTLIERS     #
    ##########################################

    # Filter bounding boxes before processing to filter out too big ones
    image_area = image.shape[0] * image.shape[1]
    bbox_areas = np.array([(x_max - x_min) * (y_max - y_min) for (x_min, y_min, x_max, y_max) in box_xyxy])

    # Compute stats for outlier rejection
    median_area = np.median(bbox_areas)
    q1 = np.percentile(bbox_areas, 25)
    q3 = np.percentile(bbox_areas, 75)
    iqr = q3 - q1

    # Define outlier threshold — e.g. anything above Q3 + 2.5 * IQR
    upper_threshold = q3 + 2.5 * iqr
    absolute_threshold = image_area / 30  # Optional: also reject anything too huge globally

    # Filter bounding boxes
    filtered_box_xyxy = []
    for idx, (bbox, area) in enumerate(zip(box_xyxy, bbox_areas)):
        if area > upper_threshold or area > absolute_threshold:
            print(f"  [WARNING] Rejecting bbox {idx} in {name} due to large area: {area:.0f} > {upper_threshold:.0f}")
            continue
        filtered_box_xyxy.append(bbox)

    box_xyxy = filtered_box_xyxy

    # Loop over the bounding boxes
    for idx, (x_min, y_min, x_max, y_max) in enumerate(filtered_box_xyxy):
        ##################################
        #     SPHERE MASK EXTRACTION     #
        ##################################
        # Ensure bounding box coordinates are integers
        y_min_unpad_im_coord = int(y_min)
        x_min_unpad_im_coord = int(x_min)
        x_max_unpad_im_coord = int(x_max)
        y_max_unpad_im_coord = int(y_max)
        # Transform the bounding box coordinates to padded image coordinates
        x_min_pad_im_coord = x_min_unpad_im_coord + PADDING
        y_min_pad_im_coord = y_min_unpad_im_coord + PADDING
        x_max_pad_im_coord = x_max_unpad_im_coord + PADDING
        y_max_pad_im_coord = y_max_unpad_im_coord + PADDING
        # Add padding to the bounding box (i.e. make it larger)
        x_min_padded_pad_im_coord = x_min_pad_im_coord - PADDING
        y_min_padded_pad_im_coord = y_min_pad_im_coord - PADDING
        x_max_padded_pad_im_coord = x_max_pad_im_coord + PADDING
        y_max_padded_pad_im_coord = y_max_pad_im_coord + PADDING

        # ALREADY CHECKED: The bounding boxes are correct

        # Extract the padded bbox from the padded image
        cutout_padded_pad_bbox_coord = img_padded[y_min_padded_pad_im_coord:y_max_padded_pad_im_coord, x_min_padded_pad_im_coord:x_max_padded_pad_im_coord, :]
        
        # cutout coordinates
        include_points_im_coord = [
            [(y_max_padded_pad_im_coord - y_min_padded_pad_im_coord) // 2, (x_max_padded_pad_im_coord - x_min_padded_pad_im_coord) // 2],
        ]

        # exclude_points = [
        #     [0, 0],  # Exclude the bottom left corner of the cutout
        #     [0, x_max_padded_pad_im_coord - x_min_padded_pad_im_coord],  # Exclude the top left corner of the cutout
        #     [y_max_padded_pad_im_coord - y_min_padded_pad_im_coord, 0],  # Exclude the top right corner of the cutout
        #     [y_max_padded_pad_im_coord - y_min_padded_pad_im_coord, x_max_padded_pad_im_coord - x_min_padded_pad_im_coord],  # Exclude the bottom right corner of the cutout
        # ]

        point_prompts = include_points_im_coord# + exclude_points
        labels = [0] * len(include_points_im_coord)# + [0] * len(exclude_points)

        # Pass the (smaller, unpadded) cutout to the predictor as a guide
        box_bbox_coord = np.array([PADDING, PADDING, x_max-x_min+PADDING, y_max-y_min+PADDING])

        # ALREADY CHECKED: The include and exclude points are correct, as are the bbox coordinates
        
        # Set the image for the predictor 
        predictor.set_image(cutout_padded_pad_bbox_coord)
        mask, _, _ = predictor.predict(point_coords=point_prompts, point_labels=labels, box=box_bbox_coord, multimask_output=False)
        # We sum the masks over all colour channels and constrain values to [0,1] to get a single mask
        mask = np.clip(np.sum(mask, axis=0), 0, 1).astype(np.uint8)
        # The previous step results in artefacts, so we only keep the largest blob as a first post-processing step

        # # DEBUG: Plot the mask
        # debug_cutout_mask = cutout_padded_pad_bbox_coord.copy()
        # cv2.rectangle(debug_cutout_mask, (int(box_bbox_coord[0]), int(box_bbox_coord[1])), (int(box_bbox_coord[2]), int(box_bbox_coord[3])), (255, 255, 0), 2)
        # for pt in include_points_im_coord:
        #     y, x = int(pt[0]), int(pt[1])
        #     cv2.circle(debug_cutout_mask, (x, y), 5, (0, 255, 0), -1)
        # # for pt in exclude_points:
        # #     y, x = int(pt[0]), int(pt[1])
        # #     cv2.circle(debug_cutout, (x, y), 5, (0, 255, 0), -1)
        # # Convert mask to 3-channel image
        # blue_mask = np.zeros_like(debug_cutout_mask)
        # blue_mask[:, :, 0] = mask * 255  # Blue channel
        # cv2.addWeighted(blue_mask, 0.3, debug_cutout_mask, 0.7, 0, debug_cutout_mask)
        # # Save the mask for debugging
        # cv2.imwrite(os.path.join(MASK_OUTPUT_FOLDER, f"{name}_debug_mask_{idx}.jpg"), debug_cutout_mask)
        
        # Define an elliptical kernel (5x5 size can be adjusted)
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9))
        mask = cv2.dilate(mask, kernel, iterations=4)
        mask = cv2.erode(mask, kernel, iterations=4)
        # mask = cv2.erode(mask, kernel, iterations=4)
        filtered_mask = MaskHelper.filter_largest_blob(mask, [int(include_points_im_coord[0][1]), int(include_points_im_coord[0][0])])

        # 1) Fill any internal holes
        mask_filled = filtered_mask.copy().astype(np.uint8)
        contours, _ = cv2.findContours(mask_filled, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_NONE)
        for cnt in contours:
            # only fill holes (parent == -1 are outer contours)
            cv2.drawContours(mask_filled, [cnt], -1, 1, thickness=cv2.FILLED)

        # 2) Morphological closing to seal small gaps
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (15, 15))
        mask_closed = cv2.morphologyEx(mask_filled, cv2.MORPH_CLOSE, kernel)

        # # DEBUG: Plot the filtered mask
        # debug_cutout = cutout_padded_pad_bbox_coord.copy()
        # mask_overlay = np.zeros_like(debug_cutout)
        # mask_overlay[filtered_mask == 1] = [255, 0, 0]  # Red channel
        # cv2.addWeighted(mask_overlay, 0.3, debug_cutout, 0.7, 0, debug_cutout)
        # cv2.imwrite(os.path.join(MASK_OUTPUT_FOLDER, f"{name}_debug_filtered_mask_{idx}.jpg"), debug_cutout)
        # filtered_mask = cv2.dilate(filtered_mask, kernel, iterations=4)

        ##################################
        #  ELLIPSE FITTING (REGRESSION)  #
        ##################################
        # Initial guess for the ellipse parameters: center, axes, and angle (in radians)
        x0 = [
            (x_max_padded_pad_im_coord - x_min_padded_pad_im_coord) // 2,  # x_c (center x)
            (y_max_padded_pad_im_coord - y_min_padded_pad_im_coord) // 2,  # y_c (center y)
            (x_max-x_min + y_max-y_min)//3,  # a (semi-major axis)
            (x_max-x_min + y_max-y_min)//5,  # b (semi-minor axis)
            np.pi/6  # theta (rotation angle in radians)
        ]
        # Bounds for [x_c, y_c, a, b, theta]
        x_dist = x_max_padded_pad_im_coord - x_min_padded_pad_im_coord
        y_dist = y_max_padded_pad_im_coord - y_min_padded_pad_im_coord
        r_max = max(x_max-x_min, y_max-y_min)/2
        bounds = [
            (3*x_dist/8, 5*x_dist/8),   # x_c bounds
            (3*y_dist/8, 5*y_dist/8),   # y_c bounds
            (0.75*r_max, 1.1*r_max),                  # a bounds (semi-major axis, min 1 to max_a)
            (0.75*r_max, 1.1*r_max),                  # b bounds (semi-minor axis, min 1 to max_b)
            (-np.pi, np.pi)          # theta bounds (angle in radians, range [-π/2, π/2])
        ]

        # ALREADY CHECKED: The initial guess and bounds are correct

        # Fit the ellipse with restarts
        best_ellipse_params, best_loss = MaskHelper.fit_ellipse_with_restarts(mask_closed, x0, bounds, num_restarts=NUM_RESTARTS, seed=42, normalization_param=x_dist * y_dist)

        # Extract the optimized ellipse parameters
        x_c_opt, y_c_opt, a_opt, b_opt, theta_opt = best_ellipse_params

        # debug_cutout = cutout_padded_pad_bbox_coord.copy()
        # cv2.ellipse(
        #     debug_cutout,
        #     (int(x_c_opt), int(y_c_opt)),
        #     (int(a_opt), int(b_opt)),
        #     theta_opt * 180/np.pi,
        #     0, 360,
        #     (0, 255, 255), 2
        # )
        # # draw bounding‐box in cutout coords (0,0)→(w,h)
        # h, w = cutout_padded_pad_bbox_coord.shape[:2]
        # cv2.rectangle(debug_cutout, (0,0), (w-1,h-1), (255,255,0), 1)

        # cv2.imwrite(os.path.join(MASK_OUTPUT_FOLDER, f"{name}_debug_cutout_{idx}.jpg"),
        #             debug_cutout)

        # convert to original-padded-image coords 
        ellipse_params = (x_c_opt - PADDING + x_min_unpad_im_coord, y_c_opt - PADDING + y_min_unpad_im_coord, a_opt, b_opt, theta_opt)

        # print(f"Best optimized ellipse parameters: {ellipse_params}")
        print(f"Best loss: {best_loss}")

        if best_loss > 0.1:
            print(f"  [WARNING] High loss detected: {best_loss}. Trying RANSAC.")

            # DEBUG: Plot the mask
            debug_cutout = cutout_padded_pad_bbox_coord.copy()
            blue_mask = np.zeros_like(debug_cutout)
            blue_mask[:, :, 0] = mask_closed * 255  # Blue channel
            cv2.addWeighted(blue_mask, 0.3, debug_cutout, 0.7, 0, debug_cutout)

            # Save debug image
            cv2.imwrite(os.path.join(MASK_OUTPUT_FOLDER, f"{name}_filtered_mask_{idx}.jpg"), debug_cutout)
            debug_global = image.copy()

            # 1) draw the original bounding box on the full image
            cv2.rectangle(
                debug_global,
                pt1=(x_min_unpad_im_coord, y_min_unpad_im_coord),
                pt2=(x_max_unpad_im_coord, y_max_unpad_im_coord),
                color=(255, 255, 0),   # cyan
                thickness=2
            )

            # 2) draw the fitted ellipse in global coords
            cv2.ellipse(
                debug_global,
                (int(x_c_opt + x_min_unpad_im_coord - PADDING), int(y_c_opt + y_min_unpad_im_coord - PADDING)),
                (int(a_opt), int(b_opt)),
                theta_opt * 180/np.pi,
                0, 360,
                (0, 255, 255),   # yellow
                2
            )

            cv2.imwrite(os.path.join(MASK_OUTPUT_FOLDER, f"{name}_debug_global_{idx}.jpg"),
                        debug_global)
            

            # 1) extract edge points from your binary mask
            edges = cv2.Canny(mask_closed*255, 50, 150)
            ys, xs = np.nonzero(edges)
            data = np.column_stack([xs, ys])

            # 2) run RANSAC
            model_robust, inliers = ransac(data, EllipseModel, min_samples=5,
                                        residual_threshold=2, max_trials=1000)

            xc, yc, a, b, theta = model_robust.params
            ellipse_params = (xc - PADDING + x_min_unpad_im_coord, yc - PADDING + y_min_unpad_im_coord, a, b, theta)
            # print("High loss detected, calculating ellipse parameters based on bbox.")
            # # If the loss is too high, use the bounding box as a fallback
            # x_c_opt = (x_min_unpad_im_coord + x_max_unpad_im_coord) / 2
            # y_c_opt = (y_min_unpad_im_coord + y_max_unpad_im_coord) / 2
            # a_opt = (x_max_unpad_im_coord - x_min_unpad_im_coord) / 2
            # b_opt = (y_max_unpad_im_coord - y_min_unpad_im_coord) / 2
            # theta_opt = 0.0  # No rotation
            # ellipse_params = (x_c_opt - PADDING + x_min_unpad_im_coord, y_c_opt - PADDING + y_min_unpad_im_coord, a_opt, b_opt, theta_opt)

        # masks_with_losses.append((fitted_ellipse_mask, final_loss, (y_min_padded, y_max_padded, x_min_padded, x_max_padded)))
        masks_with_losses.append((ellipse_params, best_loss, (mask_closed, (y_min_unpad_im_coord, y_max_unpad_im_coord, x_min_unpad_im_coord, x_max_unpad_im_coord))))

    # del image, box_xyxy, full_mask_padded, blended_mask, masks_with_losses
    # If you re-instantiated predictor below, also del that
    gc.collect()
    
    ##################################
    #  SPHERE PARAMS POST-PROCESSING #
    ##################################
    
    
    # areas = np.array([m.sum() for (_, _, (m, _)) in masks_with_losses])
    # if len(areas):
    #     med = np.median(areas)
    #     thr = med * 3.0  # e.g. reject any mask >3× median size
    #     filtered = []
    #     for entry, area in zip(masks_with_losses, areas):
    #         if area <= thr:
    #             filtered.append(entry)
    #         else:
    #             print(f"Dropping outlier mask area={area} > {thr:.0f}")
    #     masks_with_losses = filtered

    ##################################
    #  SPHERE PARAMS POST-PROCESSING #
    ##################################

    # After processing all masks, sort by loss and keep the best `NUM_SPHERES`
    # This upper-bounds the number of spheres extracted
    masks_with_losses.sort(key=lambda x: x[1])  # Sort by the loss (second element of the tuple)
    # Keep the n best masks with the lowest losses
    masks_with_losses = masks_with_losses[:min(NUM_SPHERES, len(masks_with_losses))]

    if len(masks_with_losses) < NUM_SPHERES:
        print(f"  [WARNING] Only {len(masks_with_losses)} masks extracted, less than requested {NUM_SPHERES}.")
    
    if ORDER_SPHERES:
        # Extract the sphere coordinates from your input list
        ellipses = [(x_c, y_c, a, b, theta) for ((x_c, y_c, a, b, theta), _, _) in masks_with_losses]
        # Step 1: Compute the global center
        global_center = MaskHelper.compute_global_center(ellipses)
        # Step 2 & 3: Calculate the angle for each sphere and sort by angle
        ordered_ellipses = MaskHelper.assign_numbers_by_counterclockwise_angle(ellipses, global_center)

        # Save circle params for current image
        # Format is a list of (number, x_c, y_c, radius, angle)
        ordered_ellipses_no_padding = [(number, x_c, y_c, a, b, theta, angle) for (number, x_c, y_c, a, b, theta, angle) in ordered_ellipses]
        pkl.dump(ordered_ellipses_no_padding, open(MASK_OUTPUT_FOLDER + f'/{name}_params.pkl', 'wb'))
    else:
        ellipses_no_padding = [(x_c, y_c, a, b, theta) for ((x_c, y_c, a, b, theta), _, _) in masks_with_losses]
        pkl.dump(ellipses_no_padding, open(MASK_OUTPUT_FOLDER + f'/{name}_params.pkl', 'wb'))
    
    ##################################
    #   SPHERE MASK VISUALIZATION    #
    ##################################
    # 1) Draw all fitted ellipses on a copy of the original image
    ellipse_viz = image.copy()
    for ((xc, yc, a, b, theta), _, _) in masks_with_losses:
        cv2.ellipse(
            ellipse_viz,
            (int(xc), int(yc)),
            (int(a), int(b)),
            theta * 180/np.pi,
            0, 360,
            (0, 255, 255), 2  # yellow
        )

    # If you’ve numbered them, you can overlay the numbers here:
    if ORDER_SPHERES:
        for number, xc, yc, a, b, theta, angle in ordered_ellipses:
            cv2.putText(
                ellipse_viz,
                str(number),
                (int(xc), int(yc)),
                cv2.FONT_HERSHEY_SIMPLEX,
                1.2, (0,255,255), 3
            )

    cv2.imwrite(
        os.path.join(MASK_OUTPUT_FOLDER, f"{name}_ellipse_overlay.jpg"),
        ellipse_viz
    )

    # 2) Create a mask‐blend overlay in green on the original
    # full_mask is assumed to be the same size as image
    mask_viz = image.copy().astype(np.float32)
    green_mask = np.zeros_like(mask_viz)
    if PADDING > 0:
        full_mask = full_mask_padded[PADDING:-PADDING, PADDING:-PADDING]
    else:
        full_mask = full_mask_padded
    green_mask[full_mask == 1] = (0, 255, 0)   # green

    alpha = 0.5
    cv2.addWeighted(green_mask, alpha, mask_viz, 1 - alpha, 0, mask_viz)
    mask_viz = mask_viz.astype(np.uint8)

    cv2.imwrite(
        os.path.join(MASK_OUTPUT_FOLDER, f"{name}_mask_overlay.jpg"),
        mask_viz
    )
    # # Create a mask with all the top masks for visualization purposes
    # blended_mask = cv2.cvtColor(np.zeros_like(full_mask_padded), cv2.COLOR_GRAY2RGB)
    # # for ((x_c_padded, y_c_padded, a, b, theta), _, sam2_params) in masks_with_losses:
    # #     cv2.ellipse(full_mask_padded, (int(x_c_padded), int(y_c_padded)), (int(a), int(b)), theta * 180 / np.pi, 0, 360, 1, thickness=-1)
    # #     # cv2.circle(full_mask_padded, (int(x_c_padded), int(y_c_padded)), int(radius), 1, thickness=-1)
    
    # #     sam2_mask_one_sphere, (y_min_padded, y_max_padded, x_min_padded, x_max_padded) = sam2_params
    # #     # Add the SAM2 mask to the blended mask on the green channel
    # #     blended_mask[y_min_padded:y_max_padded, x_min_padded:x_max_padded, 1] = sam2_mask_one_sphere * 255
    # for ((x_c_padded, y_c_padded, a, b, theta), _, sam2_params) in masks_with_losses:
    #     # first draw your fitted ellipse as before
    #     cv2.ellipse(full_mask_padded,
    #                 (int(x_c_padded), int(y_c_padded)),
    #                 (int(a), int(b)),
    #                 theta * 180/np.pi,
    #                 0, 360, 1, thickness=-1)

    #     # unpack the exact padded‐cutout coords you used for predictor.set_image(...)
    #     sam2_mask_one_sphere, (y0, y1, x0, x1) = sam2_params

    #     # get the mask’s own height/width
    #     h, w = sam2_mask_one_sphere.shape

    #     # now paste back into blended_mask using (y0, x0) + (h, w)
    #     blended_mask[y0 : y0 + h, x0 : x0 + w, 1] = sam2_mask_one_sphere * 255
    
    # blended_mask[:, :, 0] = full_mask_padded * 255

    # # Remove the padding from the final mask to match the original image size
    # if PADDING > 0:
    #     full_mask = full_mask_padded[PADDING:-PADDING, PADDING:-PADDING]
    #     blended_mask = blended_mask[PADDING:-PADDING, PADDING:-PADDING]
    # else:
    #     full_mask = full_mask_padded

    # # Save the image with the mask overlay
    # cv2.imwrite(MASK_OUTPUT_FOLDER + f'/{name}_blended_masks.jpg', cv2.cvtColor(blended_mask, cv2.COLOR_RGB2BGR))

    # # Save the final mask as sparse array. PNG format is lossless and supports binary images.
    # # cv2.imwrite(MASK_OUTPUT_FOLDER + f'/{name}_mask.png', full_mask * 255)

    # # Convert image to RGB if it's in BGR format (as OpenCV reads in BGR)
    # image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    # image_with_mask = image_rgb.copy().astype(np.float32)
    # # Create an RGB version of the mask
    # colored_mask = np.zeros_like(image_with_mask)
    # colored_mask[full_mask == 1] = np.array([255, 0, 0])
    # colored_mask = colored_mask.astype(np.float32)
    # # Perform blending only on the masked areas
    # alpha = 0.5  # Transparency factor for the mask
    # image_with_mask[full_mask == 1] = cv2.addWeighted(
    #     colored_mask[full_mask == 1], alpha, image_with_mask[full_mask == 1], 1 - alpha, 0
    # )

    # # Convert back to uint8 for display or saving
    # image_with_mask = image_with_mask.astype(np.uint8)
    # # image_with_mask[full_mask == 1] = [255, 0, 0]

    # if ORDER_SPHERES:
    #     # Add assigned sphere ordering to the visualization
    #     for number, x, y, a, b, theta, angle in ordered_ellipses:
    #         text = f"{number}"#, ({angle:.1f} deg)"

    #         # Calculate the size of the text (width, height)
    #         (text_width, text_height), baseline = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 2, thickness=3)
    #         # Calculate the new position to center the text
    #         x_centered = x - PADDING - text_width // 2
    #         y_centered = y - PADDING + text_height // 2  # To center correctly, add half the height
    #         cv2.putText(image_with_mask, text, (int(x_centered), int(y_centered)), cv2.FONT_HERSHEY_SIMPLEX, 1.5, 1, thickness=3, bottomLeftOrigin=False)

    # # Save the image with the mask overlay
    # cv2.imwrite(MASK_OUTPUT_FOLDER + f'/{name}_mask_overlay.jpg', cv2.cvtColor(image_with_mask, cv2.COLOR_RGB2BGR))

print("Sphere mask extraction complete. Time taken: {:.2f} seconds.".format(time() - time_start))