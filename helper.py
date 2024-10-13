import cv2
from groundingdino.util.inference import load_image, box_convert
import numpy as np
import torch
from typing import Tuple
import os
import math
from scipy.optimize import minimize

class Helper:
    def create_folder_structure(output_folder, mask_output_folder, bbox_output_folder):
        """
        Create folder structure for storing output images and masks.
        
        Args:
            output_folder (str): Path to the output folder.
            mask_output_folder (str): Path to the mask output folder.
            bbox_output_folder (str): Path to the bounding box output folder.
        """
        if not os.path.isdir(output_folder):
            os.makedirs(output_folder)
        if not os.path.isdir(mask_output_folder):
            os.makedirs(mask_output_folder)
        if not os.path.isdir(bbox_output_folder):
            os.makedirs(bbox_output_folder)

class BboxHelper:
    def get_compressed(image_path: str, compression_factor: float = 0.5, method: str = 'groundingdino') -> Tuple[np.array, torch.Tensor]:
        """
        Compress image if there is no compressed version in the same directory.
        Otherwise, load the compressed image. Wrapper around load_image from GroundingDINO.

        Args:
            image_path (str): Path to the image file.
            compression_factor (float): Factor to scale down the image. Actual compression will be compression_factor^2.
            method (str): Method to use for loading the image.
        Returns:
            Tuple[np.array, torch.Tensor]: Original image and transformed image.
        """
        if method == 'groundingdino':
            if not image_path.endswith('_compressed.jpg'):
                compressed_path = image_path.strip('.jpg') + "_compressed.jpg"
                # Load original image
                image = cv2.imread(image_path)
                # Scale down the image
                compressed_image = cv2.resize(image, (0, 0), fx=compression_factor, fy=compression_factor, interpolation=cv2.INTER_AREA)
                # Save compressed image
                cv2.imwrite(compressed_path, compressed_image)
                image_path = compressed_path
            return load_image(image_path)
        else:
            raise NotImplementedError(f"Loading for method {method} is not implemented.")
        
    def box_conversion_to_xyxy(img, boxes: torch.Tensor) -> np.array:
        """
        Convert bounding boxes from cxcywh format to xyxy format matching dimensions of passed image. Wrapper around box_convert from GroundingDINO.
        
        Args:
            img (np.array): Image.
            boxes (torch.Tensor): Bounding boxes in cxcywh
        Returns:
            np.array: Bounding boxes in xyxy format.
        """
        h, w, _ = img.shape
        boxes_expanded = boxes * torch.Tensor([w, h, w, h])
        boxes_xyxy = box_convert(boxes=boxes_expanded, in_fmt="cxcywh", out_fmt="xyxy").numpy()
        return boxes_xyxy

# Class for mask post-processing
class MaskHelper:
    def filter_largest_blob(mask, center) -> np.ndarray:
        """
        Filters the mask to only keep the largest blob, assuming the largest blob contains the center point.
        
        Args:
        mask (np.ndarray): The binary mask (0s and 1s).
        center (tuple): The (x, y) coordinates of the center point of the mask.

        Returns:
        np.ndarray: The filtered mask, containing only the largest blob.
        """
        # Ensure mask is binary (values 0 or 1)
        mask = (mask > 0).astype(np.uint8)

        # Label all connected components in the binary mask
        num_labels, labels_im = cv2.connectedComponents(mask, connectivity=4)

        # Get the label of the connected component that contains the center point
        center_label = labels_im[center[1], center[0]]  # Remember, it's (y, x) for numpy

        # Create a new mask that only contains the blob with the center point
        largest_blob = np.zeros_like(mask)
        largest_blob[labels_im == center_label] = 1  # Keep only the component containing the center

        return largest_blob
    
    ############################################################################################################
    # Ellipse Fitting
    ############################################################################################################

    # Define the constraint to keep the ellipse ratio close to 1 (for near-circle shapes)
    def ellipse_ratio_constraint(params, min_ratio=0.8, max_ratio=1.2):
        """
        Constraint function for the ellipse ratio (a / b).
        Ensure that min_ratio <= a / b <= max_ratio.
        """
        a = params[2]  # semi-major axis
        b = params[3]  # semi-minor axis
        
        # Ratio of a/b must be within the bounds
        ratio = a / b
        return [ratio - min_ratio, max_ratio - ratio]

    def create_ellipse_mask(x_c, y_c, a, b, theta, shape):
        """
        Creates a binary mask for an ellipse with center (x_c, y_c), axes (a, b), and angle theta.
        
        Parameters:
        x_c (float): x-coordinate of the ellipse center.
        y_c (float): y-coordinate of the ellipse center.
        a (float): Semi-major axis.
        b (float): Semi-minor axis.
        theta (float): Rotation angle in radians.
        shape (tuple): Shape of the output mask (height, width).
        
        Returns:
        np.ndarray: Binary mask with the ellipse.
        """
        Y, X = np.ogrid[:shape[0], :shape[1]]
        cos_angle = np.cos(theta)
        sin_angle = np.sin(theta)

        # Ellipse equation: ((X' / a)^2 + (Y' / b)^2 <= 1)
        X_rot = (X - x_c) * cos_angle + (Y - y_c) * sin_angle
        Y_rot = -(X - x_c) * sin_angle + (Y - y_c) * cos_angle

        ellipse_mask = ((X_rot / a) ** 2 + (Y_rot / b) ** 2) <= 1
        return ellipse_mask.astype(np.uint8)

    def loss_function(params, mask, alpha=1, beta=1):
        """
        Loss function that penalizes false positives and false negatives for an ellipse fit.
        
        Parameters:
        params (list): [x_c, y_c, a, b, theta] - ellipse parameters (center, axes, and angle).
        mask (np.ndarray): Binary mask of the original shape.
        alpha (float): Penalty for false positives (inside the ellipse but not in the mask).
        beta (float): Penalty for false negatives (outside the ellipse but in the mask).
        
        Returns:
        float: Loss value.
        """
        x_c, y_c, a, b, theta = params
        predicted_ellipse = MaskHelper.create_ellipse_mask(x_c, y_c, a, b, theta, mask.shape)
        
        # False positives: Inside predicted ellipse but not in the mask
        false_positives = (predicted_ellipse == 1) & (mask == 0)
        
        # False negatives: Outside predicted ellipse but inside the mask
        false_negatives = (predicted_ellipse == 0) & (mask == 1)
        
        loss = alpha * np.sum(false_positives) + beta * np.sum(false_negatives)

        return loss
    
        
    def fit_ellipse_with_restarts(filtered_mask, x0, bounds, num_restarts=5, seed=42, normalization_param=1):
        """
        Fit an ellipse using optimization, with multiple restarts by perturbing initial guesses.
        
        Parameters:
        filtered_mask (np.ndarray): The binary mask to fit the ellipse to.
        x0 (list): The initial guess for ellipse parameters [x_c, y_c, a, b, theta].
        bounds (list of tuples): The bounds for [x_c, y_c, a, b, theta].
        num_restarts (int): Number of times to restart the optimization with randomized starting parameters.
        seed (int): Seed for the random number generator to ensure reproducibility.
        
        Returns:
        tuple: Best ellipse parameters (x_c_opt, y_c_opt, a_opt, b_opt, theta_opt) and final loss.
        """
        rng = np.random.default_rng(seed)  # Pseudo-random generator for reproducibility
        best_result = None
        best_loss = float('inf')

        # Function to randomly perturb initial guess within bounds
        def randomize_initial_guess(x0, bounds, rng):
            perturbed_x0 = []
            for param, (low, high) in zip(x0, bounds):
                std_dev = (high - low) / 10  # Standard deviation for noise is 10% of the range
                perturbed_value = rng.normal(loc=param, scale=std_dev)  # Generate normal noise
                perturbed_value = np.clip(perturbed_value, low, high)  # Ensure within bounds
                perturbed_x0.append(perturbed_value)
            return perturbed_x0

        # Run multiple optimization attempts with randomized initial guesses
        for restart in range(num_restarts):
            # Randomize the initial guess
            randomized_x0 = randomize_initial_guess(x0, bounds, rng)

            # Minimize the loss function using the Powell method
            result = minimize(MaskHelper.loss_function, randomized_x0, args=(filtered_mask, 1, 1), bounds=bounds, method='Powell')

            # Calculate the final loss for the current optimization
            final_loss = MaskHelper.loss_function(result.x, filtered_mask) / normalization_param

            # Track the best result based on the loss
            if final_loss < best_loss:
                best_loss = final_loss
                best_result = result.x

            print(f"Restart {restart + 1}/{num_restarts}: Initial guess = {randomized_x0}, Optimized parameters = {result.x}, Loss = {final_loss}")

        return best_result, best_loss
    
    ############################################################################################################
    # Functions for ordering spheres
    ############################################################################################################

    def compute_global_center(spheres) -> Tuple[float, float]:
        """
        Compute the global center (average) of all sphere coordinates.
        
        Parameters:
        spheres (list of tuples): A list of tuples where each tuple contains (x_center, y_center, a, b, theta).

        Returns:
        tuple: (x_avg, y_avg) the global center
        """
        # Unpack x and y coordinates
        x_coords = [sphere[0] for sphere in spheres]
        y_coords = [sphere[1] for sphere in spheres]
        
        # Calculate the average x and y coordinates
        x_avg = np.mean(x_coords)
        y_avg = np.mean(y_coords)
        
        return x_avg, y_avg

    def calculate_angle(x_center, y_center, global_center) -> float:
        """
        Calculate the angle of a point relative to the global center.
        
        Parameters:
        x_center (float): x-coordinate of the point.
        y_center (float): y-coordinate of the point.
        global_center (tuple): (x_avg, y_avg) global center coordinates.

        Returns:
        float: The angle in degrees relative to the horizontal line going left from the center (range: [0, 360]).
        """
        x_global, y_global = global_center
        
        # Calculate the angle using atan2
        dx = x_center - x_global
        dy = y_center - y_global
        angle_rad = math.atan2(dy, dx)  # Angle in radians
        
        # Convert radians to degrees and normalize the angle to [0, 360]
        angle_deg = math.degrees(angle_rad)
        
        if angle_deg < 0:
            angle_deg += 360  # Normalize to the range [0, 360]
        
        # angle_deg = angle_deg  # Rotate the angle by 90 degrees to start from the left side
        
        return angle_deg
    
    def find_isolated_sphere(spheres) -> Tuple[float, float, float]:
        """
        Find the isolated sphere, i.e., the sphere with the greatest distance to its nearest neighbor.

        Parameters:
        spheres (list of tuples): A list of tuples where each tuple contains (x_center, y_center, a, b, theta)

        Returns:
        tuple: The isolated sphere (x_center, y_center, a, b, theta)
        """
        max_min_distance = -1
        isolated_sphere = None

        # Calculate the pairwise distances between all spheres
        for i, sphere_i in enumerate(spheres):
            x_i, y_i, _, _, _ = sphere_i
            min_distance = float('inf')

            # Find the minimum distance to any other sphere
            for j, sphere_j in enumerate(spheres):
                if i == j:
                    continue  # Skip comparing the sphere with itself
                x_j, y_j, _, _, _ = sphere_j
                distance = np.sqrt((x_j - x_i) ** 2 + (y_j - y_i) ** 2)
                min_distance = min(min_distance, distance)
            
            # Track the sphere with the largest minimum distance to its nearest neighbor
            if min_distance > max_min_distance:
                max_min_distance = min_distance
                isolated_sphere = sphere_i

        return isolated_sphere

    def assign_numbers_by_counterclockwise_angle(spheres, global_center) -> list:
        """
        Assign numbers to spheres based on their angle relative to the global center.
        
        Args:
        spheres (list of tuples): A list of tuples where each tuple contains (x_center, y_center, a, b, theta).
        global_center (tuple): (x_avg, y_avg) global center coordinates.

        Returns:
        list of tuples: A list of spheres sorted by increasing angle, with assigned numbers in the format (number, x_center, y_center, a, b, theta, angle).
        """
        # Find the isolated sphere using pairwise distances
        isolated_sphere = MaskHelper.find_isolated_sphere(spheres)

        # Calculate the angle for each sphere
        spheres_with_angles = []
        for sphere in spheres:
            x_center, y_center, a, b, theta = sphere
            angle = MaskHelper.calculate_angle(x_center, y_center, global_center)
            spheres_with_angles.append((x_center, y_center, a, b, theta, angle))

        # Sort the spheres by angle in counterclockwise direction
        sorted_spheres = sorted(spheres_with_angles, key=lambda s: s[-1])

        # Find the isolated sphere in the sorted list and start numbering from it
        isolated_sphere_angle = MaskHelper.calculate_angle(isolated_sphere[0], isolated_sphere[1], global_center)
        sorted_spheres = sorted(sorted_spheres, key=lambda s: (s[-1] > isolated_sphere_angle, -s[-1]))

        # Assign numbers based on the sorted order, starting with the isolated sphere
        numbered_spheres = []
        for i, (x_center, y_center, a, b, theta, angle) in enumerate(sorted_spheres):
            numbered_spheres.append((i + 1, x_center, y_center, a, b, theta, angle))  # i+1 for numbering starting at 1
        
        return numbered_spheres