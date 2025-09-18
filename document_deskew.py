import cv2
import numpy as np
import base64
from typing import Tuple, Optional

# Helper function (already defined, ensure it's present)
def largest_contour_from_thresh(thresh: np.ndarray) -> Optional[np.ndarray]:
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None
    return max(contours, key=cv2.contourArea)

# Helper function (already defined, ensure it's present)
def order_points(pts: np.ndarray) -> np.ndarray:
    rect = np.zeros((4, 2), dtype="float32")
    s = pts.sum(axis=1)
    rect[0] = pts[np.argmin(s)]
    rect[2] = pts[np.argmax(s)]
    diff = np.diff(pts, axis=1)
    rect[1] = pts[np.argmin(diff)]
    rect[3] = pts[np.argmax(diff)]
    return rect

# Helper function (already defined, ensure it's present)
def four_point_transform(image: np.ndarray, pts: np.ndarray) -> np.ndarray:
    rect = order_points(pts)
    (tl, tr, br, bl) = rect
    widthA = np.sqrt(((br[0] - bl[0]) ** 2) + ((br[1] - bl[1]) ** 2))
    widthB = np.sqrt(((tr[0] - tl[0]) ** 2) + ((tr[1] - tl[1]) ** 2))
    maxWidth = max(int(widthA), int(widthB))
    heightA = np.sqrt(((tr[0] - br[0]) ** 2) + ((tr[1] - br[1]) ** 2))
    heightB = np.sqrt(((tl[0] - bl[0]) ** 2) + ((tl[1] - bl[1]) ** 2))
    maxHeight = max(int(heightA), int(heightB))
    dst = np.array([
        [0, 0],
        [maxWidth - 1, 0],
        [maxWidth - 1, maxHeight - 1],
        [0, maxHeight - 1]], dtype="float32")
    M = cv2.getPerspectiveTransform(rect, dst)
    warped = cv2.warpPerspective(image, M, (maxWidth, maxHeight))
    return warped

# Helper function (already defined, ensure it's present)
def imencode_to_base64(image: np.ndarray, ext: str = ".jpg") -> str:
    _, im_arr = cv2.imencode(ext, image)
    im_bytes = im_arr.tobytes()
    return base64.b64encode(im_bytes).decode('utf-8')

# In document_deskew.py

def estimate_skew_angle(image: np.ndarray, resize_max_side: int = 800) -> float:
    """
    Estimate skew angle (in degrees).
    Accepts either a color (BGR) or a grayscale image.
    """
    h0, w0 = image.shape[:2]
    scale = 1.0
    if max(h0, w0) > resize_max_side:
        scale = resize_max_side / float(max(h0, w0))
    
    small = cv2.resize(image, (int(w0 * scale), int(h0 * scale)), interpolation=cv2.INTER_AREA)

    # --- FIX: Check if image is already grayscale ---
    if len(small.shape) == 3 and small.shape[2] == 3:
        # It's a color image, so convert it
        gray = cv2.cvtColor(small, cv2.COLOR_BGR2GRAY)
    else:
        # It's already grayscale, use it directly
        gray = small
    # --- END of FIX ---

    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    edges = cv2.Canny(blur, 50, 150, apertureSize=3)

    lines = cv2.HoughLinesP(
        edges,
        rho=1,
        theta=np.pi/180,
        threshold=80,
        minLineLength=int(min(small.shape[:2]) * 0.3),
        maxLineGap=20
    )

    if lines is None or len(lines) == 0:
        cnt = largest_contour_from_thresh(gray)
        if cnt is None:
            return 0.0
        rect = cv2.minAreaRect(cnt)
        angle = rect[2]
        w_rect, h_rect = rect[1]
        if w_rect < h_rect:
            angle = angle + 90.0
        return float(angle)
    
    angles = []
    for l in lines:
        x1, y1, x2, y2 = l[0]
        ang = np.degrees(np.arctan2((y2 - y1), (x2 - x1)))
        if ang < -90:
            ang += 180
        if ang > 90:
            ang -= 180
        if ang < -45:
            ang += 90
        elif ang > 45:
            ang -= 90
        angles.append(ang)

    if len(angles) == 0:
        return 0.0
    
    median_angle = float(np.median(np.array(angles)))
    return median_angle

# Function to find the main document/page
def find_document_polygon(image: np.ndarray) -> Optional[np.ndarray]:
    """Finds the polygon representing the main document in the image."""
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    # Use adaptive thresholding to be more robust to lighting
    thresh = cv2.adaptiveThreshold(blurred, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                   cv2.THRESH_BINARY_INV, 11, 2)
    
    # Try closing operation to connect broken lines if any
    kernel = np.ones((5,5), np.uint8)
    thresh = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)

    # Find the largest contour
    cnts = cv2.findContours(thresh.copy(), cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    cnts = cnts[0] if len(cnts) == 2 else cnts[1] # For different OpenCV versions

    if not cnts:
        return None

    doc_contour = None
    max_area = 0
    img_area = image.shape[0] * image.shape[1]

    for c in cnts:
        area = cv2.contourArea(c)
        # Filter for reasonably sized contours (e.g., > 10% of image area)
        # and ignore very small specks or very large contours that might be image border
        if area > img_area * 0.1 and area < img_area * 0.95:
            peri = cv2.arcLength(c, True)
            approx = cv2.approxPolyDP(c, 0.02 * peri, True)
            # We are looking for a quadrilateral (4 corners)
            if len(approx) == 4:
                # Check aspect ratio (paper like) and solidity
                x, y, w, h = cv2.boundingRect(approx)
                aspect_ratio = float(w) / h
                if 0.5 < aspect_ratio < 2.0: # Typical document aspect ratio
                    solidity = float(area) / (w * h)
                    if solidity > 0.7: # How "solid" the contour is (avoid C, L shapes)
                        if area > max_area:
                            max_area = area
                            doc_contour = approx
    
    if doc_contour is not None:
        # Reshape to 2D array of points
        return doc_contour.reshape(4, 2)
    
    return None

# --- Main processing function ---
def process_document_image(image: np.ndarray, padding: int = 20) -> Tuple[float, np.ndarray]:
    """
    Processes an image by finding the document, deskewing it, and cropping.
    Returns the applied rotation angle and the processed image.
    """
    
    # 1. Find the document (white page) within the image
    document_polygon = find_document_polygon(image.copy())

    deskewed_page = image # Fallback: if no document found, use original image

    if document_polygon is not None:
        # 2. Straighten and crop the document/page using perspective transform
        # This will give us a perfectly rectangular document, removing the outer black area
        deskewed_page = four_point_transform(image, document_polygon)
    
    # Now, 'deskewed_page' is a straight rectangle.
    # We proceed to deskew the *content* within this page.

    # Convert to grayscale for skew estimation, which operates on single channel
    gray_page = cv2.cvtColor(deskewed_page, cv2.COLOR_BGR2GRAY)

    # 3. Estimate the skew angle of the content within this straightened page
    angle = estimate_skew_angle(gray_page) # Pass the gray version of the deskewed_page

    # 4. Perform the rotation on the straightened page
    h, w = deskewed_page.shape[:2]
    center = (w // 2, h // 2)
    M = cv2.getRotationMatrix2D(center, angle, 1.0) # Rotate by the estimated angle
    
    # We want to rotate the entire deskewed_page, keeping all content
    rotated = cv2.warpAffine(deskewed_page, M, (w, h), flags=cv2.INTER_CUBIC,
                             borderMode=cv2.BORDER_CONSTANT, borderValue=(0,0,0)) # Fill with black

    # 5. Crop the largest content area from the rotated image
    # Find contours again on the rotated image to find the new bounds of the content
    gray_rotated = cv2.cvtColor(rotated, cv2.COLOR_BGR2GRAY)
    # Threshold for content (assuming content is generally darker than white page)
    # Using OTSU's method to find optimal threshold
    _, thresh_rotated = cv2.threshold(gray_rotated, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    
    # Find contours in the thresholded image
    cnts_rotated = cv2.findContours(thresh_rotated, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cnts_rotated = cnts_rotated[0] if len(cnts_rotated) == 2 else cnts_rotated[1]
    
    if cnts_rotated:
        max_contour = max(cnts_rotated, key=cv2.contourArea)
        x, y, w_c, h_c = cv2.boundingRect(max_contour)
        
        # Apply padding to the bounding box, clamping to image dimensions
        x = max(0, x - padding)
        y = max(0, y - padding)
        w_c = min(rotated.shape[1] - x, w_c + 2 * padding)
        h_c = min(rotated.shape[0] - y, h_c + 2 * padding)
        
        cropped_image = rotated[y:y+h_c, x:x+w_c]
    else:
        # Fallback if no content contour found after rotation (shouldn't happen often)
        cropped_image = rotated


    return angle, cropped_image