import cv2
import numpy as np
import os
from PIL import Image
import torch
import torchvision.transforms as transforms
from torchvision import models

# Global feature extractor
feature_model = None
transform = None

def load_feature_extractor():
    """Load a pre-trained CNN for feature extraction."""
    global feature_model, transform
    if feature_model is None:
        print("[INFO] Loading feature extraction model...")
        # Use ResNet18 - lightweight but effective
        feature_model = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
        # Remove the final classification layer to get features
        feature_model = torch.nn.Sequential(*list(feature_model.children())[:-1])
        feature_model.eval()
        
        transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        print("[INFO] Feature extraction model loaded!")
    return feature_model, transform


def extract_features(image, model, transform):
    """Extract CNN features from an image."""
    if isinstance(image, np.ndarray):
        image = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    
    img_tensor = transform(image).unsqueeze(0)
    
    with torch.no_grad():
        features = model(img_tensor)
    
    return features.flatten().numpy()


def find_object_in_scene(template_path, scene_path, min_matches=4, debug=True):
    """
    Find the template object in the scene using sliding window + CNN similarity.
    """
    print(f"\n{'='*60}")
    print(f"[INFO] Looking for your lost object...")
    print(f"[INFO] Template: {template_path}")
    print(f"[INFO] Scene: {scene_path}")
    
    if not os.path.exists(template_path):
        return None, "Template file not found"
    if not os.path.exists(scene_path):
        return None, "Scene file not found"
    
    template = cv2.imread(template_path)
    scene = cv2.imread(scene_path)
    
    if template is None or scene is None:
        return None, "Could not read images"
    
    print(f"[INFO] Template size: {template.shape[:2]}")
    print(f"[INFO] Scene size: {scene.shape[:2]}")
    
    # Load the feature extractor
    model, trans = load_feature_extractor()
    
    # Extract template features
    print("[INFO] Analyzing your lost object...")
    template_features = extract_features(template, model, trans)
    
    # Search using sliding window at multiple scales
    print("[INFO] Searching the scene...")
    best_match = sliding_window_search(template, scene, template_features, model, trans)
    
    if best_match is not None:
        x, y, w, h, similarity = best_match
        out = scene.copy()
        draw_result(out, x, y, w, h, similarity)
        
        if similarity > 0.7:
            return out, f"🎉 Found your object! Similarity: {similarity*100:.0f}%"
        elif similarity > 0.5:
            return out, f"Possible match found. Similarity: {similarity*100:.0f}%"
        else:
            return out, f"Best guess location. Similarity: {similarity*100:.0f}%"
    
    return None, "Could not locate the object. Try a different scene image."


def sliding_window_search(template, scene, template_features, model, trans):
    """
    Search for template in scene using sliding windows at multiple scales.
    """
    th, tw = template.shape[:2]
    sh, sw = scene.shape[:2]
    
    # Calculate aspect ratio of template
    aspect_ratio = tw / th
    
    best_similarity = 0
    best_match = None
    
    # Try multiple scales relative to scene size
    min_size = 50
    max_size = min(sh, sw) - 20
    
    # Generate window sizes based on template aspect ratio
    sizes = []
    for size in range(min_size, max_size, max(20, (max_size - min_size) // 15)):
        win_h = size
        win_w = int(size * aspect_ratio)
        if win_w < min_size:
            win_w = min_size
            win_h = int(win_w / aspect_ratio)
        if win_w < sw and win_h < sh:
            sizes.append((win_w, win_h))
    
    # Also add sizes based on original template dimensions
    for scale in [0.25, 0.5, 0.75, 1.0, 1.25, 1.5, 2.0, 2.5, 3.0]:
        win_w = int(tw * scale)
        win_h = int(th * scale)
        if min_size <= win_w < sw and min_size <= win_h < sh:
            sizes.append((win_w, win_h))
    
    # Remove duplicates and sort
    sizes = list(set(sizes))
    
    total_windows = 0
    step_factor = 0.25  # Overlap factor
    
    print(f"[INFO] Checking {len(sizes)} different scales...")
    
    for win_w, win_h in sizes:
        step_x = max(10, int(win_w * step_factor))
        step_y = max(10, int(win_h * step_factor))
        
        for y in range(0, sh - win_h, step_y):
            for x in range(0, sw - win_w, step_x):
                total_windows += 1
                
                # Extract window
                window = scene[y:y+win_h, x:x+win_w]
                
                # Quick pre-filter: check if colors are similar enough
                if not quick_color_check(template, window):
                    continue
                
                # Extract features
                window_features = extract_features(window, model, trans)
                
                # Calculate cosine similarity
                similarity = cosine_similarity(template_features, window_features)
                
                if similarity > best_similarity:
                    best_similarity = similarity
                    best_match = (x, y, win_w, win_h, similarity)
                    print(f"  [UPDATE] New best match: {similarity:.2%} at ({x}, {y})")
    
    print(f"[INFO] Checked {total_windows} windows")
    print(f"[INFO] Best similarity: {best_similarity:.2%}")
    
    if best_similarity > 0.3:  # Lower threshold to find something
        return best_match
    
    return None


def quick_color_check(template, window, threshold=0.3):
    """Quick color histogram comparison to filter obviously wrong windows."""
    t_hsv = cv2.cvtColor(cv2.resize(template, (32, 32)), cv2.COLOR_BGR2HSV)
    w_hsv = cv2.cvtColor(cv2.resize(window, (32, 32)), cv2.COLOR_BGR2HSV)
    
    t_hist = cv2.calcHist([t_hsv], [0, 1], None, [12, 12], [0, 180, 0, 256])
    w_hist = cv2.calcHist([w_hsv], [0, 1], None, [12, 12], [0, 180, 0, 256])
    
    cv2.normalize(t_hist, t_hist)
    cv2.normalize(w_hist, w_hist)
    
    score = cv2.compareHist(t_hist, w_hist, cv2.HISTCMP_CORREL)
    return score > threshold


def cosine_similarity(a, b):
    """Calculate cosine similarity between two feature vectors."""
    dot = np.dot(a, b)
    norm_a = np.linalg.norm(a)
    norm_b = np.linalg.norm(b)
    if norm_a == 0 or norm_b == 0:
        return 0
    return dot / (norm_a * norm_b)


def draw_result(image, x, y, w, h, similarity):
    """Draw the detection result on the image."""
    # Main rectangle
    color = (0, 255, 0) if similarity > 0.6 else (0, 255, 255)
    cv2.rectangle(image, (x, y), (x + w, y + h), color, 4)
    
    # Corner accents
    corner_len = min(w, h) // 4
    thickness = 5
    
    # Top-left
    cv2.line(image, (x, y), (x + corner_len, y), color, thickness)
    cv2.line(image, (x, y), (x, y + corner_len), color, thickness)
    # Top-right
    cv2.line(image, (x + w, y), (x + w - corner_len, y), color, thickness)
    cv2.line(image, (x + w, y), (x + w, y + corner_len), color, thickness)
    # Bottom-left
    cv2.line(image, (x, y + h), (x + corner_len, y + h), color, thickness)
    cv2.line(image, (x, y + h), (x, y + h - corner_len), color, thickness)
    # Bottom-right
    cv2.line(image, (x + w, y + h), (x + w - corner_len, y + h), color, thickness)
    cv2.line(image, (x + w, y + h), (x + w, y + h - corner_len), color, thickness)
    
    # Center crosshair
    cx, cy = x + w // 2, y + h // 2
    cv2.circle(image, (cx, cy), 15, (0, 0, 255), -1)
    cv2.circle(image, (cx, cy), 25, (0, 0, 255), 3)
    cv2.line(image, (cx - 35, cy), (cx + 35, cy), (0, 0, 255), 2)
    cv2.line(image, (cx, cy - 35), (cx, cy + 35), (0, 0, 255), 2)
    
    # Label with background
    label = f"FOUND HERE! {similarity*100:.0f}%"
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 1.0
    font_thickness = 2
    
    (text_w, text_h), _ = cv2.getTextSize(label, font, font_scale, font_thickness)
    
    label_y = y - 20 if y > 60 else y + h + 40
    
    # Background
    cv2.rectangle(image, (x - 5, label_y - text_h - 10), 
                  (x + text_w + 10, label_y + 10), color, -1)
    cv2.putText(image, label, (x, label_y), font, font_scale, (0, 0, 0), font_thickness)
    
    # Add arrow pointing to center
    arrow_start = (cx, y - 50 if y > 100 else y + h + 50)
    arrow_end = (cx, cy)
    cv2.arrowedLine(image, arrow_start, arrow_end, (0, 0, 255), 3, tipLength=0.3)
