import cv2
import numpy as np
import os
from PIL import Image


def find_object_in_scene(template_path, scene_path, min_matches=4, debug=True):
    """
    Find the template object in the scene using multiple OpenCV methods.
    Optimized for low memory usage.
    """
    print(f"\n{'='*60}")
    print(f"[INFO] Looking for your lost object...")
    
    if not os.path.exists(template_path):
        return None, "Template file not found"
    if not os.path.exists(scene_path):
        return None, "Scene file not found"
    
    template = cv2.imread(template_path)
    scene = cv2.imread(scene_path)
    
    if template is None or scene is None:
        return None, "Could not read images"
    
    # Resize if images are too large (save memory)
    max_dim = 800
    template = resize_if_needed(template, max_dim)
    scene = resize_if_needed(scene, max_dim)
    
    print(f"[INFO] Template size: {template.shape[:2]}")
    print(f"[INFO] Scene size: {scene.shape[:2]}")
    
    # Method 1: Multi-scale template matching
    result, msg = multi_scale_template_match(template, scene)
    if result is not None:
        return result, msg
    
    # Method 2: Feature matching (ORB - no extra dependencies)
    result, msg = orb_feature_match(template, scene)
    if result is not None:
        return result, msg
    
    # Method 3: Color-based search
    result, msg = color_histogram_search(template, scene)
    if result is not None:
        return result, msg
    
    return None, "Object not found. Try a different image or angle."


def resize_if_needed(img, max_dim):
    """Resize image if larger than max_dim to save memory."""
    h, w = img.shape[:2]
    if max(h, w) > max_dim:
        scale = max_dim / max(h, w)
        new_w = int(w * scale)
        new_h = int(h * scale)
        return cv2.resize(img, (new_w, new_h))
    return img


def multi_scale_template_match(template, scene):
    """Multi-scale template matching."""
    t_gray = cv2.cvtColor(template, cv2.COLOR_BGR2GRAY)
    s_gray = cv2.cvtColor(scene, cv2.COLOR_BGR2GRAY)
    
    best_val = 0
    best_loc = None
    best_size = None
    
    # Try different scales
    for scale in np.linspace(0.2, 2.0, 20):
        new_w = int(template.shape[1] * scale)
        new_h = int(template.shape[0] * scale)
        
        if new_w < 20 or new_h < 20:
            continue
        if new_w >= scene.shape[1] or new_h >= scene.shape[0]:
            continue
        
        resized = cv2.resize(t_gray, (new_w, new_h))
        
        try:
            result = cv2.matchTemplate(s_gray, resized, cv2.TM_CCOEFF_NORMED)
            _, max_val, _, max_loc = cv2.minMaxLoc(result)
            
            if max_val > best_val:
                best_val = max_val
                best_loc = max_loc
                best_size = (new_w, new_h)
        except:
            continue
    
    print(f"[INFO] Template match score: {best_val:.2f}")
    
    if best_val > 0.5 and best_loc:
        out = scene.copy()
        x, y = best_loc
        w, h = best_size
        draw_result(out, x, y, w, h, best_val)
        return out, f"Object found! Confidence: {best_val*100:.0f}%"
    
    return None, f"Template match: {best_val*100:.0f}%"


def orb_feature_match(template, scene):
    """ORB feature matching."""
    t_gray = cv2.cvtColor(template, cv2.COLOR_BGR2GRAY)
    s_gray = cv2.cvtColor(scene, cv2.COLOR_BGR2GRAY)
    
    orb = cv2.ORB_create(nfeatures=1000)
    kp1, des1 = orb.detectAndCompute(t_gray, None)
    kp2, des2 = orb.detectAndCompute(s_gray, None)
    
    if des1 is None or des2 is None:
        return None, "Could not detect features"
    
    if len(kp1) < 4 or len(kp2) < 4:
        return None, "Not enough features"
    
    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=False)
    try:
        matches = bf.knnMatch(des1, des2, k=2)
    except:
        return None, "Matching failed"
    
    good = []
    for m_n in matches:
        if len(m_n) == 2:
            m, n = m_n
            if m.distance < 0.75 * n.distance:
                good.append(m)
    
    print(f"[INFO] ORB matches: {len(good)}")
    
    if len(good) >= 8:
        src_pts = np.float32([kp1[m.queryIdx].pt for m in good]).reshape(-1, 1, 2)
        dst_pts = np.float32([kp2[m.trainIdx].pt for m in good]).reshape(-1, 1, 2)
        
        M, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)
        
        if M is not None:
            h, w = t_gray.shape
            pts = np.float32([[0, 0], [0, h-1], [w-1, h-1], [w-1, 0]]).reshape(-1, 1, 2)
            
            try:
                dst = cv2.perspectiveTransform(pts, M)
                area = cv2.contourArea(dst)
                
                if 100 < area < scene.shape[0] * scene.shape[1] * 0.8:
                    out = scene.copy()
                    cv2.polylines(out, [np.int32(dst)], True, (0, 255, 0), 3)
                    
                    center = np.mean(dst, axis=0).astype(int)[0]
                    cv2.circle(out, tuple(center), 12, (0, 0, 255), -1)
                    
                    # Add label
                    cv2.putText(out, f"FOUND ({len(good)} matches)", 
                               (int(dst[0][0][0]), int(dst[0][0][1]) - 10),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
                    
                    return out, f"Object found! {len(good)} feature matches"
            except:
                pass
    
    return None, f"Feature matching: {len(good)} matches"


def color_histogram_search(template, scene):
    """Search using color histogram."""
    t_hsv = cv2.cvtColor(template, cv2.COLOR_BGR2HSV)
    s_hsv = cv2.cvtColor(scene, cv2.COLOR_BGR2HSV)
    
    t_hist = cv2.calcHist([t_hsv], [0, 1], None, [30, 32], [0, 180, 0, 256])
    cv2.normalize(t_hist, t_hist, 0, 1, cv2.NORM_MINMAX)
    
    th, tw = template.shape[:2]
    sh, sw = scene.shape[:2]
    
    best_score = 0
    best_loc = None
    best_size = None
    
    step = 30
    
    for scale in [0.5, 0.75, 1.0, 1.25, 1.5]:
        win_w = int(tw * scale)
        win_h = int(th * scale)
        
        if win_w >= sw or win_h >= sh or win_w < 30 or win_h < 30:
            continue
        
        for y in range(0, sh - win_h, step):
            for x in range(0, sw - win_w, step):
                roi = s_hsv[y:y+win_h, x:x+win_w]
                roi_hist = cv2.calcHist([roi], [0, 1], None, [30, 32], [0, 180, 0, 256])
                cv2.normalize(roi_hist, roi_hist, 0, 1, cv2.NORM_MINMAX)
                
                score = cv2.compareHist(t_hist, roi_hist, cv2.HISTCMP_CORREL)
                
                if score > best_score:
                    best_score = score
                    best_loc = (x, y)
                    best_size = (win_w, win_h)
    
    print(f"[INFO] Color match score: {best_score:.2f}")
    
    if best_score > 0.5 and best_loc:
        out = scene.copy()
        x, y = best_loc
        w, h = best_size
        draw_result(out, x, y, w, h, best_score)
        return out, f"Object found by color! Confidence: {best_score*100:.0f}%"
    
    return None, f"Color match: {best_score*100:.0f}%"


def draw_result(image, x, y, w, h, confidence):
    """Draw detection result on image."""
    # Main rectangle
    cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 3)
    
    # Corner accents
    corner_len = min(w, h) // 4
    color = (0, 255, 0)
    thickness = 4
    
    cv2.line(image, (x, y), (x + corner_len, y), color, thickness)
    cv2.line(image, (x, y), (x, y + corner_len), color, thickness)
    cv2.line(image, (x + w, y), (x + w - corner_len, y), color, thickness)
    cv2.line(image, (x + w, y), (x + w, y + corner_len), color, thickness)
    cv2.line(image, (x, y + h), (x + corner_len, y + h), color, thickness)
    cv2.line(image, (x, y + h), (x, y + h - corner_len), color, thickness)
    cv2.line(image, (x + w, y + h), (x + w - corner_len, y + h), color, thickness)
    cv2.line(image, (x + w, y + h), (x + w, y + h - corner_len), color, thickness)
    
    # Center point
    cx, cy = x + w // 2, y + h // 2
    cv2.circle(image, (cx, cy), 12, (0, 0, 255), -1)
    
    # Label
    label = f"FOUND {confidence*100:.0f}%"
    label_y = y - 15 if y > 40 else y + h + 25
    cv2.rectangle(image, (x - 2, label_y - 20), (x + 150, label_y + 5), (0, 255, 0), -1)
    cv2.putText(image, label, (x, label_y), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2)
