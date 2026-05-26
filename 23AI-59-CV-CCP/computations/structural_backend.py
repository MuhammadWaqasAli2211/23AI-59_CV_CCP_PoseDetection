# computations/structural_backend.py
import numpy as np

def calculate_angle(a, b, c):
    """
    Computes the angle ABC at joint vertex 'b' using the vector dot product.
    Formula: theta = arccos( (BA . BC) / (||BA|| * ||BC||) )
    """
    a = np.array(a)  # Point A (e.g., Hip)
    b = np.array(b)  # Point B (Vertex, e.g., Knee)
    c = np.array(c)  # Point C (e.g., Ankle)
    
    # Define directional vectors branching from the joint vertex
    vector_ba = a - b
    vector_bc = c - b
    
    # Calculate the cosine of the angle with floating-point safety clipping
    dot_product = np.dot(vector_ba, vector_bc)
    magnitude_product = np.linalg.norm(vector_ba) * np.linalg.norm(vector_bc)
    
    # Avoid division by zero if keypoints overlap
    if magnitude_product == 0:
        return 0.0
        
    cosine_angle = np.clip(dot_product / magnitude_product, -1.0, 1.0)
    angle_radians = np.arccos(cosine_angle)
    
    return np.degrees(angle_radians)

def apply_ema_filter(current_coord, previous_coord, alpha=0.35):
    """
    Applies an Exponential Moving Average (EMA) filter to individual landmark coordinates
    across successive frames to eliminate high-frequency spatial tracking jitter.
    Formula: S_t = alpha * X_t + (1 - alpha) * S_{t-1}
    """
    if previous_coord is None:
        return current_coord
    return (alpha * np.array(current_coord)) + ((1 - alpha) * np.array(previous_coord))