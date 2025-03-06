import cv2
from pathlib import Path
import sys
from PIL import Image
import numpy as np
import faiss
from typing import Tuple, List, Optional
import time

def extract_frames(video_path: str, num_frames: int = 10) -> List[np.ndarray]:
    """Extract equidistant frames from video"""
    cap = cv2.VideoCapture(video_path)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    if total_frames == 0:
        return []
    
    # Calculate frame intervals
    interval = total_frames // num_frames
    frames = []
    
    for i in range(num_frames):
        cap.set(cv2.CAP_PROP_POS_FRAMES, i * interval)
        ret, frame = cap.read()
        if ret:
            # Convert BGR to RGB
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frames.append(frame_rgb)
    
    cap.release()
    return frames

def calculate_phash(image: Image.Image) -> str:
    """Calculate perceptual hash of an image"""
    # Resize to 9x8 and convert to grayscale
    resized = image.resize((9, 8), Image.Resampling.LANCZOS).convert('L')
    pixels = list(resized.getdata())
    
    # Calculate differences
    difference = []
    for row in range(8):
        row_start = row * 9
        for col in range(8):
            difference.append(pixels[row_start + col] > pixels[row_start + col + 1])
    
    # Convert to hex string
    decimal_value = 0
    hash_string = ""
    for idx, value in enumerate(difference):
        if value:
            decimal_value += 2 ** (idx % 8)
        if idx % 8 == 7:
            hash_string += f"{decimal_value:02x}"
            decimal_value = 0
            
    return hash_string

def calculate_frame_hash(frame: np.ndarray) -> Tuple[str, np.ndarray]:
    """Calculate perceptual hash for a single frame"""
    # Convert numpy array to PIL Image
    img = Image.fromarray(frame)
    # Calculate hash
    hash_hex = calculate_phash(img)
    # Convert to binary vector
    hash_bin = bin(int(hash_hex, 16))[2:].zfill(64)
    vector = np.array([int(b) for b in hash_bin], dtype=np.float32)
    return hash_hex, vector

def calculate_video_hashes(video_path: str, num_frames: int = 10) -> Tuple[List[str], np.ndarray]:
    """Calculate dhash for multiple frames and combine them"""
    frames = extract_frames(video_path, num_frames)
    if not frames:
        return [], None
    
    hex_hashes = []
    vectors = []
    
    for frame in frames:
        hex_hash, vector = calculate_frame_hash(frame)
        hex_hashes.append(hex_hash)
        vectors.append(vector)
    
    # Combine vectors into a single feature vector
    combined_vector = np.concatenate(vectors)
    return hex_hashes, combined_vector

def hamming_distance(hashes1: List[str], hashes2: List[str]) -> float:
    """Calculate average Hamming distance between two lists of hashes"""
    if len(hashes1) != len(hashes2):
        return float('inf')
    
    total_distance = 0
    for h1, h2 in zip(hashes1, hashes2):
        bin1 = bin(int(h1, 16))[2:].zfill(64)
        bin2 = bin(int(h2, 16))[2:].zfill(64)
        distance = sum(b1 != b2 for b1, b2 in zip(bin1, bin2))
        total_distance += distance
    
    return total_distance / len(hashes1)

def create_faiss_index(vectors: List[np.ndarray]) -> faiss.IndexFlatIP:
    """Create and populate FAISS index for cosine similarity search"""
    dimension = vectors[0].shape[0]
    index = faiss.IndexFlatIP(dimension)
    vectors = np.array(vectors, dtype=np.float32)
    faiss.normalize_L2(vectors)
    index.add(vectors)
    return index

def compare_videos(video_path1: str, video_path2: str, num_frames: int = 10) -> Tuple[float, float, float, float]:
    """Compare videos using both methods"""
    # Calculate hashes and vectors for both videos
    hashes1, vec1 = calculate_video_hashes(video_path1, num_frames)
    hashes2, vec2 = calculate_video_hashes(video_path2, num_frames)
    
    if not hashes1 or not hashes2:
        return None, None, None, None
    
    # Time Hamming distance similarity
    hamming_start = time.time()
    distance = hamming_distance(hashes1, hashes2)
    hamming_similarity = 1 - (distance / 64)  # Normalize to [0,1]
    hamming_time = time.time() - hamming_start
    
    # Time FAISS cosine similarity
    cosine_start = time.time()
    vectors = np.array([vec1, vec2], dtype=np.float32)
    faiss.normalize_L2(vectors)
    index = create_faiss_index([vec1])
    D, I = index.search(vectors[1:], 1)
    cosine_similarity = float(D[0][0])
    cosine_time = time.time() - cosine_start
    
    return hamming_similarity, cosine_similarity, hamming_time, cosine_time

def main():
    if len(sys.argv) != 3:
        print("Usage: python test.py <video_path1> <video_path2>")
        sys.exit(1)
    
    video_path1 = Path(sys.argv[1])
    video_path2 = Path(sys.argv[2])
    
    if not video_path1.exists() or not video_path2.exists():
        print("One or both video files do not exist!")
        sys.exit(1)
    
    num_frames = 10  # Number of frames to sample from each video
    hamming_sim, cosine_sim, hamming_time, cosine_time = compare_videos(
        str(video_path1), str(video_path2), num_frames
    )
    
    if hamming_sim is not None and cosine_sim is not None:
        print(f"\nVideo Comparison Results (using {num_frames} frames):")
        print(f"Video 1: {video_path1}")
        print(f"Video 2: {video_path2}")
        print("\nHamming Distance Method:")
        print(f"Similarity score: {hamming_sim:.4f}")
        print(f"Average Hamming distance: {int((1 - hamming_sim) * 64)}")
        print(f"Videos are {'similar' if hamming_sim >= 0.85 else 'different'}")
        print(f"Time taken: {hamming_time*1000:.2f} ms")
        print("\nCosine Similarity Method:")
        print(f"Similarity score: {cosine_sim:.4f}")
        print(f"Videos are {'similar' if cosine_sim >= 0.85 else 'different'}")
        print(f"Time taken: {cosine_time*1000:.2f} ms")
        
        print("\nPerformance Comparison:")
        print(f"Hamming Distance was {(cosine_time/hamming_time):.1f}x faster than Cosine Similarity")
    else:
        print("Failed to compare videos")

if __name__ == "__main__":
    main()
