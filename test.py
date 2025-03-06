import videohash
from pathlib import Path
import sys
from PIL import Image
import numpy as np
import faiss
from typing import Tuple, List
import time

# Compatibility fix for different Pillow versions
try:
    ANTIALIAS = Image.ANTIALIAS
except AttributeError:
    ANTIALIAS = Image.Resampling.LANCZOS

def hex_to_binary_vector(hex_hash: str) -> np.ndarray:
    """Convert hex hash to binary vector"""
    bin_str = bin(int(hex_hash, 16))[2:].zfill(64)
    return np.array([int(b) for b in bin_str], dtype=np.float32)

def calculate_video_hash(video_path: str) -> Tuple[str, np.ndarray]:
    """Calculate video hash and return both hex and vector formats"""
    try:
        Image.ANTIALIAS = ANTIALIAS
        vh = videohash.VideoHash(path=video_path)
        hex_hash = vh.hash_hex
        vector = hex_to_binary_vector(hex_hash)
        return hex_hash, vector
    except Exception as e:
        print(f"Error processing video {video_path}: {str(e)}")
        return None, None

def hamming_distance(hash1: str, hash2: str) -> int:
    """Calculate Hamming distance between two hex hashes"""
    bin1 = bin(int(hash1, 16))[2:].zfill(64)
    bin2 = bin(int(hash2, 16))[2:].zfill(64)
    return sum(b1 != b2 for b1, b2 in zip(bin1, bin2))

def create_faiss_index(vectors: List[np.ndarray]) -> faiss.IndexFlatIP:
    """Create and populate FAISS index for cosine similarity search"""
    dimension = 64  # VideoHash produces 64-bit hashes
    index = faiss.IndexFlatIP(dimension)
    
    # Normalize vectors for cosine similarity
    vectors = np.array(vectors, dtype=np.float32)
    faiss.normalize_L2(vectors)
    index.add(vectors)
    return index

def compare_videos(video_path1: str, video_path2: str) -> Tuple[float, float, float, float]:
    """
    Compare videos using both Hamming distance and FAISS cosine similarity
    Returns: (hamming_similarity, cosine_similarity, hamming_time, cosine_time)
    """
    # Calculate hashes and vectors
    hex1, vec1 = calculate_video_hash(video_path1)
    hex2, vec2 = calculate_video_hash(video_path2)
    
    if hex1 is None or hex2 is None:
        return None, None, None, None
    
    # Time Hamming distance similarity
    hamming_start = time.time()
    distance = hamming_distance(hex1, hex2)
    hamming_similarity = 1 - (distance / 64)
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
    
    # Compare videos using both methods
    hamming_sim, cosine_sim, hamming_time, cosine_time = compare_videos(
        str(video_path1), str(video_path2)
    )
    
    if hamming_sim is not None and cosine_sim is not None:
        print(f"\nVideo Comparison Results:")
        print(f"Video 1: {video_path1}")
        print(f"Video 2: {video_path2}")
        print("\nHamming Distance Method:")
        print(f"Similarity score: {hamming_sim:.4f}")
        print(f"Hamming distance: {int((1 - hamming_sim) * 64)}")
        print(f"Videos are {'similar' if hamming_sim >= 0.85 else 'different'}")
        print(f"Time taken: {hamming_time*1000:.2f} ms")
        print("\nCosine Similarity Method:")
        print(f"Similarity score: {cosine_sim:.4f}")
        print(f"Videos are {'similar' if cosine_sim >= 0.85 else 'different'}")
        print(f"Time taken: {cosine_time*1000:.2f} ms")
        
        # Print performance comparison
        print("\nPerformance Comparison:")
        print(f"Hamming Distance was {(cosine_time/hamming_time):.1f}x faster than Cosine Similarity")
    else:
        print("Failed to compare videos")

if __name__ == "__main__":
    main()
