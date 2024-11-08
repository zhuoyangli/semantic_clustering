import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

# Load GloVe embeddings function (same as before)
def load_glove_embeddings(filepath):
    embeddings = {}
    with open(filepath, 'r', encoding='utf-8') as f:
        for line in f:
            values = line.split()
            word = values[0]
            vector = np.array(values[1:], dtype='float32')
            embeddings[word] = vector
    return embeddings

# Cosine distance function (same as before)
def cosine_distance(vec1, vec2):
    return 1 - cosine_similarity([vec1], [vec2])[0][0]

def compute_exponentially_weighted_forward_flow(word_sequence, glove_embeddings, decay_factor=0.5):
    # Step 1: Get embeddings
    vectors = []
    for word in word_sequence:
        if word in glove_embeddings:
            vectors.append(glove_embeddings[word])
        else:
            print(f"Word '{word}' not found in GloVe embeddings.")
            return None, None
            
    n = len(vectors)
    forward_flow_scores = []
    boundaries = []
    
    # FIRST PASS: Original exponential forward flow
    start_index = 0
    for i in range(1, n):
        weighted_sum = 0
        weight_total = 0
        
        # Calculate forward flow for each word in current segment
        for j in range(start_index, i):
            distance = cosine_distance(vectors[i], vectors[j])
            weight = decay_factor ** (i - j - 1)
            weighted_sum += weight * distance
            weight_total += weight
        
        forward_flow_score = weighted_sum / weight_total
        forward_flow_scores.append(forward_flow_score)
        
        if forward_flow_score > 0.7:
            boundaries.append(i)
            start_index = i
            
    # SECOND PASS: Boundary refinement using segment coherence
    if len(boundaries) > 0:
        boundaries = refine_boundaries(vectors, boundaries)
    
    return forward_flow_scores, [word_sequence[i] for i in boundaries]

def refine_boundaries(vectors, initial_boundaries, window_size=2):
    """
    Refine boundary positions using segment coherence scores
    """
    boundaries = initial_boundaries.copy()
    n = len(vectors)
    
    # Go through each boundary in order
    for idx in range(len(boundaries)):
        current_b = boundaries[idx]
        best_score = float('-inf')
        best_pos = current_b
        
        # Test positions within window
        for test_pos in range(max(1, current_b - window_size), 
                            min(n-1, current_b + window_size + 1)):
            
            # Skip if position would create invalid boundary ordering
            if idx > 0 and test_pos <= boundaries[idx-1]:
                continue
            if idx < len(boundaries)-1 and test_pos >= boundaries[idx+1]:
                continue
            
            # Temporarily move boundary
            old_pos = boundaries[idx]
            boundaries[idx] = test_pos
            
            # Score the segments before and after this boundary
            score = score_segments(vectors, boundaries)
            
            if score > best_score:
                best_score = score
                best_pos = test_pos
                
            # Reset boundary
            boundaries[idx] = old_pos
            
        # Apply best position found
        boundaries[idx] = best_pos
        
    return boundaries

def score_segments(vectors, boundaries):
    """
    Score segmentation based on within-segment similarity and
    between-segment dissimilarity
    """
    segments = get_segments(len(vectors), boundaries)
    total_score = 0
    
    # Score each segment
    for segment in segments:
        if len(segment) < 2:
            continue
            
        # Get mean vector for segment
        segment_vectors = [vectors[i] for i in segment]
        mean_vec = np.mean(segment_vectors, axis=0)
        
        # Add within-segment cohesion score
        within_scores = [cosine_similarity([mean_vec], [v])[0][0] 
                        for v in segment_vectors]
        total_score += np.mean(within_scores)
    
    # Penalize similarity between adjacent segments
    for i in range(len(segments)-1):
        seg1_mean = np.mean([vectors[j] for j in segments[i]], axis=0)
        seg2_mean = np.mean([vectors[j] for j in segments[i+1]], axis=0)
        between_score = cosine_similarity([seg1_mean], [seg2_mean])[0][0]
        total_score -= between_score
        
    return total_score

def get_segments(n, boundaries):
    """Get list of segments given boundaries"""
    segments = []
    start = 0
    
    for b in sorted(boundaries):
        segments.append(range(start, b))
        start = b
    segments.append(range(start, n))
    
    return segments



# Example usage
if __name__ == "__main__":
    
    glove_path = '/Users/marcoazar/Desktop/Cognitive Research/semantic_clustering/Marco code/glove.6B.300d.txt'
    glove_vectors = load_glove_embeddings(glove_path)

    # Example word sequence
    word_chain = ["sea", "ship", "arm", "legs", "feet", "thighs", "hands", "throw", "catch", "drink", "juice", "apple"]
    
    word_chain2 = ["water", "river", "flow", "current", "electricity", "power", "strength", "muscle", 
                  "exercise", "gym", "fitness", "health", "doctor", "hospital", "patient", "sick", 
                  "virus", "morning", "global", "world", "earth", "planet", "space", "star"]
    
    word_chain3 = ["candy", "sweet", "food", "eat", "chocolate"]

    # Compute exponentially weighted forward flow scores
    decay_factor = 0.5  # Adjust decay factor as needed
    forward_flow_scores, boundaries = compute_exponentially_weighted_forward_flow(word_chain2, glove_vectors, decay_factor)

    if forward_flow_scores is not None:
        for i, score in enumerate(forward_flow_scores, start=1):
            print(f"Forward Flow Score for word {word_chain2[i]}: {score:.4f}")
        
        # Print boundary words
        print("\nBoundary words where forward flow score > 0.7:")
        print(boundaries)
