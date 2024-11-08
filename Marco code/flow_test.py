import numpy as np

def load_glove_vectors(file_path):
    print("Loading GloVe vectors...")
    with open(file_path, 'r', encoding='utf-8') as f:
        return {line.split()[0]: np.array(line.split()[1:], dtype=float) for line in f}

def get_word_embedding(word, glove_vectors):
    return glove_vectors.get(word.lower(), None)

def cosine_similarity(vec1, vec2):
    dot_product = np.dot(vec1, vec2)
    norm_a = np.linalg.norm(vec1)
    norm_b = np.linalg.norm(vec2)
    return dot_product / (norm_a * norm_b)

def compute_weighted_distance(current_word, context_words, glove_vectors, exp_base=2.0):
    """
    Compute weighted average distance between current word and all context words,
    with exponential weighting giving more importance to recent words.
    """
    if not context_words:
        return None
    
    current_vec = get_word_embedding(current_word, glove_vectors)
    if current_vec is None:
        return None

    distances = []
    weights = []
    
    # Process context words in reverse order (most recent first)
    for i, word in enumerate(reversed(context_words)):
        context_vec = get_word_embedding(word, glove_vectors)
        if context_vec is not None:
            similarity = cosine_similarity(current_vec, context_vec)
            distance = 1 - similarity
            # Exponential weighting: more recent words get higher weights
            weight = exp_base ** (2*i) 
            distances.append(distance)
            weights.append(weight)
    
    if not distances:
        return None
        
    # Normalize weights
    weights = np.array(weights)
    weights = weights / np.sum(weights)
    
    return np.average(distances, weights=weights)

def find_last_boundary(boundaries, current_idx):
    """Find the index of the last boundary before current position."""
    for i in range(current_idx - 1, -1, -1):
        if boundaries[i] > 0:
            return i + 1
    return 0

def forward_flow(word_list, glove_vectors):
    boundaries = [0] * len(word_list)
    
    for i in range(len(word_list)-1):  # Process all words except last
        current_word = word_list[i]
        next_word = word_list[i+1]
        
        # Find last boundary and get context
        last_boundary_idx = find_last_boundary(boundaries, i)
        context = word_list[last_boundary_idx:i+1]  # Include current word in context
        
        # Get weighted distance to context
        context_distance = compute_weighted_distance(next_word, context, glove_vectors) or 0
        
        # Get direct distance to previous word
        current_vec = get_word_embedding(current_word, glove_vectors)
        next_vec = get_word_embedding(next_word, glove_vectors)
        
        if current_vec is not None and next_vec is not None:
            direct_similarity = cosine_similarity(current_vec, next_vec)
            direct_distance = 1 - direct_similarity
            
            # Combine direct distance and context distance with more weight on direct distance
            combined_distance = 0.7 * direct_distance + 0.3 * context_distance
            
            # Boundary detection with adjusted thresholds
            if combined_distance > 0.75:  # Strong boundary
                boundaries[i] = 2
            elif combined_distance > 0.65:  # Weak boundary
                boundaries[i] = 1
                
            # Additional check for semantic coherence break
            if i > 0 and direct_distance > 0.8:
                boundaries[i] = 2
    
    return boundaries

def process_word_chain(word_chain, glove_vectors):
    boundaries = forward_flow(word_chain, glove_vectors)
    
    print("\nBoundary Analysis:")
    for word, boundary in zip(word_chain, boundaries):
        print(f"{word}: {boundary}")
    
    print("\nResult =", boundaries)

if __name__ == "__main__":
    try:
        glove_path = '/Users/marcoazar/Desktop/Cognitive Research/semantic_clustering/Marco code/glove.6B.300d.txt'
        glove_vectors = load_glove_vectors(glove_path)
        
        word_chain = ["sea", "ship", "arm", "legs", "feet", "thighs", "hands", "throw", "catch", "drink", "juice", "apple"]
        
        process_word_chain(word_chain, glove_vectors)
        
    except FileNotFoundError:
        print(f"Error: GloVe file not found at {glove_path}")
    except Exception as e:
        print(f"An error occurred: {str(e)}")