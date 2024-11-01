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

# Compute exponentially weighted forward flow for each word in the sequence and add boundaries
def compute_exponentially_weighted_forward_flow(word_sequence, glove_embeddings, decay_factor=0.5):
    # Step 1: Retrieve embeddings for the word sequence
    vectors = []
    for word in word_sequence:
        if word in glove_embeddings:
            vectors.append(glove_embeddings[word])
        else:
            print(f"Word '{word}' not found in GloVe embeddings.")
            return None, None

    # Step 2: Calculate exponentially weighted forward flow for each word
    n = len(vectors)
    forward_flow_scores = []
    boundaries = []

    for i in range(1, n):  # Start from the second word
        weighted_sum = 0
        weight_total = 0
        
        # Apply exponential decay for each previous word
        for j in range(i):  # For each word before the current word
            # Calculate the semantic distance from current word to word j
            distance = cosine_distance(vectors[i], vectors[j])
            # Compute the weight for this distance based on its position
            weight = decay_factor ** (i - j - 1)  # Decay more for words further back
            # Update weighted sum and total weights
            weighted_sum += weight * distance
            weight_total += weight

        # Calculate weighted forward flow score for current word
        forward_flow_score = weighted_sum / weight_total
        forward_flow_scores.append(forward_flow_score)
        
        # Check if forward flow score exceeds 0.7 to add a boundary
        if forward_flow_score > 0.7:
            boundaries.append(word_sequence[i])

    return forward_flow_scores, boundaries

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
    forward_flow_scores, boundaries = compute_exponentially_weighted_forward_flow(word_chain3, glove_vectors, decay_factor)

    if forward_flow_scores is not None:
        for i, score in enumerate(forward_flow_scores, start=1):
            print(f"Forward Flow Score for word {word_chain3[i]}: {score:.4f}")
        
        # Print boundary words
        print("\nBoundary words where forward flow score > 0.7:")
        print(boundaries)
