import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

# Load GloVe word embeddings
def load_glove_embeddings(filepath):
    embeddings = {}
    with open(filepath, 'r', encoding='utf-8') as f:
        for line in f:
            values = line.split()
            word = values[0]
            vector = np.array(values[1:], dtype='float32')
            embeddings[word] = vector
    return embeddings

# Calculate semantic distance using cosine distance
def cosine_distance(vec1, vec2):
    return 1 - cosine_similarity([vec1], [vec2])[0][0]

# Compute forward flow for a sequence of words
def compute_forward_flow(word_sequence, glove_embeddings):
    # Step 1: Extract the embeddings for the word sequence
    vectors = []
    for word in word_sequence:
        if word in glove_embeddings:
            vectors.append(glove_embeddings[word])
        else:
            print(f"Word '{word}' not found in GloVe embeddings.")
            return None

    # Step 2: Calculate pairwise semantic distances
    n = len(vectors)
    semantic_distances = np.zeros((n, n))

    for i in range(n):
        for j in range(i + 1, n):
            distance = cosine_distance(vectors[i], vectors[j])
            semantic_distances[i][j] = distance
            semantic_distances[j][i] = distance

    # Step 3: Calculate instantaneous forward flow for each word
    forward_flow_scores = []
    for i in range(1, n):
        avg_distance = np.mean([semantic_distances[i][j] for j in range(i)])
        forward_flow_scores.append(avg_distance)

    # Step 4: Calculate overall forward flow score
    overall_forward_flow = np.mean(forward_flow_scores)

    return overall_forward_flow

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

    # Compute forward flow
    forward_flow_score = compute_forward_flow(word_chain, glove_vectors)

    if forward_flow_score is not None:
        print(f"Forward Flow Score: {forward_flow_score:.4f}")
