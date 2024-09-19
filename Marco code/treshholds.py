import numpy as np
from sklearn.cluster import KMeans


# Loads GloVe word embeddings from a file into a dictionary.
# Keys are words, and values are their corresponding vector representations as NumPy arrays (which is based on semantic meaning).
def load_glove_vectors(file_path):
    print("Loading GloVe vectors...")
    with open(file_path, 'r', encoding='utf-8') as f:
        return {line.split()[0]: np.array(line.split()[1:], dtype=float) for line in f}


# Retrieves the GloVe vector for a given word from the dictionary.
# Returns None if the word is not in the GloVe vocabulary.
def get_word_embedding(word, glove_vectors):
    return glove_vectors.get(word.lower(), None)


# Computes the cosine similarity between two word vectors.
# This measures how similar the vectors are based on the angle between them.
def cosine_similarity(vec1, vec2):
    dot_product = np.dot(vec1, vec2)
    norm_a = np.linalg.norm(vec1)
    norm_b = np.linalg.norm(vec2)
    return dot_product / (norm_a * norm_b)


# Calculates the distance (1 - cosine similarity) between consecutive words in a list.
# The smaller the value, the closer the words are in meaning
def compute_consecutive_distances(word_list, glove_vectors):
    distances = []
    for i in range(len(word_list) - 1):
        vec1 = get_word_embedding(word_list[i], glove_vectors)
        vec2 = get_word_embedding(word_list[i+1], glove_vectors)
        if vec1 is not None and vec2 is not None:
            similarity = cosine_similarity(vec1, vec2)
            distance = 1 - similarity  # Convert similarity to distance
            distances.append(distance)
        else:
            distances.append(None)  # Handle words not in GloVe vocabulary
    return distances


# Finds a threshold for word distances using KMeans clustering or by calculating the mean + standard deviation.
# This threshold is used to distinguish between related and unrelated word pairs.
def find_threshold(distances, method='kmeans'):
    valid_distances = [d for d in distances if d is not None]
    if method == 'kmeans':
        kmeans = KMeans(n_clusters=2, random_state=42)
        labels = kmeans.fit_predict(np.array(valid_distances).reshape(-1, 1))
        threshold = np.mean(kmeans.cluster_centers_)
    else:
        threshold = np.mean(valid_distances) + np.std(valid_distances)
    return threshold


# Identifies word pairs whose distance exceeds the threshold, marking them as potential boundaries.
# It returns the indices of word pairs that have distances greater than the threshold.
def identify_boundaries(distances, threshold):
    return [i for i, d in enumerate(distances) if d is not None and d > threshold]




if __name__ == "__main__":
    # Load GloVe vectors
    glove_vectors = load_glove_vectors('/glove.6B/glove.6B.300d.txt')

    # Test Word Chains
    word_chain = ["water", "river", "flow", "current", "electricity", "power", "strength", "muscle", 
                  "exercise", "gym", "fitness", "health", "doctor", "hospital", "patient", "sick", 
                  "virus", "morning", "global", "world", "earth", "planet", "space", "star"]
    word_chain2 = ["water", "bottle", "cap", "lid", "pot", "pie", "apple", "cider", "hard", "boiled", 
                      "eggs", "chicken", "farm", "hand", "towel", "rack", "shelf", "book", "mark", "marker", "draw",
                      "picture", "frame", "art", "house", "fence", "gate", "latch", "key", "lock", "secret", "hide", 
                      "seek", "find", "map", "treasure", "chest", "armor", "knights", "castle", "king", "queen", "princess", 
                      "crown", "jewel", "diamond", "ruby", "emerald", "ruby", "red", "face", "makeup", "art", "drawing", "painting", 
                      "museum", "guide", "tour", "france", "crepe", "paper", "pencil", "case", "luggage", "travel"]
    word_chain3 = ["water", "board", "walk", "way", "finder", "keeper", "property", "value", "profit", "margin", 
                 	"allignment", "space", "exploration", "adventure", "rigorous", "difficult", "hard", "work", "effort", 
                 	"achievement", "celebration", "party", "socializing", "friends", "family", "tree", "nature", "outdoors", 
                	"environment", "climate", "science", "biology", "organisms", "life", "purpose", "decisions", "morality", 
                 	"personality", "behavior", "happiness", "emotion", "brain", "function"]

    # Choose which word chain to analyze
    current_chain = word_chain2  

    # Compute distances
    distances = compute_consecutive_distances(current_chain, glove_vectors)

    # Calculate threshold and identify boundaries
    threshold = find_threshold(distances)

    #manual
    #threshold = 0.8


    boundaries = identify_boundaries(distances, threshold)

    print(f"\nThreshold: {threshold:.4f}")
    print("Identified boundaries (indices):", boundaries)
    print("\nIdentified boundaries (word pairs):")
    for idx in boundaries:
        print(f"{current_chain[idx]} - {current_chain[idx+1]}")