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

# Main execution
if __name__ == "__main__":
    # Load GloVe vectors
    glove_vectors = load_glove_vectors('/Users/marcoazar/Desktop/Cognitive Research/glove.6B/glove.6B.300d.txt')

    # Example word chain
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
    current_chain = word_chain

    # Compute distances
    distances = compute_consecutive_distances(current_chain, glove_vectors)

    # Print words and distances
    print("Word Pairs and Their Distances:")
    for i, (word1, word2, distance) in enumerate(zip(current_chain[:-1], current_chain[1:], distances)):
        if distance is not None:
            print(f"{i+1}. {word1} - {word2}: {distance:.4f}")
        else:
            print(f"{i+1}. {word1} - {word2}: N/A (One or both words not in vocabulary)")