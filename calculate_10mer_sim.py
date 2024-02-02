import pandas as pd
import numpy as np
from scipy.spatial.distance import cosine
from scipy.stats import pearsonr
import Levenshtein as lv
import itertools
import numpy as np
from gensim.models import Word2Vec


# Read the CSV file
df = pd.read_csv('10mer_similarity_adjacency_matrix.csv')

# Extract the column names as the pool of possible sequences
pool = df.columns.astype(str)
pool = pool[1:-2] # remove first, the 'Antibiotic' and 'MIC' columns

# Calculate pairwise similarity
similarity_matrix = np.zeros((len(pool), len(pool)))

def hamming_distance(a, b):
    """Calculate the Hamming distance between two k-mers."""
    if len(a) != len(b):
        raise ValueError("Strings must be the same length")
    return sum(ch1 != ch2 for ch1, ch2 in zip(a, b))

def levenshtein_distance(a, b):
    """Calculate the Levenshtein distance between two k-mers."""
    return lv.distance(a, b)

def jaccard_index(a, b):
    """Calculate the Jaccard index between two k-mers."""
    set_a = set(a)
    set_b = set(b)
    return len(set_a.intersection(set_b)) / len(set_a.union(set_b))

def vectorize_kmer(kmer):
    """Convert a k-mer into a simple frequency vector."""
    vector = np.zeros(4) # Assuming only A, T, C, G
    for char in kmer:
        if char == 'A':
            vector[0] += 1
        elif char == 'T':
            vector[1] += 1
        elif char == 'C':
            vector[2] += 1
        elif char == 'G':
            vector[3] += 1
    return vector

def cosine_similarity(a, b):
    """Calculate the cosine similarity between two k-mers."""
    vec_a = vectorize_kmer(a)
    vec_b = vectorize_kmer(b)
    return 1 - cosine(vec_a, vec_b)


def sequence_alignment_score(a, b):
    """A simple sequence alignment score."""
    # This is a placeholder for a real sequence alignment algorithm
    return lv.ratio(a, b)

def needleman_wunsch(a, b, match_score=1, gap_cost=1, mismatch_cost=1):
    """Calculate the Needleman-Wunsch similarity between two k-mers."""
    len_a, len_b = len(a), len(b)
    dp = np.zeros((len_a + 1, len_b + 1))

    for i in range(len_a + 1):
        dp[i][0] = -i * gap_cost
    for j in range(len_b + 1):
        dp[0][j] = -j * gap_cost

    for i in range(1, len_a + 1):
        for j in range(1, len_b + 1):
            if a[i - 1] == b[j - 1]:
                score = match_score
            else:
                score = -mismatch_cost
            dp[i][j] = max(dp[i - 1][j - 1] + score, dp[i - 1][j] - gap_cost, dp[i][j - 1] - gap_cost)
    
    nw_score = dp[len_a][len_b]
    max_score = max(len_a, len_b) * match_score
    return (nw_score + max_score) / (2 * max_score)  # Normalize to [0, 1]

def dice_coefficient(a, b):
    """Calculate the Dice coefficient between two k-mers."""
    set_a = set(a)
    set_b = set(b)
    overlap = len(set_a.intersection(set_b))
    return 2 * overlap / (len(set_a) + len(set_b))

# Example usage
a = 'ATCG'
b = 'ATGC'
print("Hamming Distance:", hamming_distance(a, b))
print("Levenshtein Distance:", levenshtein_distance(a, b))
print("Jaccard Index:", jaccard_index(a, b))
print("Cosine Similarity:", cosine_similarity(a, b))
print("Sequence Alignment Score:", sequence_alignment_score(a, b))
# print("K-mer Spectrum Similarity:", kmer_spectrum_similarity(a, b))
# exit()

for func_name in ['hamming_distance', 'levenshtein_distance', 'jaccard_index', 'cosine_similarity', 'sequence_alignment_score', 'needleman_wunsch', 'dice_coefficient']:
    similarity_matrix = np.zeros((len(pool), len(pool)))
    for i in range(len(pool)):
        for j in range(i+1, len(pool)):
            similarity_scores = eval(func_name)(pool[i], pool[j])  # Call the function directly
            similarity_matrix[i][j] = float(similarity_scores)
            similarity_matrix[j][i] = float(similarity_scores)
   
    # Save the adjacency matrix as a CSV file
    df_similarity = pd.DataFrame(similarity_matrix, index=pool, columns=pool)

    # Normalize Hamming distance
    if func_name == 'hamming_distance':
        max_hamming_distance = len(pool[0])
        df_similarity = 1 - df_similarity / max_hamming_distance

    # Normalize Levenshtein distance
    if func_name == 'levenshtein_distance':
        max_levenshtein_distance = len(pool[0])
        df_similarity = 1 - df_similarity / max_levenshtein_distance

    df_similarity.to_csv(f'adjacency_matrix_{func_name}.csv')



