# VS Code Requirement: You will almost certainly have this already installed in your VS Code or Cursor, so no need to do anything
# - The Microsoft Python extension (ms-python.python)
# This extension provides the ability to run '%%' cells interactively.
# Below you should see "Run Cell" and "Run Below" that is written just above '# %% [markdown]' line

# %% [markdown]
# # Building a Byte Pair Encoding (BPE) Tokenizer from Scratch
#
# This tutorial walks through the process of creating a basic BPE tokenizer, a common type of tokenizer used in Large Language Models (LLMs).
#
# ## Step 1: Prepare Training Data
#
# The first step in building any tokenizer is to have a corpus of text to train it on. The tokenizer learns merge rules based on the frequency of character pairs in this data.
#
# i: 1
#
# s: 2
#
# is: 3
#
# Even though "i" and "s" are separate tokens, we create a new token "is" by merging them as they frequently appear together (is, this, his, miss, dismiss, list, fist, twist, mist, whisk, visible, vision, revise, crisis), reducing computation needs by 2x at any place where we merge those 2 tokens. This is how we will itteratively merge most frequent pairs. The new tokens can also be further merged.
#
# Let's start with a small example corpus.

# %%
# Our sample training data
corpus = [
    "This is the first document.",
    "This document is the second document.",
    "And this is the third one.",
    "Is this the first document?",
]

print("Training Corpus:")
for doc in corpus:
    print(doc)

# %% [markdown]
# ## Step 2: Initialize Vocabulary and Pre-tokenize
#
# The BPE algorithm starts with a base vocabulary consisting of all unique characters present in the training data.
#
# We also need to pre-tokenize the corpus. This usually involves splitting the text into words (or word-like units) and then representing each word as a sequence of its individual characters. We often add a special end-of-word token (like `</w>`) to mark word boundaries, which helps the tokenizer learn subword units that align better with whole words.

# %%
# Initialize vocabulary with unique characters
unique_chars = set()
for doc in corpus:
    for char in doc:
        unique_chars.add(char)

vocab = list(unique_chars)
vocab.sort() # For consistent order of characters, making the vocabulary list predictable

# Add a special end-of-word token
end_of_word = "</w>"
vocab.append(end_of_word)

print("Initial Vocabulary:")
print(vocab)
print(f"Vocabulary Size: {len(vocab)}")

# Pre-tokenize the corpus: Split into words and then characters
# We'll split by space for simplicity and add the end-of-word token
word_splits = {}
for doc in corpus:
    words = doc.split(' ')
    for word in words:
        if word:
            char_list = list(word) + [end_of_word]
            # Use tuple for immutability if storing counts later - you can't change tuple once it's created (values, order, adding, removing elements, etc.), so they can be used as dictionary keys because of that.
            word_tuple = tuple(char_list)
            if word_tuple not in word_splits:
                 word_splits[word_tuple] = 0
            word_splits[word_tuple] += 1 # Count frequency of each initial word split

print("\nPre-tokenized Word Frequencies:")
print(word_splits)

# %% [markdown]
# ### Helper Function: `get_pair_stats`
#
# This function takes the current word splits (represented as a dictionary where keys are tuples of symbols/characters forming a word and values are their frequencies) and calculates the frequency of each adjacent pair of symbols across the entire corpus.
#
# **Input Example (`splits`):**
# ```
# {('T', 'h', 'i', 's', '</w>'): 2, ('i', 's', '</w>'): 2, ...}
# ```
# **Output Example (`pair_counts`):**
# ```
# {('i', 's'): 4, ('s', '</w>'): 4, ('T', 'h'): 2, ...}
# ```

# %%
import collections

def get_pair_stats(splits):
    """Counts the frequency of adjacent pairs in the word splits."""
    # Initialize a dictionary with default values of 0 to count pairs of symbols.
    # defaultdict: It's like a regular dictionary (dict), but with a key difference.
    # If you try to access or modify a key that doesn't exist, instead of raising a KeyError,
    # it automatically creates that key and assigns it a default value.
    # int: This is the "default factory" you provide when creating the defaultdict. When a new key is created, it needs a default value, defaultdict calls this factory function. int() called with no arguments returns 0.
    pair_counts = collections.defaultdict(int)
    for word_tuple, freq in splits.items():
        symbols = list(word_tuple)
        for i in range(len(symbols) - 1):
            pair = (symbols[i], symbols[i+1])
            pair_counts[pair] += freq # Add the frequency of the word to the pair count
    return pair_counts

# %% [markdown]
# ### Helper Function: `merge_pair`
#
# This function takes a specific pair (`pair_to_merge`) that we want to combine and the current `splits`. It iterates through all the word representations in `splits`, replaces occurrences of the `pair_to_merge` with a new single token (concatenation of the pair), and returns the updated `splits`.
#
# **Input Example:**
# - `pair_to_merge`: `('i', 's')`
# - `splits`: `{('T', 'h', 'i', 's', '</w>'): 2, ('i', 's', '</w>'): 2, ...}`
#
# **Output Example (`new_splits`):**
# - `{('T', 'h', 'is', '</w>'): 2, ('is', '</w>'): 2, ...}` (assuming 'is' is the merged token)

# %%
def merge_pair(pair_to_merge, splits):
    """Merges the specified pair in the word splits."""
    new_splits = {}
    (first, second) = pair_to_merge
    merged_token = first + second
    for word_tuple, freq in splits.items():
        symbols = list(word_tuple)
        new_symbols = []
        i = 0
        while i < len(symbols):
            # If the current and next symbol match the pair to merge
            if i < len(symbols) - 1 and symbols[i] == first and symbols[i+1] == second:
                new_symbols.append(merged_token)
                i += 2 # Skip the next symbol
            else:
                new_symbols.append(symbols[i])
                i += 1
        new_splits[tuple(new_symbols)] = freq # Use the updated symbol list as the key
    return new_splits

# %% [markdown]
# ### Step 3: Iterative BPE Merging Loop
#
# Now we perform the core BPE training. We'll loop for a fixed number of merges (`num_merges`). In each iteration:
# 1. Calculate the frequencies of all adjacent pairs in the current word representations using `get_pair_stats`.
# 2. Find the pair with the highest frequency (`best_pair`).
# 3. Merge this `best_pair` across all word representations using `merge_pair`.
# 4. Add the newly formed token (concatenation of `best_pair`) to our vocabulary (`vocab`).
# 5. Store the merge rule (mapping the pair to the new token) in the `merges` dictionary.
#
# We'll add print statements to observe the state at each step of the loop.

# %%
# --- BPE Training Loop Initialization ---
num_merges = 15
# Stores merge rules, e.g., {('a', 'b'): 'ab'}
# Example: {('T', 'h'): 'Th'}
merges = {}
# Initial word splits: {('T', 'h', 'i', 's', '</w>'): 2, ('i', 's', '</w>'): 2, ...}
current_splits = word_splits.copy() # Start with initial word splits

print("\n--- Starting BPE Merges ---")
print(f"Initial Splits: {current_splits}")
print("-" * 30)

for i in range(num_merges):
    print(f"\nMerge Iteration {i+1}/{num_merges}")

    # 1. Calculate Pair Frequencies
    pair_stats = get_pair_stats(current_splits)
    if not pair_stats:
        print("No more pairs to merge.")
        break
    # Optional: Print top 5 pairs for inspection
    sorted_pairs = sorted(pair_stats.items(), key=lambda item: item[1], reverse=True)
    print(f"Top 5 Pair Frequencies: {sorted_pairs[:5]}")

    # 2. Find Best Pair
    # The 'max' function iterates over all key-value pairs in the 'pair_stats' dictionary
    # The 'key=pair_stats.get' tells 'max' to use the frequency (value) for comparison, not the pair (key) itself
    # This way, 'max' selects the pair with the highest frequency
    best_pair = max(pair_stats, key=pair_stats.get)
    best_freq = pair_stats[best_pair]
    print(f"Found Best Pair: {best_pair} with Frequency: {best_freq}")

    # 3. Merge the Best Pair
    current_splits = merge_pair(best_pair, current_splits)
    new_token = best_pair[0] + best_pair[1]
    print(f"Merging {best_pair} into '{new_token}'")
    print(f"Splits after merge: {current_splits}")

    # 4. Update Vocabulary
    vocab.append(new_token)
    print(f"Updated Vocabulary: {vocab}")

    # 5. Store Merge Rule
    merges[best_pair] = new_token
    print(f"Updated Merges: {merges}")

    print("-" * 30)


# %% [markdown]
# ### Step 4: Review Final Results
#
# After the loop finishes, we can examine the final state:
# - The learned merge rules (`merges`).
# - The final representation of words after merges (`current_splits`).
# - The complete vocabulary (`vocab`) containing initial characters and learned subword tokens.

# %%
# --- BPE Merges Complete ---
print("\n--- BPE Merges Complete ---")
print(f"Final Vocabulary Size: {len(vocab)}")
print("\nLearned Merges (Pair -> New Token):")
# Pretty print merges
for pair, token in merges.items():
    print(f"{pair} -> '{token}'")

print("\nFinal Word Splits after all merges:")
print(current_splits)

print("\nFinal Vocabulary (sorted):")
# Sort for consistent viewing
final_vocab_sorted = sorted(list(set(vocab))) # Use set to remove potential duplicates if any step introduced them
print(final_vocab_sorted)

# %%
