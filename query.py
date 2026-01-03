import zipfile
import os
import glob
import re
import random
import time
import pickle
from collections import defaultdict
from typing import Dict, List
import numpy as np
import matplotlib.pyplot as plt


def extract_dataset():
    """Extract the 20news-bydate dataset if not already extracted."""
    if not os.path.exists('data'):
        with zipfile.ZipFile('20news-bydate.zip', 'r') as zip_ref:
            zip_ref.extractall('data')
        print("Dataset extracted to data/.")
    else:
        print("Dataset already extracted to data/.")


def preprocess_document(file_path: str) -> str:
    """Read and preprocess a document file."""
    try:
        with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
            lines = f.readlines()
        
        # Find the blank line that separates headers from body
        body_start = 0
        for i in range(len(lines)):
            if lines[i].strip() == '':
                body_start = i + 1
                break
        
        # Extract body text
        body = ' '.join(lines[body_start:])
        
        # Normalize: lowercase and remove non-alphanumeric except spaces
        body = re.sub(r'[^a-zA-Z0-9\s]', '', body.lower())
        
        return body
    except Exception as e:
        return ""


def build_inverted_index(train_path: str, stopwords: set) -> Dict[str, List[int]]:
    """Build inverted index from training documents."""
    inverted_index = defaultdict(list)
    doc_id = 0
    
    # Find all files in train subdirectories (files may not have .txt extension)
    pattern = os.path.join(train_path, '*', '*')
    files = [f for f in glob.glob(pattern) if os.path.isfile(f)]
    
    print(f"Found {len(files)} files to process...")
    
    for file_path in files:
        body = preprocess_document(file_path)
        
        if not body:
            continue
        
        # Tokenize and filter
        tokens = body.split()
        tokens = [t for t in tokens if len(t) >= 3 and t not in stopwords]
        
        # Add to inverted index
        seen_tokens = set()
        for token in tokens:
            if token not in seen_tokens:
                inverted_index[token].append(doc_id)
                seen_tokens.add(token)
        
        doc_id += 1
    
    # Sort posting lists
    for token in inverted_index:
        inverted_index[token].sort()
    
    print(f"Total documents processed: {doc_id}")
    print(f"Vocabulary size: {len(inverted_index)}")
    
    return dict(inverted_index), doc_id


def intersect_postings(postings1: List[int], postings2: List[int]) -> List[int]:
    """Intersect two sorted posting lists using two-pointer merge."""
    if not postings1 or not postings2:
        return []
    
    result = []
    i, j = 0, 0
    
    while i < len(postings1) and j < len(postings2):
        if postings1[i] == postings2[j]:
            result.append(postings1[i])
            i += 1
            j += 1
        elif postings1[i] < postings2[j]:
            i += 1
        else:
            j += 1
    
    return result


def process_query(query_words: List[str], inverted_index: Dict[str, List[int]], strategy: str) -> float:
    """Process a query and return execution time."""
    # Fetch posting lists
    postings = []
    for word in query_words:
        if word in inverted_index and inverted_index[word]:
            postings.append(inverted_index[word])
    
    if len(postings) < 2:
        return 0.0
    
    # Sort based on strategy
    if strategy == 'shortest_length':
        postings.sort(key=lambda p: len(p))
    elif strategy == 'smallest_last_docid':
        postings.sort(key=lambda p: p[-1] if p else float('inf'))
    
    # Time the intersection
    start_time = time.perf_counter()
    
    result = postings[0][:]  # Copy first posting list
    for i in range(1, len(postings)):
        result = intersect_postings(result, postings[i])
        if not result:  # Early termination if empty
            break
    
    end_time = time.perf_counter()
    
    return end_time - start_time


def generate_queries(vocabulary: List[str], num_queries: int = 10000) -> List[List[str]]:
    """Generate random queries with 4-10 words."""
    queries = []
    if not vocabulary:
        print("Warning: empty vocabulary — no queries generated.")
        return queries

    for _ in range(num_queries):
        query_length = random.randint(4, 10)
        # Use sampling with replacement so small vocabularies still produce queries
        query = random.choices(vocabulary, k=query_length)
        queries.append(query)
    
    print(f"Generated {len(queries)} random queries.")
    return queries


def create_visualizations(times_view1: List[float], times_view2: List[float]):
    """Create visualization plots."""
    # Direct comparison scatter plot
    plt.figure(figsize=(10, 8))
    plt.scatter(times_view1, times_view2, alpha=0.5, s=1)
    
    # Add y=x line
    min_time = min(min(times_view1), min(times_view2))
    max_time = max(max(times_view1), max(times_view2))
    plt.plot([min_time, max_time], [min_time, max_time], 'r--', label='y=x')
    
    plt.xlabel('Time Viewpoint 1 (s)')
    plt.ylabel('Time Viewpoint 2 (s)')
    plt.title('Comparison of Response Times')
    plt.legend()
    plt.grid(True)
    plt.show()
    
    # Separate plots for each viewpoint
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4))
    
    # Viewpoint 1
    ax1.scatter(range(len(times_view1)), times_view1, alpha=0.5, s=1)
    ax1.axhline(y=np.mean(times_view1), color='r', linestyle='--', 
                label=f'Mean: {np.mean(times_view1):.6f}s')
    ax1.set_xlabel('Query Index')
    ax1.set_ylabel('Time (s)')
    ax1.set_title('Viewpoint 1')
    ax1.legend()
    ax1.grid(True)
    
    # Viewpoint 2
    ax2.scatter(range(len(times_view2)), times_view2, alpha=0.5, s=1)
    ax2.axhline(y=np.mean(times_view2), color='r', linestyle='--', 
                label=f'Mean: {np.mean(times_view2):.6f}s')
    ax2.set_xlabel('Query Index')
    ax2.set_ylabel('Time (s)')
    ax2.set_title('Viewpoint 2')
    ax2.legend()
    ax2.grid(True)
    
    plt.tight_layout()
    plt.show()


def main():
    # Define stopwords
    stopwords = {
        'the', 'be', 'to', 'of', 'and', 'a', 'in', 'that', 'have', 'i', 'it', 'for',
        'not', 'on', 'with', 'he', 'as', 'you', 'do', 'at', 'this', 'but', 'his', 'by',
        'from', 'they', 'she', 'or', 'an', 'will', 'my', 'one', 'all', 'would', 'there',
        'their', 'what', 'so', 'up', 'out', 'if', 'about', 'who', 'get', 'which', 'go', 'me'
    }
    
    # Step 1: Extract dataset
    extract_dataset()
    
    # Step 2: Build inverted index
    print("\nBuilding inverted index...")
    # The dataset directories are named '20news-bydate-train' and '20news-bydate-test'
    train_path = os.path.join('data', '20news-bydate', '20news-bydate-train')
    inverted_index, total_docs = build_inverted_index(train_path, stopwords)
    
    # Save inverted index
    with open('inverted_index.pkl', 'wb') as f:
        pickle.dump(inverted_index, f)
    print("Inverted index saved to inverted_index.pkl")
    
    # Step 3: Extract vocabulary
    vocabulary = list(inverted_index.keys())
    print(f"\nNumber of unique words: {len(vocabulary)}")
    if len(vocabulary) == 0:
        print("No vocabulary found — exiting.")
        return
    
    # Step 4: Generate queries
    print("\nGenerating queries...")
    queries = generate_queries(vocabulary, num_queries=10000)
    
    # Step 5: Run experiments
    print("\nRunning experiments...")
    times_view1 = []
    times_view2 = []
    
    for i, query in enumerate(queries):
        if (i + 1) % 1000 == 0:
            print(f"Processed {i + 1} queries...")
        
        time1 = process_query(query, inverted_index, 'shortest_length')
        time2 = process_query(query, inverted_index, 'smallest_last_docid')
        
        times_view1.append(time1)
        times_view2.append(time2)
    
    # Print mean times
    print(f"\nMean time Viewpoint 1: {np.mean(times_view1):.6f} s")
    print(f"Mean time Viewpoint 2: {np.mean(times_view2):.6f} s")
    
    # Step 6: Create visualizations
    print("\nCreating visualizations...")
    create_visualizations(times_view1, times_view2)
    
    # Step 7: Analysis
    times_array1 = np.array(times_view1)
    times_array2 = np.array(times_view2)
    percentage_faster = np.mean(times_array1 < times_array2) * 100
    
    print(f"\nViewpoint 1 average: {np.mean(times_view1):.6f} s, "
          f"Viewpoint 2 average: {np.mean(times_view2):.6f} s. "
          f"Viewpoint 1 is faster in {percentage_faster:.1f}% of queries.")


if __name__ == '__main__':
    main()