import json
from tqdm import tqdm, trange
import pickle

import numpy as np

from typarse import BaseParser

class Parser(BaseParser):
    path: str = "arxiv-metadata-oai-snapshot.json"

    _help = {
        "path": "Path to the arxiv metadata file"
    }

    _abbrev = {
        "path": "p"
    }


if __name__ == "__main__":
    args = Parser()
    abstracts = []
    categories = []

    print("Reading the file...")
    # Read the file, line by line, and extract the abstract and categories
    with open(args.path, "r") as f:
        for line in tqdm(f, total=2246567):
            data = json.loads(line)
            abstracts.append(data['abstract'].strip())
            categories.append(data['categories'].split(' '))

    # Save the abstracts and categories for later use
    with open("abstracts.pkl", "wb") as f:
        pickle.dump(abstracts, f)

    with open("categories.pkl", "wb") as f:
        pickle.dump(categories, f)

    # Get all categories present in the dataset
    all_categories = sorted(  # Sort and convert to list
        set(  # Remove duplicates
            [item for sublist in categories for item in sublist]  # Flatten the list
        )
    )

    # Create a mapping from category to index and vice versa
    idx_to_cat = all_categories
    cat_to_idx = {name: idx for idx, name in enumerate(all_categories)}

    with open("endecoder.pkl", "wb") as f:
        pickle.dump((cat_to_idx, idx_to_cat), f)


    # For simplicity, use only the first category of each paper
    main_labels = np.array([cat_to_idx[c[0]] for c in categories])

    # Save the first category for each paper
    np.save("main_labels", main_labels)

    # Get only the top-level categories -- e.g. astro-ph.CO -> astro-ph
    idx_to_cat = sorted(list(set([c.split('.')[0] for c in all_categories])))
    cat_to_idx = {name: idx for idx, name in enumerate(idx_to_cat)}

    with open("top_endecoder.pkl", "wb") as f:
        pickle.dump((cat_to_idx, idx_to_cat), f)

    top_labels_data = np.array([cat_to_idx[c[0].split('.')[0]] for c in categories])

    np.save("top_labels", top_labels_data)