import os

args = {
    "models": [
        # "bigscience/bloomz-560m",  # A reduced-size BLOOM model with 560 million parameters.
        # "facebook/opt-125m",        # A compact version of OPT with 125 million parameters.
        # "layonsan/google-t5-small", # The small version of T5 with 60 million parameters.
        # "openai-community/gpt2",    # GPT-2 model with 117 million parameters.
        "distilbert-base-uncased",  # A distilled version of BERT with 66 million parameters.
        "google/mobilebert-uncased",  # MobileBERT optimized for mobile devices with 25 million parameters.
    ],
    "datasets": [
        # {
        #     "name": "yelp_review_full",
        #     "config": None,
        #     "split": "train",
        #     "text_column": "text"
        # },
        # {
        #     "name": "wikitext",
        #     "config": "wikitext-2-raw-v1",
        #     "split": "train",
        #     "text_column": "text"
        # },
        {"name": "ag_news", "config": None, "split": "train", "text_column": "text"},
    ],
    "max_texts": 100,
    "batch_size": 64,
    "drift_strengths": [0.0, 0.25, 0.5, 0.75, 1.0],
    "pca_components": 8,
    "output_dir": "data",
    "num_seeds": 1,
    "kll_k": 8,
    "kll_bins": 20,
}

# Ensure output directory exists
os.makedirs(args["output_dir"], exist_ok=True)
