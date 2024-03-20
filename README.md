# Twitter Sentiment Analysis

## Problem Statement

This problem involves developing a sentiment analysis solution specifically designed for analysing the sentiment expressed in the social media presence of individuals and organizations. With the significant impact of social media on personal and organizational reputation, understanding the sentiment of social media posts, comments, and interactions has become essential for individuals and businesses alike. Sentiment analysis refers to the process of automatically determining the sentiment or emotional tone conveyed by text or speech. In the context of social media, sentiment analysis can provide valuable insights into public perception, customer feedback, and brand reputation. By analysing the sentiments expressed in social media content, individuals and organizations can gauge the overall sentiment trends, identify potential issues, and take appropriate actions to maintain or enhance their online presence.

## Proposed Solution

### Transformers

- Transformers excel in language modelling due to their efficient capture of long-range dependencies in text.
- Their attention mechanisms enable effective focus on relevant parts of input sequences, surpassing limitations of RNNs and CNNs.
- Parallel computation and positional encodings enhance scalability and sequential relationship modelling, making transformers the preferred architecture for language tasks.

### Implementation

- Hugging Face Transformers
- PyTorch
- Pandas
- Matplotlib
- Flask
- Jinja Templates

## Setup

### Initialize a virtual environment

- Python 3.11.8
- pip 24.0

```bash
python3 -m venv .venv
```

### Activate the virtual environment

```bash
source .venv/bin/activate
```

### Install the packages

```bash
pip install -r requirements.txt
```

### Run the flask app

```bash
python -m flask --app app run
```
