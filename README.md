# Matryoshka Embeddings Example

**Attention**: The code in this repository is intended for experimental use only and is not fully tested, documented, or supported by SingleStore. Visit the [SingleStore Forums](https://www.singlestore.com/forum) to ask questions about this repository.

Demonstrates using Matryoshka-capable embeddings with SingleStore. Shows how a single embedding can be truncated to different dimensions and compares search results at each dimensionality.

## Quick Start

```bash
# Install dependencies
pip install -r requirements.txt

# Configure database connection
cp config.json.example config.json
# Edit config.json with your SingleStore connection details

# Run the demo
python matryoshka_demo.py
```

## What This Demonstrates

- Generate ONE 768d embedding per document
- Truncate to create 64d, 128d, 256d, 512d, 768d prefixes
- Store each prefix in SingleStore (separate columns)
- Compare search results across dimensions
- Measure overlap with full 768d baseline


**Note:** This example demonstrates an example scenario with a Matryoshka-trained model on a small literary dataset. Real-world results will vary.

## Example Output

```
Query: "Why does Elizabeth initially dislike Mr. Darcy?"

64d:  overlap 40%
256d: overlap 80%
512d: overlap 100%
768d: overlap 100% (baseline)
```

## How It Works

```python
# 1. Generate ONE full embedding
embedding = model.encode(f"search_document: {text}", ...)

# 2. Truncate to different sizes
embedding_256 = embedding[:256]
embedding_768 = embedding[:768]

# 3. Normalize after truncation
embedding_256 = embedding_256 / np.linalg.norm(embedding_256)

# 4. Store and search at different dimensions
```

## Files

- `matryoshka_demo.py` - Working demo with Nomic Embed
- `pride_and_prejudice.txt` - Sample data
- `requirements.txt` - Dependencies
- `config.json.example` - Database configuration template

## References

- [Matryoshka Representation Learning Paper](https://arxiv.org/abs/2205.13147)
- [Nomic Embed Documentation](https://huggingface.co/nomic-ai/nomic-embed-text-v1.5)
