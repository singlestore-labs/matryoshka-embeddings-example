# Matryoshka Embeddings Example

Demonstrates using Matryoshka-capable embeddings with SingleStore. Shows how a single embedding can be truncated to different dimensions and compares search results at each dimensionality.

## Quick Start

```bash
pip install -r requirements.txt
python matryoshka_demo.py
```

## What This Demonstrates

- Generate ONE 768d embedding per document
- Truncate to create 64d, 128d, 256d, 512d, 768d prefixes
- Store each prefix in SingleStore (separate columns)
- Compare search results across dimensions
- Measure overlap with full 768d baseline

**Key insight:** Matryoshka models preserve meaning in prefixes. The example shows how search quality changes as dimensions decrease.

**Note:** This example demonstrates a best-case scenario with a Matryoshka-trained model on a small literary dataset. Real-world results will vary.

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

## Key Points

✅ **One embedding, multiple uses** - Truncate, don't regenerate  
✅ **Normalize after truncation** - Important for Matryoshka embeddings  
✅ **Nomic requires task prefixes** - Use `search_document:` and `search_query:`  
✅ **Overlap ≠ Recall** - Measures result overlap, not true retrieval accuracy

❌ **Not a performance benchmark** - Dataset too small (100 documents)  
❌ **Not measuring speed** - Focus is on result quality/overlap

## Limitations

- Requires Matryoshka-capable models (Nomic Embed v1.5, OpenAI text-embedding-3-*)
- Overlap measurement requires ground-truth labels for true recall
- Small dataset not suitable for performance testing

## Files

- `matryoshka_demo.py` - Working demo with Nomic Embed
- `pride_and_prejudice.txt` - Sample data
- `requirements.txt` - Dependencies
- `GOOGLE_DOC_DRAFT.md` - Complete documentation

## References

- [Matryoshka Representation Learning Paper](https://arxiv.org/abs/2205.13147)
- [Nomic Embed Documentation](https://huggingface.co/nomic-ai/nomic-embed-text-v1.5)
