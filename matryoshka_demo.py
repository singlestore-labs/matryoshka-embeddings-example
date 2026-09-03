#!/usr/bin/env python3
# Copyright 2026 SingleStore, Inc.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""
Matryoshka Embeddings - Dimension Comparison

Compares vector search results across 64, 128, 256, 512, and 768 dimensions.
"""

import json
import os
import numpy as np
import pymysql
from sentence_transformers import SentenceTransformer
from tabulate import tabulate

# Load configuration
with open('config.json', 'r') as f:
    config = json.load(f)
    DB_CONFIG = config['singlestore']

DIMENSIONS = [64, 128, 256, 512, 768]

print("Loading Nomic Embed model (first run downloads ~500MB)...")
model = SentenceTransformer(
    "nomic-ai/nomic-embed-text-v1.5",
    trust_remote_code=True
)
print("Model loaded!")


def connect_db():
    conn = pymysql.connect(
        host=DB_CONFIG["host"],
        port=DB_CONFIG["port"],
        user=DB_CONFIG["user"],
        password=DB_CONFIG["password"]
    )
    return conn


def load_text_chunks():
    """Load Pride and Prejudice, split into chunks"""
    with open('pride_and_prejudice.txt', 'r') as f:
        text = f.read()

    # Filter chunks: skip frontmatter, use meaningful paragraphs
    chunks = [p.strip() for p in text.split('\n\n') if p.strip() and len(p.strip()) > 100]
    # Use first 100 chunks for more meaningful search results
    return chunks[20:120]


def generate_document_embedding(text, dimensions):
    """Generate ONE full embedding and create dimension prefixes

    Nomic Embed v1.5 uses Matryoshka Representation Learning.

    Important: Normalize AFTER truncation, not before.
    This follows the Matryoshka procedure:
    1. Generate full embedding
    2. Truncate to desired dimension
    3. Normalize the truncated vector
    """
    # Use task prefix as recommended by Nomic
    # Do NOT normalize yet - we'll normalize after truncation
    full_embedding = model.encode(
        f"search_document: {text}",
        convert_to_numpy=True,
        normalize_embeddings=False
    )

    # Create prefixes by truncating, then normalize each
    result = {}
    for dim in dimensions:
        truncated = full_embedding[:dim]
        # Normalize the truncated vector
        normalized = truncated / np.linalg.norm(truncated)
        result[dim] = normalized.tolist()

    return result


def generate_query_embedding(text, dimensions):
    """Generate query embedding with proper task prefix

    Important: Normalize AFTER truncation, not before.
    """
    # Use task prefix as recommended by Nomic
    # Do NOT normalize yet - we'll normalize after truncation
    full_embedding = model.encode(
        f"search_query: {text}",
        convert_to_numpy=True,
        normalize_embeddings=False
    )

    # Create prefixes by truncating, then normalize each
    result = {}
    for dim in dimensions:
        truncated = full_embedding[:dim]
        # Normalize the truncated vector
        normalized = truncated / np.linalg.norm(truncated)
        result[dim] = normalized.tolist()

    return result


def setup_database(conn):
    cursor = conn.cursor()
    cursor.execute(f"CREATE DATABASE IF NOT EXISTS {DB_CONFIG['database']}")
    cursor.execute(f"USE {DB_CONFIG['database']}")

    cursor.execute("DROP TABLE IF EXISTS documents")
    cursor.execute("""
        CREATE TABLE documents (
            id INT PRIMARY KEY AUTO_INCREMENT,
            text TEXT,
            embedding_64 VECTOR(64, F32) NOT NULL,
            embedding_128 VECTOR(128, F32) NOT NULL,
            embedding_256 VECTOR(256, F32) NOT NULL,
            embedding_512 VECTOR(512, F32) NOT NULL,
            embedding_768 VECTOR(768, F32) NOT NULL,
            VECTOR INDEX (embedding_64) INDEX_OPTIONS '{"metric_type":"DOT_PRODUCT"}',
            VECTOR INDEX (embedding_128) INDEX_OPTIONS '{"metric_type":"DOT_PRODUCT"}',
            VECTOR INDEX (embedding_256) INDEX_OPTIONS '{"metric_type":"DOT_PRODUCT"}',
            VECTOR INDEX (embedding_512) INDEX_OPTIONS '{"metric_type":"DOT_PRODUCT"}',
            VECTOR INDEX (embedding_768) INDEX_OPTIONS '{"metric_type":"DOT_PRODUCT"}'
        )
    """)
    conn.commit()
    cursor.close()


def load_documents(conn, chunks):
    cursor = conn.cursor()
    cursor.execute(f"USE {DB_CONFIG['database']}")

    # Prepare column names once
    cols = ["text"] + [f"embedding_{dim}" for dim in DIMENSIONS]
    placeholders = ", ".join(["%s"] * (1 + len(DIMENSIONS)))

    # Collect all data for batch insert
    batch_data = []
    for i, text in enumerate(chunks, 1):
        print(f"  Processing chunk {i}/{len(chunks)}...", end='\r')
        # Generate ONE 768d embedding, then create prefixes
        embeddings = generate_document_embedding(text, DIMENSIONS)
        # Convert vectors to JSON strings as per documentation
        values = [text] + [json.dumps(embeddings[dim]) for dim in DIMENSIONS]
        batch_data.append(values)

    print()  # New line after progress

    # Batch insert all documents
    cursor.executemany(
        f"INSERT INTO documents ({', '.join(cols)}) VALUES ({placeholders})",
        batch_data
    )
    conn.commit()
    cursor.close()


def search_at_dimension(conn, query_embedding, dimension, limit=5):
    cursor = conn.cursor()
    cursor.execute(f"USE {DB_CONFIG['database']}")

    # Convert query embedding to JSON string
    query_vec_json = json.dumps(query_embedding)
    cursor.execute(f"""
        SELECT id, text,
               DOT_PRODUCT(embedding_{dimension}, %s :> VECTOR({dimension})) AS score
        FROM documents
        ORDER BY score DESC
        LIMIT %s
    """, (query_vec_json, limit))
    results = cursor.fetchall()

    cursor.close()
    return results


def calculate_overlap(reference_results, test_results):
    """Calculate top-5 overlap with 768d baseline

    This measures how many of the same documents appear in both result sets,
    not true recall (which would require ground-truth relevance labels).
    """
    ref_ids = set(r[0] for r in reference_results[:5])
    test_ids = set(r[0] for r in test_results[:5])
    if not ref_ids:
        return 0.0
    return (len(ref_ids & test_ids) / len(ref_ids)) * 100


def compare_search(conn, query_text):
    print(f'\nQuery: "{query_text}"')
    print("="*70)

    # Generate query embedding with proper task prefix
    query_embeddings = generate_query_embedding(query_text, DIMENSIONS)

    all_results = {}

    for dim in DIMENSIONS:
        results = search_at_dimension(conn, query_embeddings[dim], dim)
        all_results[dim] = results

    reference_dim = max(DIMENSIONS)
    reference_results = all_results[reference_dim]

    for dim in DIMENSIONS:
        results = all_results[dim]
        overlap = calculate_overlap(reference_results, results)

        print(f"\n{dim}d: top-5 overlap {overlap:.0f}%")
        for i, (doc_id, text, score) in enumerate(results[:3], 1):
            preview = text[:60] + "..." if len(text) > 60 else text
            print(f"  {i}. {preview}")

    # Summary
    table_data = []
    for dim in DIMENSIONS:
        overlap = calculate_overlap(reference_results, all_results[dim])

        table_data.append([
            f"{dim}d",
            f"{overlap:.0f}%" if dim != reference_dim else "100%"
        ])

    print(f"\n{tabulate(table_data, headers=['Dimensions', 'Top-5 overlap with 768d'], tablefmt='grid')}")


def main():
    print("Matryoshka Embeddings - Dimension Comparison")
    print("="*70)

    print("Connecting to database...")
    conn = connect_db()
    print("Connected!")

    try:
        print("Loading text chunks...")
        chunks = load_text_chunks()
        print(f"Loaded {len(chunks)} text chunks")

        print("Setting up database...")
        setup_database(conn)
        print("Database ready!")

        print("Generating embeddings and loading documents...")
        load_documents(conn, chunks)
        print("Documents loaded!")

        # Semantic queries that test understanding, not just keyword matching
        compare_search(conn, "Why does Elizabeth initially dislike Mr. Darcy?")
        compare_search(conn, "What are the Bennet family's financial concerns?")
        compare_search(conn, "How do first impressions deceive the characters?")

    finally:
        conn.close()


if __name__ == "__main__":
    main()
