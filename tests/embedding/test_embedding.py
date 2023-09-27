# Copyright (C) 2023-present The Project Contributors
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
import os

import pytest
from langchain import FAISS
from langchain.document_loaders import PyPDFLoader
from langchain.embeddings import HuggingFaceEmbeddings, HuggingFaceInstructEmbeddings, OpenAIEmbeddings
from langchain.schema.embeddings import Embeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter

from confirms.core.settings import Settings


def run_similarity_search_for_document(embeddings: Embeddings, filename: str, query: str, chunk_size: int,
                                       chunk_overlap: int, top_k: int = None):
    """
    Perform similarity search on a PDF document.

    Args:
        embeddings (Embeddings): Embedding model.
        filename (str): The path to the PDF file.
        query (str): The query string to search for similarity.
        chunk_size (int): The size of text chunks for splitting the document.
        chunk_overlap (int): The overlap between text chunks.
        top_k (int, optional): The number of top matching segments to retrieve. If None, retrieves all segments.
    """
    # Load PDF document
    document = PyPDFLoader(filename).load()
    # Split text into chunks
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    # Split document into segments
    documents = text_splitter.split_documents(document)
    # Create FAISS index
    db = FAISS.from_documents(documents, embeddings, distance_strategy="EUCLIDEAN_DISTANCE")
    # Perform similarity search
    results = db.similarity_search_with_score(query, k=len(documents) if top_k is None else top_k)
    # Display results
    for result in results:
        print(result[1]) # Similarity score
        print('---------')
        print(result[0].page_content)
        print('---------') # Content of the segment


def run_maturity_date_extraction(embeddings: Embeddings):
    file_path = 'test_embedding.pdf'
    if not os.path.isfile(file_path):
        raise Exception(f"The file '{file_path}' does not exist. Please add a file with this name in this directory.")
    chunk_size = 500
    chunk_overlap = 50

    run_similarity_search_for_document(embeddings, file_path, 'Maturity Date:', chunk_size, chunk_overlap, 5)
    print('=====')
    run_similarity_search_for_document(embeddings, file_path, 'What is maturity date?', chunk_size, chunk_overlap, 5)
    print('=====')
    run_similarity_search_for_document(embeddings, file_path, 'Extract maturity date.', chunk_size, chunk_overlap, 5)


def test_maturity_date_extraction_hf():
    """Maturity date extraction test for hugging face embeddings"""
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    run_maturity_date_extraction(embeddings)


def test_maturity_date_extraction_instruct():
    """Maturity date extraction test for hugging face instruct embeddings"""
    embeddings = HuggingFaceInstructEmbeddings(model_name='hkunlp/instructor-xl')
    run_maturity_date_extraction(embeddings)


def test_maturity_date_extraction_ada():
    """Maturity date extraction test for OpenAI embeddings"""
    # Load settings
    Settings.load()
    embeddings = OpenAIEmbeddings(model="text-embedding-ada-002")
    run_maturity_date_extraction(embeddings)


if __name__ == '__main__':
    pytest.main([__file__])
