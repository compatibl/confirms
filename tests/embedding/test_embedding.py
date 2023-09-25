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

import pytest
import chromadb
from chromadb.utils import embedding_functions
from chromadb import Documents, EmbeddingFunction, Embeddings
from confirms.core.settings import Settings


class CustomEmbeddingFunction(EmbeddingFunction):
    def __call__(self, texts: Documents) -> Embeddings:
        # TODO: Implement sample function
        embeddings = [1.0, 2.0]
        return embeddings


def test_smoke():
    """Test for embedding experiments."""

    settings = Settings()
    huggingface_ef = embedding_functions.HuggingFaceEmbeddingFunction(
        api_key="YOUR_API_KEY",
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )
    openai_ef = embedding_functions.OpenAIEmbeddingFunction(
        model_name="text-embedding-ada-002"
    )
    sentence_transformer_ef = embedding_functions.SentenceTransformerEmbeddingFunction(model_name="all-MiniLM-L6-v2")
    custom_ef = CustomEmbeddingFunction()

    # client = chromadb.Client(chromadb.config.Settings(chroma_db_impl="duckdb+parquet", persist_directory="db/"))
    client = chromadb.Client()  # In-memory
    collection = client.create_collection(name="Students")

    student_info = """
    Alexandra Thompson, a 19-year-old computer science sophomore with a 3.7 GPA,
    is a member of the programming and chess clubs who enjoys pizza, swimming, and hiking
    in her free time in hopes of working at a tech company after graduating from the University of Washington.
    """

    club_info = """
    The university chess club provides an outlet for students to come together and enjoy playing
    the classic strategy game of chess. Members of all skill levels are welcome, from beginners learning
    the rules to experienced tournament players. The club typically meets a few times per week to play casual games,
    participate in tournaments, analyze famous chess matches, and improve members' skills.
    """

    university_info = """
    The University of Washington, founded in 1861 in Seattle, is a public research university
    with over 45,000 students across three campuses in Seattle, Tacoma, and Bothell.
    As the flagship institution of the six public universities in Washington state,
    UW encompasses over 500 buildings and 20 million square feet of space,
    including one of the largest library systems in the world.
    """

    collection.add(
        documents=[student_info, club_info, university_info],
        metadatas=[{"source": "student info"}, {"source": "club info"}, {'source': 'university info'}],
        ids=["id1", "id2", "id3"]
    )

    results = collection.query(
        query_texts=["What is the student name?"],
        n_results=2
    )

    results
    pass


if __name__ == '__main__':
    pytest.main([__file__])
