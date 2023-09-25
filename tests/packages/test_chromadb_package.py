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
from confirms.core.settings import Settings


def test_smoke():
    """Confirm that chromadb package is installed correctly."""

    settings = Settings()
    client = chromadb.Client()

    collection = client.create_collection("sample_collection")

    # Add docs to the collection. Can also update and delete. Row-based API coming soon!
    collection.add(
        documents=["This is document1", "This is document2"],  # we embed for you, or bring your own
        metadatas=[{"source": "notion"}, {"source": "google-docs"}],  # filter on arbitrary metadata!
        ids=["doc1", "doc2"],  # must be unique for each doc
    )

    results = collection.query(
        query_texts=["This is a query document"],
        n_results=2,
        # where={"metadata_field": "is_equal_to_this"}, # optional filter
        # where_document={"$contains":"search_string"}  # optional filter
    )
    print(results)
    pass


if __name__ == '__main__':
    pytest.main([__file__])