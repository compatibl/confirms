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
import numpy as np
import faiss  # noqa package faiss-cpu is used
from confirms.core.settings import Settings


def test_smoke():
    """Confirm that faiss package is installed correctly."""

    settings = Settings()
    d = 64  # dimension
    nb = 100000  # database size
    nq = 10000  # nb of queries
    np.random.seed(1234)  # make reproducible
    xb = np.random.random((nb, d)).astype('float32')
    xb[:, 0] += np.arange(nb) / 1000.
    xq = np.random.random((nq, d)).astype('float32')
    xq[:, 0] += np.arange(nq) / 1000.

    index = faiss.IndexFlatL2(d)  # build the index
    print(index.is_trained)
    index.add(xb)  # add vectors to the index
    print(index.ntotal)

    k = 4  # we want to see 4 nearest neighbors
    d_mat, i_mat = index.search(xb[:5], k)  # sanity check
    print(i_mat)
    print(d_mat)
    d_mat, i_mat = index.search(xq, k)  # actual search
    print(i_mat[:5])  # neighbors of the 5 first queries
    print(i_mat[-5:])  # neighbors of the 5 last queries


if __name__ == '__main__':
    pytest.main([__file__])
