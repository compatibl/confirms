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
import umap  # noqa umap-learn
import umap.plot # noqa umap-learn
import matplotlib.pyplot as plt
from sklearn.datasets import load_digits
from confirms.core.settings import Settings


def test_smoke():
    """Confirm that umap package is installed correctly."""

    settings = Settings()

    # TODO: Do not load dyamically, create locally instead
    digits = load_digits()

    mapper = umap.UMAP().fit(digits.data)
    umap.plot.points(mapper, labels=digits.target)

    # TODO: Convert the plot to html5 instead
    # Uncomment to see the plot
    # plt.show()


if __name__ == '__main__':
    pytest.main([__file__])
