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
import lmql
import confirms.core.settings


@lmql.query
def two_times_two():
    '''lmql
    argmax
        "Two times two=[ANSWER]\n"
    from
        "openai/text-ada-001"
    where
        len(TOKENS(ANSWER)) == 1
    '''


def test_smoke():
    """Confirm that lmql package is installed correctly."""

    response = two_times_two()
    prompt = response[0].prompt
    result = int(response[0].variables["ANSWER"])
    assert prompt == "Two times two=4\n"
    assert result == 4


if __name__ == '__main__':
    pytest.main([__file__])
