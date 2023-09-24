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
import aiohttp
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

async def look_up(term):
    # looks up term on wikipedia
    url = f"https://en.wikipedia.org/w/api.php?format=json&action=query&prop=extracts&exintro&explaintext&redirects=1&titles={term}&origin=*"
    async with aiohttp.ClientSession() as session:
        async with session.get(url) as response:
            # get the first sentence on first page
            page = (await response.json())["query"]["pages"]
            return list(page.values())[0]["extract"].split(".")[0]
@lmql.query
def greet(term):
    '''lmql
    argmax
        """Greet {term} ({await look_up(term)}):
        Hello[WHO]
        """
    from
        "openai/text-davinci-003"
    where
        STOPS_AT(WHO, "\n")
    '''


def test_smoke():
    """Confirm that lmql package is installed correctly."""

    response = two_times_two()
    prompt = response[0].prompt
    result = int(response[0].variables["ANSWER"])
    assert prompt == "Two times two=4\n"
    assert result == 4


def test_rag():
    """Test retrieval augmentation."""

    response = greet("Earth")
    prompt = response[0].prompt
    result = response[0].variables["WHO"]
    print(prompt)
    print(result)
    pass


if __name__ == '__main__':
    pytest.main([__file__])
