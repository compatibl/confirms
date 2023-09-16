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
from langchain import LLMChain, PromptTemplate
from langchain.llms import OpenAI


class TestLangChain:
    """Confirm that the OpenAI package has been installed correctly and its API key is present."""

    def test_model_availability(self):
        """Run a simple test to ensure each model is available."""

        self._test_2x2(model_name="gpt-3.5-turbo")
        self._test_2x2(model_name="gpt-4")

    def _test_2x2(self, *, model_name: str) -> None:
        """Simple question to check model availability."""

        llm = OpenAI(model_name=model_name, temperature=0.)
        output = llm("Two times two. Reply with the numerical result only, not a full sentence.")
        assert output == "4"


if __name__ == '__main__':
    pytest.main([__file__])
