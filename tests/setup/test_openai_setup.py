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
from langchain.llms import OpenAI


def test_llm():
    """Run a simple test to ensure each model is available."""

    model_names = ["gpt-3.5-turbo", "gpt-4"]
    for model_name in model_names:
        llm = OpenAI(model_name=model_name, temperature=0.0)
        output = llm("Two times two. Reply with the numerical result only, not a full sentence. "
                     "Your answer should include no text other than the number.")
        assert output == "4"


if __name__ == '__main__':
    pytest.main([__file__])
