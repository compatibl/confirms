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

from confirms.core.llm.gpt_lang_chain_llm import GptLangChainLlm
from confirms.core.llm.gpt_native_llm import GptNativeLlm


def test_smoke():
    """Run a smoke test to ensure each model is available."""

    gpt_model_types = ["gpt-3.5-turbo", "gpt-4"]
    for model_type in gpt_model_types:
        llm = GptNativeLlm(model_type=model_type, temperature=0.0)
        output = llm.completion(
            "Two times two. Reply with the numerical result only, not a full sentence. "
            "Your answer should include no text other than the number."
        )
        assert output == "4"


def test_function_completion():
    """Function completion"""

    gpt_model_types = ["gpt-3.5-turbo", "gpt-4"]
    for model_type in gpt_model_types:
        question = (
            "Return interest schedule from this description: "
            "First unadjusted payment date is on January 15, 2000, "
            "last unadjusted payment date is on January 15, 2005, and "
            "payment frequency is 6M."
        )
        llm = GptNativeLlm(model_type=model_type, temperature=0.0)
        answer = llm.function_completion(question)
        assert answer["first_unadjusted_payment_date"] == "2000-01-15"
        assert answer["last_unadjusted_payment_date"] == "2005-01-15"
        assert answer["payment_frequency"] == "6M"


if __name__ == '__main__':
    pytest.main([__file__])
