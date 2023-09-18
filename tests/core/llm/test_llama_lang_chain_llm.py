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

from confirms.core.llm.llama_lang_chain_llm import LlamaLangChainLlm


def test_smoke():
    """Run a smoke test to ensure each model is available."""

    llama_model_types = ["llama-2-7b-chat.Q4_K_M.gguf", "llama-2-13b-chat.Q4_K_M.gguf"]
    for model_type in llama_model_types:
        llm = LlamaLangChainLlm(model_type=model_type, temperature=0.0, seed=0)
        output = llm.completion("What is two times two?")
        assert output == "\n\nAnswer: Two times two is equal to 4."


def test_function_completion():
    """Function completion"""

    llama_model_types = ["llama-2-7b-chat.Q4_K_M.gguf", "llama-2-13b-chat.Q4_K_M.gguf", "llama-2-70b-chat.Q4_K_M.gguf"]
    for model_type in llama_model_types:
        prompt = ("Act as a trade entry specialist whose goal is to extract parameters for a function "
                  "generating interest rate schedule from the text specified by the user. "
                  "These parameters are first_unadjusted_payment_date, last_unadjusted_payment_date, and"
                  "payment_frequency."
                  "Provide response only using the text specified by the user."
                  "Do not make up answers. Use ISO 8601 format for dates, namely YYYY-MM-DD, in your answer.")
        question = ("First unadjusted payment date is on January 15, 2000, "
                    "last unadjusted payment date is on January 15, 2005, and "
                    "payment frequency is 6M.")
        llm = LlamaLangChainLlm(model_type=model_type, temperature=0.0, grammar_file="payment_schedule_params.gbnf")
        answer = llm.completion(question, prompt=prompt)
        pass
        # assert answer["first_unadjusted_payment_date"] == "2000-01-15"
        # assert answer["last_unadjusted_payment_date"] == "2005-01-15"
        # assert answer["payment_frequency"] == "6M"


if __name__ == '__main__':
    pytest.main([__file__])
