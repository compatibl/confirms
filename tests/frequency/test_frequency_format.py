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


def test_gbnf_enforced_format():
    """Function completion with GBNF grammar."""

    model_types = ["llama-2-7b-chat.Q4_K_M.gguf", "llama-2-13b-chat.Q4_K_M.gguf"]  #, "llama-2-70b-chat.Q4_K_M.gguf"]
    for model_type in model_types:
        # TODO: Implement prompt
        question = ("```Issue Date: 9 July 2009 (Settlement Date), Maturity Date: 9 July 2013, Interest Payment Dates: The 9th "
            "of each January, April, July, and October commencing 9 October 2009 with a final payment on the "
            "Maturity Date.```")
        request = (
            "<s>[INST] Pay attention and remember information below, which will help to answer the question or imperative after the context ends. "
            f"Context: {question}. According to only the information in the document sources provided within the context above, the payment frequency is [/INST]")
        llm = LlamaLangChainLlm(model_type=model_type, temperature=0.2)  # , grammar_file="frequency.gbnf")
        answer = llm.completion(request)
        print(answer)


if __name__ == '__main__':
    pytest.main([__file__])
