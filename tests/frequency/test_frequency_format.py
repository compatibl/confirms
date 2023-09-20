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
        # TODO - implement prompt
        question = ("First unadjusted payment date is on January 15, 2000, "
                    "last unadjusted payment date is on January 15, 2005, and "
                    "payment frequency is semiannual."
                    )
        request = (
            "<s>[INST] <<SYS>>Act as a trade entry specialist whose goal is to extract only the payment"
            "frequency from the context."
            "Answer with payment frequency using strict format based on these examples: "
            "1D, 2W, 3M, or an empty string. "
            "NEVER hallucinate the answer - return an empty string if payment frequency is"
            "not known. \n <</SYS>> \n\n"
            f"Context: {question}. Only return the helpful answer below and nothing else. Helpful answer:[/INST]")
        llm = LlamaLangChainLlm(model_type=model_type, temperature=0.0, grammar_file="frequency.gbnf")
        answer = llm.completion(request)
        print(answer)


if __name__ == '__main__':
    pytest.main([__file__])
