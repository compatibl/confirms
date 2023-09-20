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

from confirms.core.llm.gpt_native_llm import GptNativeLlm
from confirms.core.llm.llama_lang_chain_llm import LlamaLangChainLlm


def test_sally_riddle():
    """Test for solving Sally and her siblings riddle."""

    model_types = ["gpt-3.5-turbo", "gpt-4", "llama-2-7b-chat.Q4_K_M.gguf", "llama-2-13b-chat.Q4_K_M.gguf"]
    for model_type in model_types:
        # TODO - implement prompt
        question = ("```Sally has three brothers. Each of Sally's brothers has two sisters.```")
        request = (
            "<s>[INST] Pay attention and remember information below, which will help to answer the question or imperative after the context ends. "
            f"Context: {question}. According to only the information in the document sources provided within the context above, how many sisters does Sally have? [/INST]")

        if model_type.startswith("llama"):
            llm = LlamaLangChainLlm(model_type=model_type, temperature=0.2)  # , grammar_file="frequency.gbnf")
        elif model_type.startswith("gpt"):
            llm = GptNativeLlm(model_type=model_type, temperature=0.2)
        else:
            raise RuntimeError(f"Unknown model type: {model_type}")

        answer = llm.completion(request)
        print(answer)


if __name__ == '__main__':
    pytest.main([__file__])
