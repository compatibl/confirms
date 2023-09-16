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

import os

import pytest
from huggingface_hub import hf_hub_download
from langchain import LlamaCpp
from langchain.callbacks import StreamingStdOutCallbackHandler
from langchain.callbacks.manager import CallbackManager

_model_names = []


def get_llm() -> LlamaCpp:
    """Create LLM instance."""

    repo_id = "TheBloke/Llama-2-13B-chat-GGUF"
    model_name = "llama-2-13b-chat.Q4_K_M"
    model_filename = f"{model_name}.gguf"
    model_dir = os.path.join(os.path.dirname(__file__), "../../downloads")
    model_path = os.path.join(model_dir, model_filename)
    if not os.path.exists(model_path):
        print(f"Model file {model_filename} not found in `project_root/downloads` directory,"
              f"downloading from Hugging Face. This can some time depending on network speed.")
        hf_hub_download(
            repo_id=repo_id,
            filename=model_filename,
            local_dir=model_dir,
            local_dir_use_symlinks=False
        )

    callback_manager = CallbackManager([StreamingStdOutCallbackHandler()])
    llm = LlamaCpp(
        model_path=model_path,
        temperature=0.0,
        max_tokens=2000,
        top_p=0.95,
        top_k=40,
        repetition_penalty=1.1,
        last_n_tokens=64,
        seed=-1,
        max_new_tokens=1024,
        reset=True,
        batch_size=8,
        context_length=-1,
        stop=None,
        callback_manager=callback_manager,
        verbose=True,  # Verbose is required to pass to the callback manager
    )
    return llm


def test_llm():
    """Run a simple test to ensure the model is available."""

    llm = get_llm()
    output = llm("What is two times two?")
    assert output == "\n\nAnswer: Two times two is equal to 4."


if __name__ == '__main__':
    pytest.main([__file__])
