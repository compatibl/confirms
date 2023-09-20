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
from dataclasses import dataclass, field
from typing import Optional

from huggingface_hub import hf_hub_download
from langchain import LlamaCpp, PromptTemplate, LLMChain
from langchain.callbacks import StreamingStdOutCallbackHandler
from langchain.callbacks.manager import CallbackManager
from confirms.core.llm.llm import Llm


@dataclass
class LlamaLangChainLlm(Llm):
    """LLAMA 2 models loaded using LangCpp adapter."""

    temperature: float = field(default=None)
    """Model temperature (note that for GPT models zero value does not mean reproducible answers)."""

    seed: int = field(default=None)
    """Model seed (use the same seed to reproduce the answer)."""

    grammar_file: str = field(default=None)
    """Grammar filename including extension located in project_root/grammar directory."""

    _llm: LlamaCpp = field(default=None)

    def load_model(self, grammar_name: str = None):
        """Load model after fields have been set."""

        # Skip if already loaded
        if self._llm is None:

            model_filename = self.model_type
            if model_filename.startswith("llama-2-7b-chat."):
                repo_id = "TheBloke/Llama-2-7B-chat-GGUF"
            elif model_filename.startswith("llama-2-13b-chat."):
                repo_id = "TheBloke/Llama-2-13B-chat-GGUF"
            elif model_filename.startswith("llama-2-70b-chat."):
                repo_id = "TheBloke/Llama-2-70B-chat-GGUF"
            else:
                raise RuntimeError(f"Repo not specified for model type {self.model_type}")

            model_dir = os.path.join(os.path.dirname(__file__), "../../../downloads")
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

            if self.grammar_file is not None:
                grammar_dir = os.path.join(os.path.dirname(__file__), "../../../grammar")
                grammar_path = os.path.join(grammar_dir, self.grammar_file)
            else:
                grammar_path = None

            callback_manager = CallbackManager([StreamingStdOutCallbackHandler()])
            self._llm = LlamaCpp(
                model_path=model_path,
                temperature=0.0,
                n_gpu_layers=0,  # Change to the value appropriate for the GPU RAM size if using GPU
                max_tokens=2000,
                top_p=0.95,
                top_k=40,
                repeat_penalty=1.1,  # This is the default
                last_n_tokens_size=64,
                seed=self.seed if self.seed is not None else -1,
                n_batch=8,  # This is the default
                n_ctx=512,  # This is the default, context window -- check if prev value was correct
                stop=None,
                grammar_path=grammar_path,
                callback_manager=callback_manager,
                verbose=True,  # Verbose is required to pass to the callback manager
            )

    def completion(self, question: str, *, prompt: Optional[PromptTemplate] = None) -> str:
        """Simple completion with optional prompt."""

        # Load model (multiple calls do not need to reload)
        self.load_model()

        if prompt is None:
            answer = self._llm(question)
        else:
            llm_chain = LLMChain(prompt=prompt, llm=self._llm)
            answer = llm_chain.run(question)
        return answer
