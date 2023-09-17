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

from dataclasses import dataclass, field
from typing import Optional
from langchain import OpenAI
from confirms.core.llm.llm import Llm


@dataclass
class GptLangChainLlm(Llm):
    """GPT LLM family."""

    temperature: float = field(default=None)
    """Model temperature (note that for GPT models zero value does not mean reproducible answers)."""

    _llm: OpenAI = field(default=None)

    def load_model(self):
        """Load model after fields have been set."""

        # Skip if already loaded
        if self._llm is None:

            gpt_model_types = ["gpt-3.5-turbo", "gpt-4"]
            if self.model_type not in gpt_model_types:
                raise RuntimeError(f"GPT LLM model type {self.model_type} is not recognized. "
                                   f"Valid model types are {gpt_model_types}")

            self._llm = OpenAI(
                model_name=self.model_type,
                temperature=self.temperature if self.temperature is not None else 0.0
            )

    def completion(self, question: str, *, prompt: Optional[str] = None) -> str:
        """Simple completion with optional prompt."""

        # Load model (multiple calls do not need to reload)
        self.load_model()

        # TODO - No prompt yet
        answer = self._llm(question)
        return answer
