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

import openai

from confirms.core.llm.llm import Llm


@dataclass
class GptNativeLlm(Llm):
    """GPT model family using native OpenAI API."""

    temperature: float = field(default=None)
    """Model temperature (note that for GPT models zero value does not mean reproducible answers)."""

    _llm: bool = field(default=None)

    def load_model(self):
        """Load model after fields have been set."""

        # Skip if already loaded
        if self._llm is None:

            gpt_model_types = ["gpt-3.5-turbo", "gpt-4"]
            if self.model_type not in gpt_model_types:
                raise RuntimeError(f"GPT Native LLM model type {self.model_type} is not recognized. "
                                   f"Valid model types are {gpt_model_types}")

            # Native OpenAI API calls are stateless. This means no object is needed at this time.
            self._llm = True

    def completion(self, question: str, *, prompt: Optional[str] = None) -> str:
        """Simple completion with optional prompt."""

        if prompt is not None:
            messages = [{"role": "system", "content": prompt}]
        else:
            messages = []

        messages = messages + [{"role": "user", "content": question}]

        response = openai.ChatCompletion.create(
            model=self.model_type,
            messages=messages
        )
        answer = response['choices'][0]['message']['content']
        return answer


