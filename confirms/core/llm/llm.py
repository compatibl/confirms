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

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Optional


@dataclass
class Llm(ABC):
    """Base class for the core LLM."""

    model_id: str = field(default=None)
    """Identifies LLM type and settings."""

    model_type: str = field(default=None)
    """LLM type in the format accepted by the vendor API (e.g. GPT-4) or name of the file from which 
    the LLM is loaded including extension (e.g. llama-2-13b-chat.Q4_K_M.gguf)."""

    @abstractmethod
    def load_model(self):
        """Load model after fields have been set."""

    @abstractmethod
    def completion(self, question: str, *, prompt: Optional[str] = None) -> str:
        """Simple completion with optional prompt."""

