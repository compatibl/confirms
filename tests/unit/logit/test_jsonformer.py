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
from jsonformer import Jsonformer
from transformers import AutoModelForCausalLM, AutoTokenizer
from confirms.core.settings import Settings


def test_smoke():
    """Run a smoke test for Jsonformer."""

    settings = Settings()
    model_name = "Llama-2-7b-chat-hf"
    model_path = settings.get_model_path(model_name)

    model = AutoModelForCausalLM.from_pretrained(model_path)
    tokenizer = AutoTokenizer.from_pretrained(model_path)

    json_schema = {
        "type": "object",
        "properties": {
            "name": {"type": "string"},
            "age": {"type": "number"},
            "is_student": {"type": "boolean"},
            "courses": {
                "type": "array",
                "items": {"type": "string"}
            }
        }
    }

    prompt = "Generate a person's information based on the following schema:"
    jsonformer = Jsonformer(model, tokenizer, json_schema, prompt)
    generated_data = jsonformer()

    print(generated_data)


if __name__ == '__main__':
    pytest.main([__file__])
