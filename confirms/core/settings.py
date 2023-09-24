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
from pathlib import Path

import openai
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Optional, ClassVar
from dotenv import load_dotenv

# Load additional environment variables from .env file during import of this module
load_dotenv()

# Set OpenAI key explicitly because it does not automatically load the variables set by load_dotenv()
openai.api_key = os.getenv("OPENAI_API_KEY")


@dataclass(init=False)
class Settings:
    """Default settings may be modified before the settings object is passed to the model."""

    def __init__(self):
        """Set default values."""

        # Check environment variable first
        self.model_dir = os.getenv("CONFIRMS_MODEL_DIR")

        # If not set, use project_root/models
        if self.model_dir is None:
            self.model_dir = os.path.join(Path(os.path.dirname(__file__)).parent.parent, "models")
            
        if not os.path.exists(self.model_dir):
            RuntimeError(f"Model directory {self.model_dir} does not exist.")
        if not os.path.isdir(self.model_dir):
            RuntimeError(f"Path specified for model directory {self.model_dir} is not a directory.")

    model_dir: str = field(default=None)
    """Models are located in model_dir/model_name where model_name is either filename or directory name."""

    def get_model_path(self, model_name: str) -> str:
        """Get model path from model name using model_dir or its default value project_root/models,
        and check that it exists.

        Models are located in model_dir/model_name where model_name is either filename or directory name,
        where model_dir is CONFIRMS_MODEL_DIR environment variable or project_root/models when not set.
        
        - When model is a file, model_name should include extension
        - When model is a file, it should not
        """
        model_path = os.path.join(self.model_dir, model_name)
        if not os.path.exists(model_path):
            raise RuntimeError(f"Model path {model_path} does not exist.")
        return model_path