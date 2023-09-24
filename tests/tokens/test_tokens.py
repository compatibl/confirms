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
import tiktoken
from confirms.core.settings import Settings


def test_smoke():
    """Test tokens for common and uncommon words."""

    settings = Settings()
    enc = tiktoken.get_encoding("cl100k_base")

    # Convert to numerical tokens
    text = "Hello, world! Hello, QWERTY!"
    tokens_num = enc.encode(text)
    restored = enc.decode(tokens_num)
    assert restored == text

    # Safely convert to token bytes, use decode_single_token_bytes()
    # because decode() will be lossy if tokens are not on UTF-8 boundaries.
    token_bytes = [enc.decode_single_token_bytes(token) for token in tokens_num]
    assert token_bytes[0] == b'Hello'


if __name__ == '__main__':
    pytest.main([__file__])
