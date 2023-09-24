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

prompt = """Using this context ```["Min value is less than max value", "Max value is obtained externally"]```, 
comment the code below. All comments are on a separate line from code and should match the context exactly.
Do not try to comment the code with new text, instead find the most suitable part of the context. 
```
x = get_max_value()
y = x-1
```
"""
