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

import datetime as dt
from dataclasses import dataclass, field
from typing import List

from confirms.core.schedule.interest_schedule import InterestSchedule


@dataclass
class FloatingInterestSchedule(InterestSchedule):
    """Floating interest schedule adds coupon reset dates to InterestSchedule."""

    unadj_start: List[dt.date] = field(default=None)
    """List of unadjusted accrual start dates."""

    unadj_end: List[dt.date] = field(default=None)
    """List of unadjusted accrual end dates."""

    adj_start: List[dt.date] = field(default=None)
    """List of adjusted accrual start dates."""

    adj_end: List[dt.date] = field(default=None)
    """List of adjusted accrual end dates."""
