# Copyright (C) 2025-2026 LejuRobotics.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
# ---
#
# This project includes code from LeRobot (https://github.com/huggingface/lerobot),
# which is licensed under the Apache License, Version 2.0.

from datetime import datetime

class Logger:
    def __init__(self, log_file):
        self.log_file = log_file
        # Clear log file on initialization
        with open(self.log_file, "w") as f:
            f.write("")

    def log(self, text):
        with open(self.log_file, "a") as f:
            f.write(f"{datetime.now().strftime('%Y-%m-%d %H:%M:%S')} - {text}\n")