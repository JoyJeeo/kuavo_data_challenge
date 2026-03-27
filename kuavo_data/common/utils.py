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

import json
import os
import shutil
import subprocess
import rosbag
from pathlib import Path
from typing import Any

def load_json(fpath: Path) -> Any:
    with open(fpath) as f:
        return json.load(f)


def write_json(data: dict, fpath: Path) -> None:
    fpath.parent.mkdir(exist_ok=True, parents=True)
    with open(fpath, "w") as f:
        json.dump(data, f, indent=4, ensure_ascii=False)


def reindex_rosbag(bag_file)->str:
    bag_file = str(bag_file)
    try:
        with rosbag.Bag(bag_file, 'r') as bag:
            return bag_file
    except rosbag.bag.ROSBagException as e:
        print(f"Error reading '{bag_file}': {e}")
    
    # bag is corrupted.
    try:
        print(f"Warning: The bag file '{bag_file}' is corrupted, reindexing...")
        command = [
            "rosbag",
            "reindex",
            bag_file
            ]
        subprocess.run(command, check=True)
        if bag_file.endswith(".bag.active"):
            base_name = bag_file.replace(".bag.active", "")
            rosbag_orig_file = f"{base_name}.bag.orig.active"
        elif bag_file.endswith(".bag"):
            base_name = bag_file.replace(".bag", "")
            rosbag_orig_file = f"{bag_file}.orig.bag"
        if os.path.exists(rosbag_orig_file):
            os.remove(rosbag_orig_file)
        if os.path.exists(bag_file):
            rosbag_file = f"{base_name}.bag"
            shutil.move(bag_file, rosbag_file)
            return rosbag_file
        return None
    except subprocess.CalledProcessError as e:
        print(f"Error reindexing bag file: {e}")
        return None
    
