# coding=utf-8
# Copyright 2022 The TensorFlow Datasets Authors.
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

"""Dummy dataset self-contained in a directory."""
from typing import Dict

from etils import epath
import numpy as np

import tensorflow_datasets.public_api as tfds
from tensorflow_datasets.vision_language.laion.laion400m_dataset_builder import Laion400mDatasetBuilder


class DummyLaion400mDataset(Laion400mDatasetBuilder, skip_registration=True):
  """Dummy class for tests."""
  VERSION = tfds.core.Version('1.0.0')
  RELEASE_NOTES = {'1.0.0': 'Dummy notes.'}
  SHARD_NUM = 1
  EXAMPLES_IN_SHARD = []  # overwrite in tests

  def _info(self) -> tfds.core.DatasetInfo:
    """Returns the dataset metadata."""
    return self.create_dataset_info(
        ordered_features={'dummy': tfds.features.Scalar(dtype=np.int32)})

  def _download_data(
      self, dl_manager: tfds.download.DownloadManager) -> Dict[str, epath.Path]:
    """Downloads data."""
    return {}

  def _generate_examples_one_shard(self,
                                   dl_manager: tfds.download.DownloadManager,
                                   file_name_to_dl_path: Dict[str, epath.Path],
                                   shard_idx: int):
    """Yields examples from a single shard."""
    for idx, item in enumerate(self.EXAMPLES_IN_SHARD):
      key = str(idx)
      example = {
          'dummy': item,
          'caption': '',
          'nsfw': 'NSFW',
          'similarity': 0.0,
          'license': '',
          'url': '',
          'original_width': 0,
          'original_height': 0,
      }
      yield (key, example)
