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

"""Tests for laion400m_dataset_builder."""
import tensorflow_datasets.public_api as tfds

_EXAMPLES_IN_SHARD = [1, 2, 3]


class TestDummyLaion400mDataset(tfds.testing.DatasetBuilderTestCase):
  SPLITS = {'train': len(_EXAMPLES_IN_SHARD)}

  @classmethod
  def setUpClass(cls):
    from tensorflow_datasets.vision_language.laion.dummy_laion400m_dataset import DummyLaion400mDataset  # pylint: disable=g-import-not-at-top
    cls.DATASET_CLASS = DummyLaion400mDataset
    cls.DATASET_CLASS.EXAMPLES_IN_SHARD = _EXAMPLES_IN_SHARD
    super().setUpClass()

  def test_registered(self):
    # Custom datasets shouldn't be registered
    self.assertNotIn(tfds.ImageFolder.name, tfds.list_builders())


if __name__ == '__main__':
  tfds.testing.test_main()
