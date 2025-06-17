# Copyright (c) 2022, NVIDIA CORPORATION. All rights reserved.
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

#NOTE: Train and Test sets combined for training purposes right now. Will need to switch them at a later point

import os
import pandas as pd
import horovod.tensorflow as hvd
from runtime.utils import get_config_file, is_main_process
from sklearn.model_selection import KFold

from data_loading.dali_loader import fetch_dali_loader
from data_loading.utils import get_path, get_split, get_test_fnames, load_data


class DataModule:
    def __init__(self, args):
        super().__init__()
        self.args = args
        if args.use_hvd == True:
            self.num_gpus = hvd.size()
        else:
            self.num_gpus = args.gpus
        self.train_imgs = []
        self.train_lbls = []
        self.train_dist_maps = []
        self.val_imgs = []
        self.val_lbls = []
        self.val_dist_maps = []
        self.test_imgs = []
        #self.kfold = KFold(n_splits=self.args.nfolds, shuffle=True, random_state=12345)
        self.data_path = get_path(args)
        #configs = get_config_file(self.args)
        configs = {'patch_size': [128]*3, 'spacings': [1.0]*3, 'n_class': 1, 'in_channels': 1}
        self.patch_size = configs["patch_size"]
        self.kwargs = {
            "dim": self.args.dim,
            "patch_size": self.patch_size,
            "seed": self.args.seed,
            "gpus": self.num_gpus,
            "num_workers": self.args.num_workers,
            "oversampling": self.args.oversampling,
            "benchmark": self.args.benchmark,
            "nvol": self.args.nvol,
            "bench_steps": self.args.bench_steps,
            "meta": load_data(self.data_path, "*_meta.npy"),
        }

    def setup(self, stage=None):
        csv_file_path = os.path.join(self.data_path, '_dataset_split.csv')
        if os.path.exists(csv_file_path):
            df_csv = pd.read_csv(csv_file_path, index_col=0)
            #
            #can optionally combine train and test for time being to increase training set size
            train_names = list(df_csv['Patient_Name'][df_csv['Fold_' + str(self.args.fold)] == 0])
            #train_names = list(df_csv['Patient_Name'][(df_csv['Fold_' + str(self.args.fold)] == 0) | (df_csv['Fold_' + str(self.args.fold)] == 2)])
            #
            val_names = list(df_csv['Patient_Name'][df_csv['Fold_' + str(self.args.fold)] == 1])
            test_names = list(df_csv['Patient_Name'][df_csv['Fold_' + str(self.args.fold)] == 2])
            #
            self.train_imgs = sorted([train_name + 'x.npy' for train_name in train_names])
            self.train_lbls = sorted([train_name + 'y.npy' for train_name in train_names])
            self.train_dist_maps = sorted([train_name + 'z.npy' for train_name in train_names])
            self.val_imgs = sorted([val_name + 'x.npy' for val_name in val_names])
            self.val_lbls = sorted([val_name + 'y.npy' for val_name in val_names])
            self.val_dist_maps = sorted([val_name + 'z.npy' for val_name in val_names])
            self.test_imgs = sorted([test_name + 'x.npy' for test_name in test_names])
            self.test_lbls = sorted([test_name + 'y.npy' for test_name in test_names])
            self.test_dist_maps = sorted([test_name + 'z.npy' for test_name in test_names])
        else:
            self.train_imgs = load_data(os.path.join(self.data_path, 'Train/'), "*_x.npy")
            self.train_lbls = load_data(os.path.join(self.data_path, 'Train/'), "*_y.npy")
            self.train_dist_maps = load_data(os.path.join(self.data_path, 'Train/'), "*_z.npy")
            self.val_imgs = load_data(os.path.join(self.data_path, 'Val/'), "*_x.npy")
            self.val_lbls = load_data(os.path.join(self.data_path, 'Val/'), "*_y.npy")
            self.val_dist_maps = load_data(os.path.join(self.data_path, 'Val/'), "*_z.npy")
        extra_data_dir = '/autofs/cluster/qtim/users/jn85/DeepLearningExamples/TensorFlow2/Segmentation/nnUNet/data/METS_Jay_extra/'
        if os.path.exists(extra_data_dir):
            self.test_imgs = load_data(extra_data_dir, "*_x.npy")
        #self.test_imgs = self.val_imgs
        #
        if self.args.exec_mode != "predict" or self.args.benchmark:
            if is_main_process():
                ntrain, nval = len(self.train_imgs), len(self.val_imgs)
                print(f"Number of examples: Train {ntrain} - Val {nval}")

            # Shard the validation data
            if self.args.use_hvd == True:
                self.val_imgs = self.val_imgs[hvd.rank() :: hvd.size()]
                self.val_lbls = self.val_lbls[hvd.rank() :: hvd.size()]
            self.cached_val_loader = None
        elif is_main_process():
            print(f"Number of test examples: {len(self.test_imgs)}")

    def train_dataset(self):
        return fetch_dali_loader(
            self.train_imgs,
            self.train_lbls,
            self.train_dist_maps,
            self.args.batch_size,
            "train",
            self.args.use_hvd,
            **self.kwargs,
        )

    def train_size(self):
        return len(self.train_imgs)

    def val_dataset(self):
        if self.cached_val_loader is None:
            self.cached_val_loader = fetch_dali_loader(
                self.val_imgs,
                self.val_lbls,
                self.val_dist_maps,
                1,
                "eval",
                self.args.use_hvd,
                **self.kwargs)
        return self.cached_val_loader

    def val_size(self):
        return len(self.val_imgs)

    def test_dataset(self):
        if self.kwargs["benchmark"]:
            return fetch_dali_loader(
                self.train_imgs,
                self.train_lbls,
                self.train_dist_maps,
                self.args.batch_size,
                "test",
                self.args.use_hvd,
                **self.kwargs,
            )
        return fetch_dali_loader(self.test_imgs,
                                 None,
                                 None,
                                 1,
                                 "test",
                                 self.args.use_hvd,
                                 **self.kwargs)

    def test_size(self):
        return len(self.test_imgs)

    def test_fname(self, idx):
        return self.test_imgs[idx]
