# Copyright 2021 Huawei Technologies Co., Ltd
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ============================================================================
"""
predict.
"""
import os
import argparse
from mindspore import context
from mindspore.train.model import Model
from mindspore.common import set_seed
from mindspore.train.serialization import load_checkpoint, load_param_into_net
from src.dataset import create_dataset
from src.config import config
from src.mobilenetv3 import mobilenet_v3_small

parser = argparse.ArgumentParser(description='Image classification')
# Ascend parameter
parser.add_argument('--dataset_path', type=str, default=None, help='Dataset path')
parser.add_argument('--checkpoint_path', type=str, default=None, help='Checkpoint file path')
parser.add_argument('--device_id', type=int, default=0, help='Device id')

args_opt = parser.parse_args()

set_seed(1)

if __name__ == '__main__':

    context.set_context(mode=context.GRAPH_MODE, device_target='Ascend', save_graphs=False)
    net = mobilenet_v3_small(num_classes=config.num_classes, multiplier=1.)

    
   
    dataset = create_dataset(dataset_path=args_opt.dataset_path,
                                do_train=False,
                                batch_size=config.batch_size)
    param_dict = load_checkpoint(args_opt.checkpoint_path)
    step_size = dataset.get_dataset_size()
    load_param_into_net(net, param_dict)
    net.set_train(False)

    # 读取图片并进行标准化
    dataset = dataset.map(operations=net._apply_eval_eval(), input_columns=["image"])
    dataset = dataset.batch(1, drop_remainder=True)
    dataset = dataset.repeat(1)

    # 定义model
    model = Model(net)

    # 预测
    result = model.predict(dataset)
    print(result) 


