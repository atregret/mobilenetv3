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
"""train_imagenet in qizhi."""

import os
import ast
import argparse
import time
import json
from mindspore import context
from mindspore import Tensor

from mindspore.nn import RMSProp
from mindspore.nn.loss import SoftmaxCrossEntropyWithLogits
from mindspore.train.model import Model
from mindspore.context import ParallelMode
from mindspore.train.callback import ModelCheckpoint, CheckpointConfig
from mindspore.train.loss_scale_manager import FixedLossScaleManager
from mindspore.train.serialization import load_checkpoint, load_param_into_net
from mindspore.common import set_seed
from mindspore.communication.management import init

from src.dataset import create_dataset
from src.lr_generator import get_lr
from src.config import config
from src.loss import CrossEntropyWithLabelSmooth
from src.monitor import Monitor
from src.mobilenetv3 import mobilenet_v3_small

import moxing as mox




set_seed(1)

parser = argparse.ArgumentParser(description='Image classification')
# modelarts parameter
parser.add_argument('--data_url', type=str, default=None, help='Dataset path')
parser.add_argument('--train_url', type=str, default=None, help='Train output path')
parser.add_argument('--pretrain_url',help='pre_train_model path in obs') 
parser.add_argument('--multi_data_url',help='path to multi dataset',default= '/cache/data/')
parser.add_argument('---device_target',default="Ascend",type=str,help='device target')
# Ascend parameter
parser.add_argument('--dataset_path', type=str, default=None, help='Dataset path')
parser.add_argument('--run_distribute', type=ast.literal_eval, default=False, help='Run distribute')
parser.add_argument('--device_id', type=int, default=0, help='Device id')

parser.add_argument('--run_modelarts', type=ast.literal_eval, default=False, help='Run mode')
parser.add_argument('--pre_trained', type=str, default=None, help='Pretrained checkpoint path')
args_opt = parser.parse_args()

#context.set_context(mode=context.GRAPH_MODE, device_target=config.device, save_graphs=False)


### Copy multiple datasets from obs to training image and unzip###  
def C2netMultiObsToEnv(multi_data_url, data_dir):
    #--multi_data_url is json data, need to do json parsing for multi_data_url
    multi_data_json = json.loads(multi_data_url)  
    print("multi_data_json:",multi_data_json)
    for i in range(len(multi_data_json)):
        zipfile_path = data_dir + "/" + multi_data_json[i]["dataset_name"]
        try:
            mox.file.copy(multi_data_json[i]["dataset_url"], zipfile_path) 
            print("Successfully Download {} to {}".format(multi_data_json[i]["dataset_url"],zipfile_path))
            #get filename and unzip the dataset
            filename = os.path.splitext(multi_data_json[i]["dataset_name"])[0]
            filePath = data_dir + "/" + filename
            if not os.path.exists(filePath):
                os.makedirs(filePath)
            os.system("unzip {} -d {}".format(zipfile_path, filePath))

        except Exception as e:
            print('moxing download {} to {} failed: '.format(
                multi_data_json[i]["dataset_url"], zipfile_path) + str(e))
    #Set a cache file to determine whether the data has been copied to obs. 
    #If this file exists during multi-card training, there is no need to copy the dataset multiple times.
    f = open("/cache/download_dataset_input.txt", 'w')    
    f.close()
    try:
        if os.path.exists("/cache/download_dataset_input.txt"):
            print("download_input succeed")
    except Exception as e:
        print("download_dataset_input failed")
    return 
### Copy ckpt file from obs to training image###
### To operate on folders, use mox.file.copy_parallel. If copying a file. 
### Please use mox.file.copy to operate the file, this operation is to operate the file

def ObsUrlToEnv(obs_ckpt_url, ckpt_url):
    try:
        mox.file.copy(obs_ckpt_url, ckpt_url)
        print("Successfully Download {} to {}".format(obs_ckpt_url,ckpt_url))
    except Exception as e:
        print('moxing download {} to {} failed: '.format(obs_ckpt_url, ckpt_url) + str(e)) 
    return      
### Copy the output model to obs ###  

def EnvToObs(train_dir, obs_train_url):
    try:
        mox.file.copy_parallel(train_dir, obs_train_url)
        print("Successfully Upload {} to {}".format(train_dir,
                                                    obs_train_url))
    except Exception as e:
        print('moxing upload {} to {} failed: '.format(train_dir,
                                                    obs_train_url) + str(e))
    return   

### Copy multiple pretrain file from obs to training image and unzip###  

def C2netModelToEnv(model_url, model_dir):
    #--ckpt_url is json data, need to do json parsing for ckpt_url_json
    model_url_json = json.loads(model_url)  
    print("model_url_json:",model_url_json)
    for i in range(len(model_url_json)):
        modelfile_path = model_dir + "/" + "checkpoint.ckpt"
        try:
            mox.file.copy(model_url_json[i]["model_url"], modelfile_path) 
            print("Successfully Download {} to {}".format(model_url_json[i]["model_url"],modelfile_path))
        except Exception as e:
            print('moxing download {} to {} failed: '.format(
                model_url_json[i]["model_url"], modelfile_path) + str(e))
    return                                                         

def DownloadDataFromQizhi(multi_data_url, data_dir):
    device_num = int(os.getenv('RANK_SIZE'))
    if device_num == 1:
        C2netMultiObsToEnv(multi_data_url,data_dir)
        context.set_context(mode=context.GRAPH_MODE,device_target=args_opt.device_target)
    if device_num > 1:
        # set device_id and init for multi-card training
        context.set_context(mode=context.GRAPH_MODE, device_target=args_opt.device_target, device_id=int(os.getenv('ASCEND_DEVICE_ID')))
        context.reset_auto_parallel_context()
        context.set_auto_parallel_context(device_num = device_num, parallel_mode=ParallelMode.DATA_PARALLEL, gradients_mean=True, parameter_broadcast=True)
        init()
        #Copying obs data does not need to be executed multiple times, just let the 0th card copy the data
        local_rank=int(os.getenv('RANK_ID'))
        if local_rank%8==0:
            C2netMultiObsToEnv(multi_data_url,data_dir)
        #If the cache file does not exist, it means that the copy data has not been completed,
        #and Wait for 0th card to finish copying data
        while not os.path.exists("/cache/download_dataset_input.txt"):
            time.sleep(1)  
    return

def DownloadModelFromQizhi(model_url, model_dir):
    device_num = int(os.getenv('RANK_SIZE'))
    if device_num == 1:
        C2netModelToEnv(model_url,model_dir)
    if device_num > 1:
        #Copying obs data does not need to be executed multiple times, just let the 0th card copy the data
        local_rank=int(os.getenv('RANK_ID'))
        if local_rank%8==0:
            C2netModelToEnv(model_url,model_dir)
    return    

def UploadToQizhi(train_dir, obs_train_url):
    device_num = int(os.getenv('RANK_SIZE'))
    local_rank = int(os.getenv('RANK_ID'))
    if device_num == 1:
        EnvToObs(train_dir, obs_train_url)
    if device_num > 1:
        if local_rank % 8 == 0:
            EnvToObs(train_dir, obs_train_url)
    return



if __name__ == '__main__':
    # init distributed
    if args_opt.run_modelarts:
        ######
        import moxing as mox
        device_id = int(os.getenv('DEVICE_ID'))
        device_num = int(os.getenv('RANK_SIZE'))
        context.set_context(device_id=device_id)
        local_data_url = '/cache/data'
        local_train_url = '/cache/ckpt'
        if device_num > 1:
            init()
            context.set_auto_parallel_context(device_num=device_num, parallel_mode='data_parallel', gradients_mean=True)
            local_data_url = os.path.join(local_data_url, str(device_id))
        mox.file.copy_parallel(args_opt.data_url, local_data_url)
    else:
        if args_opt.run_distribute:
            device_id = int(os.getenv('DEVICE_ID'))
            device_num = int(os.getenv('RANK_SIZE'))
            context.set_context(device_id=device_id)
            init()
            context.reset_auto_parallel_context()
            context.set_auto_parallel_context(device_num=device_num,
                                              parallel_mode=ParallelMode.DATA_PARALLEL,
                                              gradients_mean=True)
        else:
            context.set_context(device_id=args_opt.device_id)
            device_num = 1
            device_id = 0

    data_dir = '/cache/data'
    train_dir = '/cache/output'
    model_dir = '/cache/pretrain'
    try:
        if not os.path.exists(data_dir):
            os.makedirs(data_dir)
        if not os.path.exists(train_dir):
            os.makedirs(train_dir)
        if not os.path.exists(model_dir):
            os.makedirs(model_dir)
    except Exception as e:
        print("path already exists")

    ###把模型的url链接，下载到本地model_dir目录下
    DownloadModelFromQizhi(args_opt.pretrain_url, model_dir)
      
    ###把数据集的url链接，下载到本地data_dir目录下
    DownloadDataFromQizhi(args_opt.multi_data_url, data_dir)


    # define net
    net = mobilenet_v3_small(num_classes=config.num_classes, multiplier=1.)

    # define loss
    if config.label_smooth > 0:
        loss = CrossEntropyWithLabelSmooth(
            smooth_factor=config.label_smooth, num_classes=config.num_classes)
    else:
        loss = SoftmaxCrossEntropyWithLogits(sparse=True, reduction='mean')

    # define dataset
    if args_opt.run_modelarts:
        dataset = create_dataset(dataset_path=local_data_url,
                                 do_train=True,
                                 batch_size=config.batch_size,
                                 device_num=device_num, rank=device_id)
    else:
        dataset = create_dataset(dataset_path=data_dir+'/imagenet/imagenet/train',
                                 do_train=True,
                                 batch_size=config.batch_size,
                                 device_num=device_num, rank=device_id)
    step_size = dataset.get_dataset_size()

    # resume
    if args_opt.pre_trained:
        param_dict = load_checkpoint(args_opt.pre_trained)
        load_param_into_net(net, param_dict)

    # define optimizer
    loss_scale = FixedLossScaleManager(
        config.loss_scale, drop_overflow_update=False)
    lr = Tensor(get_lr(global_step=0,
                       lr_init=0,
                       lr_end=0,
                       lr_max=config.lr,
                       warmup_epochs=config.warmup_epochs,
                       total_epochs=config.epoch_size,
                       steps_per_epoch=step_size))
    opt = RMSProp(net.trainable_params(), learning_rate=lr, decay=0.9, weight_decay=config.weight_decay,
                  momentum=config.momentum, epsilon=0.001, loss_scale=config.loss_scale)

    # define model
    model = Model(net, loss_fn=loss, optimizer=opt,
                  loss_scale_manager=loss_scale, amp_level='O3')

    # define callbacks
    cb = [Monitor(lr_init=lr.asnumpy())]
    if config.save_checkpoint and (device_num == 1 or device_id == 0):
        config_ck = CheckpointConfig(save_checkpoint_steps=config.save_checkpoint_epochs * step_size,
                                     keep_checkpoint_max=config.keep_checkpoint_max)
        if args_opt.run_modelarts:
            ckpt_cb = ModelCheckpoint(prefix="mobilenetV3", directory=local_train_url, config=config_ck)
        else:
            save_ckpt_path = os.path.join(config.save_checkpoint_path, 'model_' + str(device_id) + '/')
            ckpt_cb = ModelCheckpoint(prefix="mobilenetV3", directory=save_ckpt_path, config=config_ck)
        cb += [ckpt_cb]

    # begine train
    model.train(config.epoch_size, dataset, callbacks=cb, dataset_sink_mode=True)

    ######### modelarts upload ##########
    if args_opt.run_modelarts and config.save_checkpoint and (device_num == 1 or device_id == 0):
        mox.file.copy_parallel(local_train_url, args_opt.train_url)
