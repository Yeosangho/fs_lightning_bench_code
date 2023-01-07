import os
import argparse
import pandas as pd
from tqdm import tqdm
import torch
import torch.nn as nn
import torch.distributed as dist 
from torch.utils.data import Dataset, DataLoader, random_split, RandomSampler, SequentialSampler
from torch.utils.data.distributed import DistributedSampler
from torch.profiler import profile, record_function, ProfilerActivity

from torchvision import datasets
from torchvision import transforms
import torchvision.models as models
from torchvision.models.resnet import BasicBlock, Bottleneck, ResNet

from transformers import GPT2LMHeadModel,  GPT2Tokenizer, GPT2Config, GPT2LMHeadModel
from fairscale.optim.oss import OSS

from auto_wrap_custom import wrap, enable_wrap, auto_wrap
#from fairscale.nn.wrap import wrap, enable_wrap, auto_wrap
from fairscale.nn.data_parallel import FullyShardedDataParallel as FSDP
from fairscale.nn.data_parallel import ShardedDataParallel as SDP



def get_args_or_env(env_key_name, args_key_name, args):
    value = os.environ.get(env_key_name, None)
    if value is None :
        value = vars(args).get(args_key_name, None)
    return value 


def set_env(target_env_key_name, args, source_env_key_name=None, source_args_key_name=None):

    if os.environ.get(target_env_key_name, None) is None :
        if source_env_key_name is not None :
            env_val = os.environ.get(source_env_key_name, None)
            if env_val is not None:
                print(target_env_key_name)
                print(env_val)				
                os.environ[target_env_key_name] = env_val
        if source_args_key_name is not None:
            args_val = vars(args).get(source_args_key_name, None)
            if args_val is not None:
                print(target_env_key_name)
                print(args_val)
                os.environ[target_env_key_name] = args_val


if __name__ == '__main__':

    
    parser = argparse.ArgumentParser()
    parser.add_argument('--target_memory', default=7.0, type=float)
    parser.add_argument('--sdp_ratio', default=0, type=float)
    parser.add_argument('--fsdp_ratio', default=0, type=float)
    parser.add_argument('--dp_ratio', default=0, type=float)
    parser.add_argument('--bucket_size', default=20, type=float)
    parser.add_argument('--exp_tag', type=str)
    parser.add_argument("--world_size", type=int, default=2)
    parser.add_argument("--rank", type=int, default=0)
    parser.add_argument("--master_addr", type=str, default="210.107.197.218")
    parser.add_argument("--master_port", type=str, default="30002")
    parser.add_argument("--profile", type=str, default="false")
    args = parser.parse_args()

    world_size = int(get_args_or_env("WORLD_SIZE", "world_size", args))
    rank = int(get_args_or_env("SLURM_PROCID", "rank", args))

    set_env('MASTER_PORT', args, source_env_key_name='TRAINER_PORT', source_args_key_name='master_port')
    set_env("MASTER_ADDR", args, source_args_key_name='master_addr',)

    dist.init_process_group(backend='nccl', world_size=world_size, rank=rank)

    ngpus_per_node = torch.cuda.device_count()
    device_id = rank%ngpus_per_node
    torch.cuda.set_device(device_id)



    train_dataset = datasets.CIFAR10(
            root='../shardscheduler/cifar10-data', train=True, download=False, transform=transforms.ToTensor())
    train_sampler = DistributedSampler(dataset=train_dataset, shuffle=False)
    train_loader = torch.utils.data.DataLoader(
            train_dataset , batch_size=32, sampler=train_sampler, shuffle=False, num_workers=2)
    model = ResNet(Bottleneck,  [3, 4, 6, 3]) #it means "resnet50 model"
    model.cuda()
    model.train()

    model_parameter_names = {}
    fsdp_params = dict(wrapper_cls=FSDP, flatten_parameters=True, reshard_after_forward=False, bucket_cap_mb=50)
    fsdp_params_no_cls = dict( flatten_parameters=True, reshard_after_forward=False, bucket_cap_mb=50)

    with enable_wrap(**fsdp_params_no_cls):
        sharded_module = auto_wrap(model)
        sharded_module = FSDP(sharded_module, **fsdp_params_no_cls)
        sharded_module._lazy_init()
    optimizer = torch.optim.Adam(sharded_module.parameters(), lr=0.001, betas=(0.9, 0.999), eps=1e-08, weight_decay=0, amsgrad=False)
    criterion = nn.CrossEntropyLoss()
    iter_count = 0
    sharded_module.train()
    with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA]) as prof:
        with record_function("test_model"):	
            count =0
            for data, target in tqdm(train_loader):

                data = data.cuda()
                target = target.cuda()
                output = sharded_module(data)

                loss = criterion(output, target)

                loss.backward()
                optimizer.step()
                optimizer.zero_grad()
                count += 1
                if(count ==5):
                    break
    if(rank == 0):		
        prof.export_chrome_trace("trace.json")                       
