import os
import argparse
import pandas as pd
from tqdm import tqdm
import torch
import torch.nn as nn
import torch.distributed as dist 
from torch.utils.data import Dataset, DataLoader, random_split, RandomSampler, SequentialSampler
from torch.utils.data.distributed import DistributedSampler

from torchvision import datasets
from torchvision import transforms
import torchvision.models as models
from torchvision.models.resnet import BasicBlock, Bottleneck, ResNet

from transformers import GPT2LMHeadModel,  GPT2Tokenizer, GPT2Config, GPT2LMHeadModel
from pytorch_lightning import Trainer
from pytorch_lightning.strategies import DDPStrategy, DDPFullyShardedStrategy
from transformers import GPT2LMHeadModel,  GPT2Tokenizer, GPT2Config, GPT2LMHeadModel
from fairscale.optim.oss import OSS
from auto_wrap import wrap, enable_wrap, auto_wrap
from fsdp_default import FullyShardedDataParallel as FSDP
from fairscale.nn.data_parallel import ShardedDataParallel as SDP
def get_reviews(review_path="/scratch/hpc72a03/review_dataset/Reviews.csv"):
    df = pd.read_csv (review_path)  
    df = df[:600]
    print(df)
    print(len(df))
    df.dropna(inplace=True)
    reviews = df.Text.copy() 
    return reviews

class GPT2Dataset(Dataset):

  def __init__(self, txt_list, tokenizer, gpt2_type="gpt2", max_length=768):

    self.tokenizer = tokenizer
    self.input_ids = []
    self.attn_masks = []

    for txt in txt_list:

      encodings_dict = tokenizer('<|startoftext|>'+ txt + '<|endoftext|>', truncation=True, max_length=max_length, padding="max_length")

      self.input_ids.append(torch.tensor(encodings_dict['input_ids']))
      self.attn_masks.append(torch.tensor(encodings_dict['attention_mask']))
    
  def __len__(self):
    return len(self.input_ids)

  def __getitem__(self, idx):
    return self.input_ids[idx], self.attn_masks[idx] 


world_size = int(os.environ["WORLD_SIZE"])
rank = int(os.environ["SLURM_PROCID"])
os.environ['MASTER_PORT'] = os.environ['TRAINER_PORT']
dist.init_process_group(backend='nccl', world_size=world_size, rank=rank)

ngpus_per_node = torch.cuda.device_count()
device_id = rank%ngpus_per_node
torch.cuda.set_device(device_id)

reviews = get_reviews()
tokenizer = GPT2Tokenizer.from_pretrained('gpt2', bos_token='<|startoftext|>', eos_token='<|endoftext|>', pad_token='<|pad|>')
train_dataset = GPT2Dataset(reviews, tokenizer)
train_loader = torch.utils.data.DataLoader(train_dataset , batch_size=2, shuffle=False, num_workers=2)
configuration = GPT2Config.from_pretrained('gpt2', output_hidden_states=False)
model = GPT2LMHeadModel.from_pretrained("gpt2", config=configuration)
model.resize_token_embeddings(len(tokenizer))

#train_dataset = datasets.CIFAR10(
#        root='/scratch/hpc72a03/cifar10-data', train=True, download=False, transform=transforms.ToTensor())
#train_sampler = DistributedSampler(dataset=train_dataset, shuffle=False)
#train_loader = torch.utils.data.DataLoader(
#        train_dataset , batch_size=32, sampler=train_sampler, shuffle=False, num_workers=2)
#model = ResNet(Bottleneck,  [3, 4, 6, 3]) #it means "resnet50 model"
model.cuda()
model.train()

model_parameter_names = {}
fsdp_params = dict(wrapper_cls=FSDP, flatten_parameters=True, reshard_after_forward=False, model_parameter_names=model_parameter_names)
fsdp_params_no_cls = dict( flatten_parameters=True, reshard_after_forward=False, model_parameter_names=model_parameter_names)

#with enable_wrap(**fsdp_params):
#  sharded_module = auto_wrap(model)
#
#  sharded_module = FSDP(sharded_module, **fsdp_params_no_cls)
#  sharded_module._lazy_init()
#  for n, p in sharded_module.named_parameters():
#    model_parameter_names[p] = n
optimizer = torch.optim.Adam(model.parameters(), lr=0.001, betas=(0.9, 0.999), eps=1e-08, weight_decay=0, amsgrad=False)
criterion = nn.CrossEntropyLoss()
iter_count = 0
for batch in tqdm(train_loader):

  b_input_ids = batch[0].cuda()
  b_labels = batch[0].cuda()
  b_masks = batch[1].cuda()


  output = model( b_input_ids,
                      labels=b_labels, 
                      attention_mask = b_masks,
                      token_type_ids=None
                    )
  loss = output[0] 
  #loss = criterion(output, target)
  if(rank ==0):
    print(loss)
  loss.backward()
  #optimizer.step()
  #optimizer.zero_grad()                       

