import os
import argparse
import pandas as pd
from tqdm import tqdm
import torch
import torch.distributed as dist 
from torch.utils.data import Dataset, DataLoader, random_split, RandomSampler, SequentialSampler
from torch.utils.data.distributed import DistributedSampler
from transformers import GPT2LMHeadModel,  GPT2Tokenizer, GPT2Config, GPT2LMHeadModel
from pytorch_lightning import Trainer
from pytorch_lightning.strategies import DDPStrategy, DDPFullyShardedStrategy
from transformers import GPT2LMHeadModel,  GPT2Tokenizer, GPT2Config, GPT2LMHeadModel
from fairscale.optim.oss import OSS
from fairscale.nn.wrap import wrap, enable_wrap, auto_wrap
from fairscale.nn.data_parallel import FullyShardedDataParallel as FSDP
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

# instantiate the model
model = GPT2LMHeadModel.from_pretrained("gpt2", config=configuration)

# this step is necessary because I've added some tokens (bos_token, etc) to the embeddings
# otherwise the tokenizer and model tensors won't match up
model.resize_token_embeddings(len(tokenizer))
model.cuda()
model.train()
base_optimizer_arguments = { "lr": 1e-4}

# Wrap a base optimizer into OSS
base_optimizer = torch.optim.Adam  # any pytorch compliant optimizer
optimizer = OSS(
    params=model.parameters(),
    optim=base_optimizer,
    **base_optimizer_arguments)

#fsdp_params = dict(wrapper_cls=SDP)
#with enable_wrap(**fsdp_params):
#  sharded_module = auto_wrap(model)
#optimizer = torch.optim.Adam(sharded_module.parameters(), lr=0.001, betas=(0.9, 0.999), eps=1e-08, weight_decay=0, amsgrad=False)
model = SDP(model, optimizer)
sharded_module =model
for batch in tqdm(train_loader):
  b_input_ids = batch[0].cuda()
  b_labels = batch[0].cuda()
  b_masks = batch[1].cuda()      
  output = sharded_module( b_input_ids,
                      labels=b_labels, 
                      attention_mask = b_masks,
                      token_type_ids=None
                    ) 
  loss = output[0] 
  loss.backward()
  optimizer.step()                       


