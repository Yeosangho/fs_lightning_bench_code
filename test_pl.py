import os
import argparse
import pandas as pd
from pl_gpt2 import GPT2_PL 
import torch
from torch.utils.data import Dataset, DataLoader, random_split, RandomSampler, SequentialSampler
from torch.utils.data.distributed import DistributedSampler
from transformers import GPT2LMHeadModel,  GPT2Tokenizer, GPT2Config, GPT2LMHeadModel
from pytorch_lightning import Trainer
from pytorch_lightning.strategies import DDPStrategy, DDPFullyShardedStrategy

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


def main(hparams):
    reviews = get_reviews()
    tokenizer = GPT2Tokenizer.from_pretrained('gpt2', bos_token='<|startoftext|>', eos_token='<|endoftext|>', pad_token='<|pad|>')
    train_dataset = GPT2Dataset(reviews, tokenizer)
    train_loader = torch.utils.data.DataLoader(train_dataset , batch_size=2, shuffle=False, num_workers=20)

    model = GPT2_PL(tokenizer)
    fsdp_strategy = DDPFullyShardedStrategy(reshard_after_forward=True, bucket_cap_mb=1000)
    trainer = Trainer(accelerator='auto', benchmark=False, devices=4, num_nodes=2, strategy="ddp_sharded")

    trainer.fit(model, train_dataloaders=train_loader)


if __name__ == "__main__":
    root_dir = os.path.dirname(os.path.realpath(__file__))
    parser = argparse.ArgumentParser(add_help=False)
    hyperparams = parser.parse_args()

    # TRAIN
    main(hyperparams)