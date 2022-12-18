import pytorch_lightning as pl
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import GPT2LMHeadModel,  GPT2Tokenizer, GPT2Config, GPT2LMHeadModel


class GPT2_PL(pl.LightningModule):
	def __init__(self,  tokenizer):
		super(GPT2_PL, self).__init__()
		self.tokenizer = tokenizer
		self.__build_model()

	def __build_model(self):
		configuration = GPT2Config.from_pretrained('gpt2-large', output_hidden_states=False)
		
		# instantiate the model
		self.model = GPT2LMHeadModel.from_pretrained("gpt2-large", config=configuration)
		
		# this step is necessary because I've added some tokens (bos_token, etc) to the embeddings
		# otherwise the tokenizer and model tensors won't match up
		self.model.resize_token_embeddings(len(self.tokenizer))
		self.model.train()
	def training_step(self, batch: tuple, batch_nb: int, *args, **kwargs) -> dict:
		b_input_ids = batch[0].cuda()
		b_labels = batch[0].cuda()

		b_masks = batch[1].cuda()
		outputs = self.model(  b_input_ids,
				labels=b_labels, 
				attention_mask = b_masks,
				token_type_ids=None
				)
		loss = outputs[0]
		return loss                                

	def configure_optimizers(self):
		""" Sets Learning rate for different parameter groups. """
		return torch.optim.Adam(self.model.parameters(), lr=0.02)