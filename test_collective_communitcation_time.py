import torch
import time, os 
import torch.distributed as dist 

world_size = int(os.environ["WORLD_SIZE"])
rank = int(os.environ["SLURM_PROCID"])
os.environ['MASTER_PORT'] = os.environ['TRAINER_PORT']
dist.init_process_group(backend='nccl', world_size=world_size, rank=rank)

ngpus_per_node = torch.cuda.device_count()
device_id = rank%ngpus_per_node
torch.cuda.set_device(device_id)
tensor_size = 200 * 1024
buffer_size = 25 * 200 * 1024
buffer = torch.zeros(buffer_size).cuda()
tensor = torch.ones(tensor_size).cuda()

start_time = time.time()
handles = []
for i in range(100):
	handle = dist.all_reduce(tensor,async_op=True)
	handles.append(handle)
for i in range(100):
	handles[i].wait()
#print(tensor)
end_time = time.time()
if(rank == 0):
	print(end_time-start_time)
#
#
#tensor = torch.ones(tensor_size).cuda()
#
#comm_stream = torch.cuda.Stream(device_id)
#start_time = time.time()
#with torch.cuda.stream(comm_stream):
#	for i in range(100):
#		dist.all_reduce(tensor,async_op=False)
#torch.cuda.current_stream().wait_stream(comm_stream)
##print(tensor)
#end_time = time.time()
#if(rank == 0):
#	print(end_time-start_time)



tensor = torch.ones(tensor_size).cuda()

start_time = time.time()
handles = []
for i in range(100):
	handle = dist.all_reduce(tensor,async_op=True)
	handles.append(handle)
for i in range(100):
	handles[i].wait()
#print(tensor)
end_time = time.time()
if(rank == 0):
	print(end_time-start_time)

tensor = torch.ones(tensor_size).cuda()

start_time = time.time()
for i in range(25):
	buffer[i*200 * 1024:(i+1)*200 * 1024].copy_(tensor)
handles = []
for i in range(4):
	handle = dist.all_reduce(tensor,async_op=True)
	handles.append(handle)
for i in range(4):
	handles[i].wait()
for i in range(25):
	tensor.copy_(buffer[i*200 * 1024:(i+1)*200 * 1024])

#print(tensor)
end_time = time.time()
if(rank == 0):
	print(end_time-start_time)