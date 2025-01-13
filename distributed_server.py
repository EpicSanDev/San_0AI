import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.nn.parallel import DistributedDataParallel as DDP
from san_0ai import SanAI
import os

class DistributedAICluster:
    def __init__(self, world_size):
        self.world_size = world_size
        self.model_shards = {}
        
    def setup(self, rank):
        os.environ['MASTER_ADDR'] = 'localhost'
        os.environ['MASTER_PORT'] = '12355'
        dist.init_process_group("nccl" if torch.cuda.is_available() else "gloo", 
                              rank=rank, 
                              world_size=self.world_size)
        
    def run_node(self, rank):
        self.setup(rank)
        model = SanAI()
        model = DDP(model.model, device_ids=[rank] if torch.cuda.is_available() else None)
        
        # Configuration avancée du modèle distribué
        model.register_comm_hook(state=None, hook=self._gradient_compression_hook)
        
        while True:
            dist.barrier()
            self._synchronize_parameters(model)
            
    @staticmethod
    def _gradient_compression_hook(state, bucket):
        # Compression des gradients pour optimiser la communication
        return bucket.div_(dist.get_world_size())
        
    def _synchronize_parameters(self, model):
        for param in model.parameters():
            dist.all_reduce(param.grad.data, op=dist.ReduceOp.AVG)

def start_distributed_server(world_size):
    cluster = DistributedAICluster(world_size)
    mp.spawn(cluster.run_node,
             args=(world_size,),
             nprocs=world_size,
             join=True)
