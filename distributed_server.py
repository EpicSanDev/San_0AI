import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.nn.parallel import DistributedDataParallel as DDP
from san_0ai import SanAI
import os
import socket
import json

class DistributedAICluster:
    def __init__(self, world_size=None):
        self.gpu_count = torch.cuda.device_count()
        self.world_size = world_size if world_size else self.gpu_count
        self.node_config = self._auto_generate_config()
        
    def _auto_generate_config(self):
        """Génère automatiquement la configuration en fonction des GPUs disponibles"""
        hostname = socket.gethostname()
        gpu_ids = list(range(self.gpu_count))
        
        config = {
            "nodes": [{
                "host": hostname,
                "port": "12355",
                "gpu_ids": gpu_ids,
                "gpu_info": [self._get_gpu_info(i) for i in gpu_ids],
                "role": "master"
            }],
            "master_node": hostname,
            "total_gpus": self.gpu_count,
            "communication": {
                "timeout": 1800,
                "backend": "nccl" if torch.cuda.is_available() else "gloo",
                "init_method": "tcp"
            }
        }
        
        self._save_config(config)
        return config
        
    def _get_gpu_info(self, gpu_id):
        """Récupère les informations détaillées sur un GPU"""
        if not torch.cuda.is_available():
            return {"available": False}
            
        return {
            "name": torch.cuda.get_device_name(gpu_id),
            "memory_total": torch.cuda.get_device_properties(gpu_id).total_memory,
            "memory_available": torch.cuda.memory_allocated(gpu_id),
            "capability": torch.cuda.get_device_capability(gpu_id)
        }
        
    def _save_config(self, config):
        """Sauvegarde la configuration détectée"""
        with open('cluster_config.json', 'w') as f:
            json.dump(config, f, indent=4)
            
    def setup(self, rank):
        """Configuration adaptative du processus distribué"""
        if self.gpu_count == 0:
            device = torch.device("cpu")
            backend = "gloo"
        else:
            device = torch.device(f'cuda:{rank}')
            backend = "nccl"
            
        os.environ['MASTER_ADDR'] = self.node_config['master_node']
        os.environ['MASTER_PORT'] = self.node_config['nodes'][0]['port']
        
        dist.init_process_group(backend=backend,
                              rank=rank,
                              world_size=self.world_size)
        
        return device
        
    def run_node(self, rank):
        device = self.setup(rank)
        torch.cuda.set_device(device)  # Optimisation pour CUDA
        
        model = SanAI()
        model = model.to(device)
        
        if self.gpu_count > 0:
            model = DDP(model.model, device_ids=[rank])
            
        # Configuration des optimisations de communication
        if self.gpu_count > 0:
            model.register_comm_hook(state=None, hook=self._gradient_compression_hook)
            
        self._run_training_loop(model, device, rank)
        
    def _run_training_loop(self, model, device, rank):
        """Boucle d'entraînement avec gestion des ressources"""
        try:
            while True:
                if self.gpu_count > 0:
                    # Monitoring des ressources GPU
                    memory_used = torch.cuda.memory_allocated(device) / 1024**2
                    if memory_used > self._get_gpu_info(rank)["memory_total"] * 0.9:
                        torch.cuda.empty_cache()
                        
                dist.barrier()
                self._synchronize_parameters(model)
                
        except Exception as e:
            print(f"Erreur sur le nœud {rank}: {str(e)}")
            self._handle_node_failure(rank)
            
    def _process_distributed_batch(self, model, device):
        # Traitement distribué des lots
        batch_size = 32 // self.world_size  # Répartition de la charge
        # Exemple de traitement des lots
        data_loader = self._get_data_loader(batch_size)
        
        optimizer = torch.optim.SGD(model.parameters(), lr=0.01)  # Define the optimizer
        
        for batch in data_loader:
            inputs, labels = batch
            inputs, labels = inputs.to(device), labels.to(device)
            
            outputs = model(inputs)
            loss = self._compute_loss(outputs, labels)
            
            model.zero_grad()
            loss.backward()
            dist.barrier()  # Synchronisation des gradients
            self._synchronize_parameters(model)
            
            optimizer.step()
            
    def _get_data_loader(self, batch_size):
        # Création d'un DataLoader pour les données d'entraînement
        dataset = ...  # Charger votre dataset ici
        return torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True)
        
    def _compute_loss(self, outputs, labels):
        # Calcul de la perte
        criterion = torch.nn.CrossEntropyLoss()
        return criterion(outputs, labels)
        
    @staticmethod
    def _gradient_compression_hook(state, bucket):
        # Compression améliorée des gradients
        compressed_tensor = bucket.float() / 256
        compressed_tensor = compressed_tensor.to(torch.int8)
        return compressed_tensor.float() * 256
        
    def _synchronize_parameters(self, model):
        for param in model.parameters():
            dist.all_reduce(param.grad.data, op=dist.ReduceOp.AVG)

def start_distributed_server(world_size=None):
    cluster = DistributedAICluster(world_size)
    available_gpus = torch.cuda.device_count()
    
    if available_gpus == 0:
        print("Aucun GPU détecté, utilisation du CPU")
        world_size = 1
    else:
        print(f"Détection de {available_gpus} GPUs")
        world_size = world_size if world_size else available_gpus
        
    mp.spawn(cluster.run_node,
             args=(),
             nprocs=world_size,
             join=True)
