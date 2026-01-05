#!/usr/bin/env python
import os
import sys
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.nn import Linear
import torch.nn as nn

# https://docs.pytorch.org/tutorials/intermediate/dist_tuto.html
def run(rank, size):
    """ Distributed function to be implemented later. """
    pass

"""Blocking point-to-point communication."""
def ptp_run(rank, size):
    tensor = torch.zeros(1, dtype=torch.int32)
    if rank == 0:
        tensor += 1
        # Send the tensor to process 1
        dist.send(tensor=tensor, dst=1)
    else:
        # Receive tensor from process 0
        dist.recv(tensor=tensor, src=0)
    print('Rank ', rank, ' has data ', tensor[0])

def reduce_run(rank, size):
    tensor = torch.zeros(2, dtype=torch.int32)
    if rank == 0:
        tensor += 1
    if rank == 1:
        tensor += 2
    # Note: rank 2 and 3 have tensor = [0, 0]
    
    root_rank = 0
    # ALL ranks must call dist.reduce(), but only root_rank receives the result
    dist.reduce(tensor, op=dist.ReduceOp.SUM, dst=root_rank)
    
    print('Rank ', rank, ' has data ', tensor)
    # Rank 0: has data tensor([3, 3], dtype=torch.int32) - receives the sum
    # Rank 1: has data tensor([2, 2], dtype=torch.int32) - unchanged (only root gets result)
    # Rank 2: has data tensor([0, 0], dtype=torch.int32) - unchanged
    # Rank 3: has data tensor([0, 0], dtype=torch.int32) - unchanged


def reduce_broadcast_run(rank, size):
    '''Reduce and broadcast pattern: reduce to root, then broadcast to all'''
    tensor = torch.zeros(2, dtype=torch.int32)
    if rank == 0:
        tensor += 1
    if rank == 1:
        tensor += 2
    # Note: rank 2 and 3 have tensor = [0, 0]
    
    root_rank = 0
    
    # Step 1: Reduce - all ranks participate, but only root_rank receives the result
    # ALL ranks must call dist.reduce() - it's a collective operation
    dist.reduce(tensor, op=dist.ReduceOp.SUM, dst=root_rank)
    
    # At this point:
    # - Rank 0 has tensor([3, 3]) - the sum
    # - Rank 1, 2, 3 still have their original values
    
    # Step 2: Broadcast - root_rank sends the result to all ranks
    # ALL ranks must call dist.broadcast() - it's a collective operation
    dist.broadcast(tensor, src=root_rank)
    
    # Now all ranks have the same result
    print('Rank ', rank, ' has data ', tensor)
    # All ranks: have data tensor([3, 3], dtype=torch.int32)

def all_reduce_run(rank, size):
    tensor = torch.zeros(2, dtype=torch.int32)
    if rank == 0:
        tensor += 1
    if rank == 1:
        tensor += 2
    dist.all_reduce(tensor, op=dist.ReduceOp.SUM)
    print('Rank ', rank, ' has data ', tensor)
    # Rank  1  has data  tensor([3, 3], dtype=torch.int32)
    # Rank  0  has data  tensor([3, 3], dtype=torch.int32)



def all_gather_run(rank, size=4):
    '''Simulate tensor parallel operation: split Linear(24, 32) weight across ranks and gather it back'''
    # Linear layer: input_size=24, output_size=32
    # Weight shape is (out_features, in_features) = (32, 24)
    in_features, out_features = 24, 32
    
    # In tensor parallelism, we split the weight along the output dimension
    # Since 32 is divisible by 4, each rank gets exactly 32/4 = 8 output features
    chunk_size = out_features // size  # 32 // 4 = 8
    
    # Calculate start and end indices for this rank's chunk
    start_idx = rank * chunk_size
    end_idx = start_idx + chunk_size
    
    # Create a full weight matrix (simulating what each rank would have locally)
    # In real tensor parallelism, each rank only creates its chunk
    # For demonstration, we create the full weight and then split it
    # IMPORTANT: Set the same seed for all ranks so they generate the same weight matrix
    torch.manual_seed(42)  # Fixed seed ensures all ranks generate the same matrix
    full_weight = torch.randn(out_features, in_features)
    
    # Each rank holds only its portion of the weight
    local_weight = full_weight[start_idx:end_idx, :].clone()
    
    print(f'Rank {rank}: Local weight shape {local_weight.shape}, handles outputs {start_idx}:{end_idx}')
    
    # Use all_gather to collect all weight chunks from all ranks
    # Since all chunks have the same shape (8, 24), we can directly gather them
    gathered_list = [torch.zeros_like(local_weight) for _ in range(size)]
    dist.all_gather(gathered_list, local_weight)
    
    # Concatenate all gathered chunks to reconstruct full weight
    full_weight_gathered = torch.cat(gathered_list, dim=0)
    
    print(f'Rank {rank}: Gathered full weight shape {full_weight_gathered.shape}')
    if rank == 0:  # Only print comparison once to avoid clutter
        match = torch.allclose(full_weight, full_weight_gathered)
        print(f'Rank {rank}: Original and gathered weights match: {match}')
        if not match:
            print(f'  Max diff: {torch.max(torch.abs(full_weight - full_weight_gathered))}')


def gather_run(rank, size=4):
    '''Simulate gather operation: only root rank (rank 0) collects all weight chunks'''
    # Linear layer: input_size=24, output_size=32
    # Weight shape is (out_features, in_features) = (32, 24)
    in_features, out_features = 24, 32
    
    # In tensor parallelism, we split the weight along the output dimension
    # Since 32 is divisible by 4, each rank gets exactly 32/4 = 8 output features
    chunk_size = out_features // size  # 32 // 4 = 8
    
    # Calculate start and end indices for this rank's chunk
    start_idx = rank * chunk_size
    end_idx = start_idx + chunk_size
    
    # Create a full weight matrix (simulating what each rank would have locally)
    # IMPORTANT: Set the same seed for all ranks so they generate the same weight matrix
    torch.manual_seed(42)  # Fixed seed ensures all ranks generate the same matrix
    full_weight = torch.randn(out_features, in_features)
    
    # Each rank holds only its portion of the weight
    local_weight = full_weight[start_idx:end_idx, :].clone()
    
    print(f'Rank {rank}: Local weight shape {local_weight.shape}, handles outputs {start_idx}:{end_idx}')
    
    # Use gather to collect all weight chunks from all ranks to root rank (rank 0)
    # Only rank 0 needs to prepare the gather_list
    root_rank = 0
    if rank == root_rank:
        # Root rank prepares a list to receive tensors from all ranks
        gathered_list = [torch.zeros_like(local_weight) for _ in range(size)]
        dist.gather(local_weight, gather_list=gathered_list, dst=root_rank)
        
        # Concatenate all gathered chunks to reconstruct full weight
        full_weight_gathered = torch.cat(gathered_list, dim=0)
        
        print(f'Rank {rank}: Gathered full weight shape {full_weight_gathered.shape}')
        match = torch.allclose(full_weight, full_weight_gathered)
        print(f'Rank {rank}: Original and gathered weights match: {match}')
        if not match:
            print(f'  Max diff: {torch.max(torch.abs(full_weight - full_weight_gathered))}')
    else:
        # Non-root ranks only send their tensor, gather_list should be None
        dist.gather(local_weight, gather_list=None, dst=root_rank)
        print(f'Rank {rank}: Sent local weight to rank {root_rank}')


def scatter_run(rank, size=4):
    '''Simulate scatter operation: root rank (rank 0) distributes full weight to all ranks'''
    # Linear layer: input_size=24, output_size=32
    # Weight shape is (out_features, in_features) = (32, 24)
    in_features, out_features = 24, 32
    
    # In tensor parallelism, we split the weight along the output dimension
    # Since 32 is divisible by 4, each rank gets exactly 32/4 = 8 output features
    chunk_size = out_features // size  # 32 // 4 = 8
    
    # Calculate start and end indices for this rank's chunk
    start_idx = rank * chunk_size
    end_idx = start_idx + chunk_size
    
    root_rank = 0
    
    # Only root rank creates the full weight matrix
    if rank == root_rank:
        torch.manual_seed(42)  # Fixed seed for reproducibility
        full_weight = torch.randn(out_features, in_features)
        
        # Split the full weight into chunks for scattering
        scatter_list = []
        for r in range(size):
            r_start = r * chunk_size
            r_end = r_start + chunk_size
            scatter_list.append(full_weight[r_start:r_end, :].clone())
        
        print(f'Rank {rank}: Full weight shape {full_weight.shape}, splitting into {size} chunks')
        
        # Root rank scatters chunks to all ranks (including itself)
        # The first chunk goes to rank 0, second to rank 1, etc.
        local_weight = torch.zeros(chunk_size, in_features)
        dist.scatter(local_weight, scatter_list=scatter_list, src=root_rank)
        
        print(f'Rank {rank}: Received local weight shape {local_weight.shape}, handles outputs {start_idx}:{end_idx}')
        
        # Verify that rank 0 received the correct chunk
        expected_chunk = full_weight[start_idx:end_idx, :]
        match = torch.allclose(local_weight, expected_chunk)
        print(f'Rank {rank}: Received chunk matches expected: {match}')
        if not match:
            print(f'  Max diff: {torch.max(torch.abs(local_weight - expected_chunk))}')
    else:
        # Non-root ranks prepare an empty tensor to receive their chunk
        local_weight = torch.zeros(chunk_size, in_features)
        dist.scatter(local_weight, scatter_list=None, src=root_rank)
        
        print(f'Rank {rank}: Received local weight shape {local_weight.shape}, handles outputs {start_idx}:{end_idx}')
        
        # Non-root ranks can verify by checking if the received chunk is non-zero
        # (In a real scenario, they would compare with expected values if available)
        print(f'Rank {rank}: Received chunk is non-zero: {torch.any(local_weight != 0)}')



def init_process(rank, size, fn, backend='gloo'):
    """ Initialize the distributed environment. """
    os.environ['MASTER_ADDR'] = '127.0.0.1'
    os.environ['MASTER_PORT'] = '29500'
    dist.init_process_group(backend, rank=rank, world_size=size)
    fn(rank, size)
    dist.destroy_process_group()

if __name__ == "__main__":
    # Set multiprocessing start method once (before the loop)
    if "google.colab" in sys.modules:
        print("Running in Google Colab")
        mp.get_context("spawn")
    else:
        try:
            mp.set_start_method("spawn")
        except RuntimeError:
            # Start method already set, ignore
            pass
    
    # Menu of available functions
    functions = {
        '1': ('ptp_run', ptp_run, 'Point-to-point communication (send/recv)'),
        '2': ('reduce_run', reduce_run, 'Reduce operation (only root gets result)'),
        '3': ('reduce_broadcast_run', reduce_broadcast_run, 'Reduce + Broadcast pattern'),
        '4': ('all_reduce_run', all_reduce_run, 'All-reduce operation (all ranks get result)'),
        '5': ('all_gather_run', all_gather_run, 'All-gather operation (tensor parallel simulation)'),
        '6': ('gather_run', gather_run, 'Gather operation (only root collects)'),
        '7': ('scatter_run', scatter_run, 'Scatter operation (root distributes)'),
    }
    # Main loop - show menu and run selected function
    while True:
        print("=" * 70)
        print("PyTorch Distributed Training Examples")
        print("=" * 70)
        print("\nAvailable functions:")
        for key, (name, func, desc) in functions.items():
            print(f"  {key}. {name:25s} - {desc}")
        print("  8. Exit")
        print("=" * 70)
        
        # Get user selection
        choice = input("\nSelect a function to run (1-8): ").strip()
        
        if choice == '8' or choice.lower() == 'q':
            print("Exiting...")
            sys.exit(0)
        
        if choice in functions:
            selected_name, selected_func, selected_desc = functions[choice]
            print(f"\n{'=' * 70}")
            print(f"Running: {selected_name}")
            print(f"Description: {selected_desc}")
            print(f"{'=' * 70}\n")
            
            world_size = 4
            processes = []
            
            for rank in range(world_size):
                p = mp.Process(target=init_process, args=(rank, world_size, selected_func))
                p.start()
                processes.append(p)

            for p in processes:
                p.join()
            
            print(f"\n{'=' * 70}")
            print(f"Completed: {selected_name}")
            print(f"{'=' * 70}\n")
            # Loop continues to show menu again
        else:
            print(f"Invalid choice '{choice}'. Please select 1-8.")
            print()