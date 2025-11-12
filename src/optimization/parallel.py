"""
Parallelization helpers (MPI / multiprocessing wrappers).
Provides a single interface 'run_parallel' to dispatch
tasks to different parallel backends.
"""

import multiprocessing as mp
import os
import time

# --- Optional MPI Import ---
# Try to import mpi4py. If it's not installed, MPI backend will be disabled.
try:
    from mpi4py import MPI
    MPI_AVAILABLE = True
except (ImportError, RuntimeError):
    MPI_AVAILABLE = False
    MPI = None # Define MPI as None so later checks don't fail
# ---------------------------


def run_parallel(func, args_list, backend="multiprocessing", **kwargs):
    """
    Runs a function in parallel over a list of arguments using the specified backend.

    This function uses a 'starmap' paradigm, meaning each item in 'args_list'
    is an iterable (like a tuple) that will be unpacked as the arguments for 'func'.

    Example:
        func = lambda x, y: x + y
        args_list = [(1, 2), (3, 4)]
        run_parallel(func, args_list) -> [3, 7]

    Args:
        func (callable): The function to execute in parallel.
        args_list (list): A list of argument-tuples. Each tuple will be
                          unpacked and passed to 'func'.
        backend (str): The parallel backend to use.
                         Supported: "multiprocessing", "mpi".
        **kwargs:
            num_processes (int): For 'multiprocessing' backend. Number of
                                 processes to spawn. (Default: os.cpu_count())

    Returns:
        list: A list of results in the same order as 'args_list'.
              If backend is 'mpi', only the root process (rank 0) returns
              the list. All other processes return None.
    """
    
    if not args_list:
        return []

    print(f"--- Running parallel job with backend: {backend} ---")

    # ==================================
    # === MULTIPROCESSING (Default) ===
    # ==================================
    if backend == "multiprocessing":
        num_processes = kwargs.get("num_processes", os.cpu_count())
        print(f"Spawning pool with {num_processes} processes...")
        
        # Use a Pool with 'with' statement for automatic cleanup
        with mp.Pool(processes=num_processes) as pool:
            # starmap applies func to each item in args_list
            # e.g., func(*(args_list[0])), func(*(args_list[1])), ...
            results = pool.starmap(func, args_list)
        
        print("--- Multiprocessing job complete ---")
        return results

    # ==================================
    # ===      MPI Backend           ===
    # ==================================
    elif backend == "mpi":
        if not MPI_AVAILABLE:
            raise ImportError(
                "MPI backend selected, but 'mpi4py' library is not installed. "
                "Please install it: pip install mpi4py"
            )
        
        comm = MPI.COMM_WORLD
        rank = comm.Get_rank()
        size = comm.Get_size()
        
        # --- Root Process (Rank 0): Scatter ---
        if rank == 0:
            print(f"MPI job started with {size} processes.")
            # Split the args_list into 'size' chunks
            # This 'chunking' method preserves the original order
            chunks = [[] for _ in range(size)]
            indices = [[] for _ in range(size)]
            for i, arg_set in enumerate(args_list):
                target_rank = i % size
                chunks[target_rank].append(arg_set)
                indices[target_rank].append(i) # Store original index
        else:
            chunks = None
            indices = None

        # Scatter the chunks to all processes
        my_chunk = comm.scatter(chunks, root=0)
        my_indices = comm.scatter(indices, root=0)
        
        # --- All Processes: Compute ---
        # Each process computes results for its own chunk
        # We store (original_index, result) to re-sort later
        my_results_with_indices = []
        for i, arg_set in zip(my_indices, my_chunk):
            result = func(*arg_set)
            my_results_with_indices.append((i, result))
            
        # --- Root Process (Rank 0): Gather ---
        # Gather lists of (index, result) tuples from all processes
        all_results_chunks = comm.gather(my_results_with_indices, root=0)
        
        if rank == 0:
            # Flatten the list of lists
            flat_results = [item for sublist in all_results_chunks for item in sublist]
            
            # Sort by the original index to restore order
            flat_results.sort(key=lambda x: x[0])
            
            # Extract just the results
            final_results = [result for index, result in flat_results]
            
            print("--- MPI job complete ---")
            return final_results
        else:
            # Other processes return None
            return None

    # ==================================
    # ===      Unknown Backend       ===
    # ==================================
    else:
        raise NotImplementedError(f"Backend '{backend}' is not supported.")


# --- Example Usage ---
def sample_task(x, y):
    """A simple task that simulates some work."""
    time.sleep(0.01) # Simulate I/O or computation
    return x * y

if __name__ == "__main__":
    
    # 1. Create a large list of tasks
    task_args = [(i, i + 1) for i in range(50)]
    
    # --- Example 1: Using 'multiprocessing' ---
    print("=== Testing 'multiprocessing' backend ===")
    start_time = time.time()
    mp_results = run_parallel(sample_task, task_args, backend="multiprocessing")
    end_time = time.time()
    
    print(f"Multiprocessing took: {end_time - start_time:.4f} seconds")
    if mp_results:
        print(f"First 5 results: {mp_results[:5]}")
        print(f"Total results: {len(mp_results)}\n")
    
    
    # --- Example 2: Using 'mpi' ---
    # This block will run on all processes, but only rank 0 will print
    if MPI_AVAILABLE:
        # Make sure all processes are synchronized before starting the next test
        MPI.COMM_WORLD.Barrier() 
        
        if MPI.COMM_WORLD.Get_rank() == 0:
            print("=== Testing 'mpi' backend ===")
            print("To run this test effectively, use:")
            print(f"mpiexec -n 4 python {__file__}\n")
            start_time = time.time()
        else:
            start_time = None # Only rank 0 measures time
            
        mpi_results = run_parallel(sample_task, task_args, backend="mpi")
        
        if MPI.COMM_WORLD.Get_rank() == 0:
            end_time = time.time()
            print(f"MPI took: {end_time - start_time:.4f} seconds")
            if mpi_results:
                print(f"First 5 results: {mpi_results[:5]}")
                print(f"Total results: {len(mpi_results)}")
    
    elif not MPI_AVAILABLE:
        print("=== 'mpi' backend test skipped ===")
        print("Install 'mpi4py' and an MPI implementation (like OpenMPI) to test.")