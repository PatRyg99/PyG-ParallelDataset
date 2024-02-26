# PyG-ParallelDataset

Multi-processing parallel PyTorch Geometric InMemoryDataset.

## Features:
* Run prefiltering and preprocessing on multiple jobs
* Save preprocessed data in chunks of specified length (zipping large files may run into RAM issues even when the dataset itself can fit into RAM)

## Dependencies:
* torch
* pyg
* pqdm
