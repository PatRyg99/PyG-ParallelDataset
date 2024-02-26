import sys
from typing import Any, List

import torch
from pqdm.processes import pqdm
from torch_geometric.data import InMemoryDataset
from torch_geometric.data.dataset import files_exist
from torch_geometric.transforms.compose import Compose


class InMemoryParallelDataset(InMemoryDataset):
    r"""PyTorch geometric InMemoryDataset enhanced with multiprocessing pre-transforn
    and data chunking for the pre-transformed .pt folders.

    Args:
        root (str, optional): Root directory where the dataset should be saved.
            (optional: :obj:`None`)
        data_list (List(Any)): List of data samples on which the dataset should
            be called e.g. list of paths to all the samples
        transform (callable, optional): A function/transform that takes in a
            :class:`~torch_geometric.data.Data` or
            :class:`~torch_geometric.data.HeteroData` object and returns a
            transformed version.
            The data object will be transformed before every access.
            (default: :obj:`None`)
        pre_transform (callable, optional): A function/transform that takes in
            a :class:`~torch_geometric.data.Data` or
            :class:`~torch_geometric.data.HeteroData` object and returns a
            transformed version.
            The data object will be transformed before being saved to disk.
            (default: :obj:`None`)
        pre_filter (callable, optional): A function that takes in a
            :class:`~torch_geometric.data.Data` or
            :class:`~torch_geometric.data.HeteroData` object and returns a
            boolean value, indicating whether the data object should be
            included in the final dataset. (default: :obj:`None`)
        num_workers (int, optional): Number of processes to spawn for
            data prefiltering and preprocessing, use 1 for debugging
        chunk_num_samples: (int, optional): Number of samples to include in
            every .pt chunk of preprocessed dataset (zipping large datasets
            requires much more RAM then whole dataset, this way we can process
            dataset without running into RAM issues). Since the data is loaded
            from all chunks at once to the RAM once processed, the training speed
            is the same as for InMemoryDataset
    """

    def __init__(
        self,
        root: str,
        data_list: str,
        transform: Compose = None,
        pre_transform: Compose = None,
        pre_filter: Compose = None,
        num_workers: int = 1,
        chunk_num_samples: int = 500,
    ):
        self.root = root
        self.data_list = data_list
        self.num_workers = num_workers
        self.chunk_num_samples = chunk_num_samples

        self.data_list_split = self._split_into_chunks(self.data_list)

        super().__init__(root, transform, pre_transform, pre_filter)

        data_list = []
        for processed_path in self.processed_paths:
            data_list_chunk = torch.load(processed_path)
            data_list.extend(data_list_chunk)

        self.data, self.slices = self.collate(data_list)
        del data_list

    @property
    def raw_file_names(self):
        return []

    @property
    def processed_file_names(self):
        return [f"processed_data_{i}.pt" for i in range(len(self.data_list_split))]

    def _split_into_chunks(self, data_list: List[Any]):
        split_markers = [
            (i + 1) * self.chunk_num_samples
            for i in range(len(data_list) // self.chunk_num_samples)
        ]

        split_markers = [0] + split_markers
        data_list_split = [
            data_list[i:j] for i, j in zip(split_markers, split_markers[1:])
        ] + [data_list[split_markers[-1] :]]

        return data_list_split

    def process(self):
        for i, data_list in enumerate(self.data_list_split):

            if files_exist([self.processed_paths[i]]):
                if self.log:
                    print(
                        f"Chunk [{i+1}/{len(self.data_list_split)}]: skipping (chunk already processed)"
                    )
            else:
                if self.log:
                    print(f"Chunk [{i+1}/{len(self.data_list_split)}]: processing")

                if self.pre_filter is not None:
                    data_list = pqdm(
                        data_list, self.pre_filter, n_jobs=self.num_workers
                    )

                if self.pre_transform is not None:
                    data_list = pqdm(
                        data_list, self.pre_transform, n_jobs=self.num_workers
                    )

                torch.save(data_list, self.processed_paths[i])

    def _process(self):
        if self.log:
            print(f"Loading data from root: {self.root}", file=sys.stderr)

        super()._process()
