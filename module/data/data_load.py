import os
from torch.utils.data import DataLoader, distributed

from module.data.dataset import TrainValDataset
from module.utils.event import LOGGER
from module.utils.event import torch_distributed_zero_first

def create_dataloader(
    path,
    img_size,
    batch_size,
    hyp=None,
    augment=False,
    rank=-1,
    workers=8,
    shuffle=False,
    task="Train",
):
    """Create general dataloader.

    Returns dataloader and dataset
    """
    with torch_distributed_zero_first(rank):
        dataset = TrainValDataset(
            path,
            img_size,
            batch_size,
            augment=augment,
            hyp=hyp,
            rank=rank,
            task=task,
        )

    batch_size = min(batch_size, len(dataset))
    workers = min(
        [
            os.cpu_count() // int(os.getenv("WORLD_SIZE", 1)),
            batch_size if batch_size > 1 else 0,
            workers,
        ]
    )  # number of workers
    sampler = (
        None if rank == -1 else distributed.DistributedSampler(dataset, shuffle=shuffle)
    )
    return DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=shuffle,
            sampler=sampler,
            pin_memory=True,
        )