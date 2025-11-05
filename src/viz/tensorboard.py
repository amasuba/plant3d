from __future__ import annotations
from torch.utils.tensorboard import SummaryWriter
def get_writer(logdir):
    return SummaryWriter(logdir)
