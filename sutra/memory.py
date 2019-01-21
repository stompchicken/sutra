import os
import tracemalloc
import logging
import gc
import psutil
import gpustat
import torch

LOGGER = logging.getLogger(__name__)


def get_allocated_memory():
    gc.collect()
    snapshot = tracemalloc.take_snapshot()
    current = snapshot.statistics('filename')
    return sum(stat.size for stat in current) / (2 ** 20)


def get_allocated_gpu_memory():
    return torch.cuda.memory_allocated() / (2 ** 20)


def get_process_memory():
    process = psutil.Process(os.getpid())
    mem = process.memory_info()
    rss = mem.rss / (2 ** 20)
    return rss


def get_process_gpu_memory(gpu_id=0):
    stat = gpustat.new_query()
    process = [xi for xi in stat.gpus[gpu_id].processes if xi['pid'] == os.getpid()] # noqa
    if len(process) == 1:
        return process[0]['gpu_memory_usage']
    else:
        return 0


class MemoryProfiler():

    def __init__(self, output_dir):
        tracemalloc.start()
        self.counter = 1

    def memory_usage(self, message=None):
        if not message:
            message = f"{self.counter}"

        allocated = get_allocated_memory()
        allocated_gpu = get_allocated_gpu_memory()
        rss = get_process_memory()
        gpu = get_process_gpu_memory()
        line = f"[{message}] "
        values = [("allocated", allocated),
                  ("allocated_gpu", allocated_gpu),
                  ("process", rss),
                  ("process_gpu", gpu)]

        def format_memory(name, value):
            return f"{name}: {value:.0f}M"

        line += ", ".join(format_memory(name, val) for name, val in values)
        LOGGER.debug(line)

        self.counter += 1


profiler = MemoryProfiler(os.getcwd())
