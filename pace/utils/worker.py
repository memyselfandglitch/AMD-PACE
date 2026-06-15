# ******************************************************************************
# Copyright (c) 2025 Advanced Micro Devices, Inc.
# All rights reserved.
# ******************************************************************************

import multiprocessing as mp
from pace.core import thread_bind
from pace.utils.logging import PACE_DEBUG


class Worker:
    """
    A class to handle asynchronous processing using multiple CPU cores.

    Attributes:
        cores_list: A list specifying cores to bind the process to.
        condition: A multiprocessing Condition object for thread synchronization.
        flag: A multiprocessing Value used as a flag to signal when conditions are met.
        target_func: The target function to execute in a separate process.
        target_args: Arguments for the target function.
        init_func: Initialization function to run.
        init_args: Arguments for the initialization function.
        process: The multiprocessing Process that runs the `run` method.
    """

    def __init__(
        self, worker_id, cores_list, init_func, init_args, target_func, target_args
    ):
        self._initialize(
            worker_id, cores_list, init_func, init_args, target_func, target_args
        )

    def _initialize(
        self, worker_id, cores_list, init_func, init_args, target_func, target_args
    ):
        """Handles the initialization logic for the Worker class."""
        PACE_DEBUG(f"Worker created with cores list: {cores_list}")
        self.worker_id = worker_id
        self.cores_list = cores_list
        self.condition = mp.Condition()
        self.flag = mp.Value("i", 0)
        self.target_func = target_func
        self.target_args = target_args
        self.init_func = init_func
        self.init_args = init_args
        self.process = mp.Process(target=self.run, args=(cores_list,))
        self.process.start()

    def run(self, cores_list):
        """
        Binds the process to specific cores and executes the initialization
        followed by the target function once conditions are met.
        """
        thread_bind(cores_list)
        self.init_func(*self.init_args)
        PACE_DEBUG("Before waiting on the condition")
        with self.condition:
            while self.flag.value == 0:
                self.condition.wait()
        PACE_DEBUG("After waiting on the condition")
        PACE_DEBUG("Calling the target function")
        self.target_func(*self.target_args)

    def start(self):
        """
        Starts the worker by notifying the condition, allowing the target
        function to execute.
        """
        PACE_DEBUG("Worker start. Before notifying")
        with self.condition:
            self.flag.value = 1
            self.condition.notify()

    def join(self):
        """Waits for the process to finish."""
        self.process.join()


class MultipleProcesses:
    """
    Manages multiple Worker instances.
    Attributes:
        workers: A list of Worker instances.
    """

    def __init__(
        self,
        workers: list,
    ):
        self.workers = workers

    def run(self):
        """Starts all the Worker instances."""
        for worker in self.workers:
            worker.start()

    def join(self):
        """Joins all the Worker instances."""
        for worker in self.workers:
            worker.join()
