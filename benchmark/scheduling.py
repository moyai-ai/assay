"""Process-wide resource budgets. Reserve whole frozen pools, not individual containers."""
import asyncio
from collections import deque
from contextlib import asynccontextmanager


class Budget:
    """FIFO weighted semaphore with cancellation-safe grants and observable peaks."""
    def __init__(self, capacity):
        if type(capacity) is not int or capacity < 1:
            raise ValueError('Budget capacity must be a positive integer')
        self.capacity = capacity
        self.active = self.peak = 0
        self.queue = deque()

    def _drain(self):
        while self.queue:
            weight, future = self.queue[0]
            if future.cancelled():
                self.queue.popleft()
                continue
            if self.active + weight > self.capacity:
                break
            self.queue.popleft()
            self.active += weight
            self.peak = max(self.peak, self.active)
            future.set_result(None)

    async def acquire(self, weight=1):
        if type(weight) is not int or not 1 <= weight <= self.capacity:
            raise ValueError('Reservation exceeds budget capacity')
        future = asyncio.get_running_loop().create_future()
        self.queue.append((weight, future))
        self._drain()
        try:
            await future
        except BaseException:
            # Cancellation may land after a grant but before the caller resumes.
            if future.done() and not future.cancelled():
                self.release(weight)
            else:
                future.cancel()
                self._drain()
            raise

    def release(self, weight=1):
        if not 1 <= weight <= self.active:
            raise RuntimeError('Unbalanced resource release')
        self.active -= weight
        self._drain()

    @asynccontextmanager
    async def lease(self, weight=1):
        await self.acquire(weight)
        try:
            yield
        finally:
            self.release(weight)

    def snapshot(self):
        return {'capacity': self.capacity, 'active': self.active, 'peak': self.peak,
                'queued': sum(not future.cancelled() for _, future in self.queue)}


class Resources:
    def __init__(self, manifest):
        n = manifest['max_candidates'] + int(bool(manifest.get('baseline')))
        capacities = {
            'tasks': manifest.get('task_concurrency', 1),
            'containers': manifest.get('max_live_containers', n),
            'generation': manifest['concurrency'],
            'provider': manifest.get('provider_concurrency', 32),
            'baseline': manifest.get('baseline_concurrency', manifest['concurrency']),
            'grading': manifest.get('grading_concurrency', manifest['concurrency']),
            'setup': manifest.get('setup_concurrency', manifest['concurrency']),
            'snapshots': manifest.get('snapshot_concurrency', 1),
            'arms': manifest.get('arm_concurrency', 1),
            'judge': manifest.get('judge_concurrency', manifest['verifier']['concurrency']),
        }
        self.budgets = {name: Budget(capacity) for name, capacity in capacities.items()}
        for name, budget in self.budgets.items():
            setattr(self, name, budget)
        self.phases = {}

    def snapshot(self):
        return {'budgets': {name: budget.snapshot() for name, budget in self.budgets.items()},
                'task_phases': dict(self.phases)}
