from __future__ import annotations

import asyncio
from collections.abc import AsyncIterator


async def drain[T](queue: asyncio.Queue[T]) -> AsyncIterator[T]:
    """Yield the items of the queue until it is shut down and empty."""
    while True:
        try:
            yield await queue.get()
        except asyncio.QueueShutDown:
            return
