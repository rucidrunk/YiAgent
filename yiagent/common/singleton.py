"""Thread-safe singleton and simple utility helpers."""

from __future__ import annotations

import asyncio
import threading
from typing import Any, Callable, TypeVar

T = TypeVar("T")


class SingletonMeta(type):
    """Metaclass that makes a class a thread-safe singleton.

    ``__call__`` is synchronous (called during ``MyClass()``), so
    threading.Lock() is correct — it never yields to the event loop.
    """
    _instances: dict = {}
    _lock = threading.Lock()

    def __call__(cls, *args, **kwargs):
        if cls not in cls._instances:
            with cls._lock:
                if cls not in cls._instances:
                    cls._instances[cls] = super().__call__(*args, **kwargs)
        return cls._instances[cls]


class AsyncSingletonMeta(type):
    """Metaclass for singletons that need lazy async initialisation.

    Subclasses should define an async classmethod (e.g. ``get()``)
    that calls this metaclass's ``__call__`` synchronously for the
    instance slot, then runs async init.  The ``_init_lock`` here is
    ONLY for the sync ``__call__`` — async initialisation stages must
    use their own ``asyncio.Lock``.
    """
    _instances: dict = {}
    _init_lock = threading.Lock()

    def __call__(cls, *args, **kwargs):
        if cls not in cls._instances:
            with cls._init_lock:
                if cls not in cls._instances:
                    cls._instances[cls] = super().__call__(*args, **kwargs)
        return cls._instances[cls]


class LazyAsync:
    """Lazy-initialised async resource.

    Uses ``asyncio.Lock()`` because ``get()`` is an async method that
    may ``await`` while holding the lock — threading.Lock would block
    the entire event loop.

    Usage::

        resource = await LazyAsync(factory).get()
    """

    def __init__(self, factory: Callable):
        self._factory = factory
        self._value: Any = None
        self._ready = False
        self._lock = asyncio.Lock()

    async def get(self):
        if self._ready:
            return self._value
        async with self._lock:
            if self._ready:
                return self._value
            self._value = await self._factory()
            self._ready = True
            return self._value

    async def close(self):
        async with self._lock:
            if self._ready and hasattr(self._value, "close"):
                await self._value.close()
                self._ready = False
                self._value = None
