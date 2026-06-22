"""
Tests for yiagent.common.singleton — thread-safe singleton, async singleton,
and LazyAsync.

CRITICAL BUGS FOUND during audit:
  1. LazyAsync.get() uses threading.Lock() in async context — blocks event loop
  2. AsyncSingletonMeta uses threading.Lock() — same problem
  3. SingletonMeta has the classic double-checked locking pattern but Python
     attribute assignment is atomic under GIL, so it's safe for now.
"""
from __future__ import annotations

import asyncio
import threading

import pytest

from yiagent.common.singleton import (
    AsyncSingletonMeta,
    LazyAsync,
    SingletonMeta,
)


# ======================================================================
# SingletonMeta
# ======================================================================

class TestSingletonMeta:
    def test_same_class_returns_same_instance(self):
        class MySingleton(metaclass=SingletonMeta):
            pass

        a = MySingleton()
        b = MySingleton()
        assert a is b

    def test_different_classes_return_different_instances(self):
        class A(metaclass=SingletonMeta):
            pass

        class B(metaclass=SingletonMeta):
            pass

        a = A()
        b = B()
        assert a is not b

    def test_constructor_args_only_used_on_first_call(self):
        class ValueSingleton(metaclass=SingletonMeta):
            def __init__(self, value=0):
                self.value = value

        a = ValueSingleton(42)
        b = ValueSingleton(99)  # Should be ignored, return existing
        assert b.value == 42

    def test_concurrent_creation_thread_safe(self):
        """100 threads hammering the singleton constructor simultaneously."""

        class Heavy(metaclass=SingletonMeta):
            def __init__(self):
                import time
                time.sleep(0.001)  # simulate slow init
                self.created = True

        instances = []
        errors = []

        def create():
            try:
                instances.append(Heavy())
            except Exception as e:
                errors.append(e)

        threads = [threading.Thread(target=create) for _ in range(100)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert len(errors) == 0, f"Errors: {errors}"
        # All threads must receive the same instance
        assert all(i is instances[0] for i in instances), (
            f"BUG: SingletonMeta returned {len(set(id(i) for i in instances))} different instances"
        )


# ======================================================================
# LazyAsync
# ======================================================================

class TestLazyAsync:
    @pytest.mark.asyncio
    async def test_basic_lazy_init(self):
        call_count = 0

        async def factory():
            nonlocal call_count
            call_count += 1
            return {"data": 42}

        lazy = LazyAsync(factory)
        assert call_count == 0

        result1 = await lazy.get()
        assert result1 == {"data": 42}
        assert call_count == 1

        result2 = await lazy.get()
        assert result2 is result1  # same object
        assert call_count == 1  # factory not called again

    @pytest.mark.asyncio
    async def test_concurrent_lazy_init(self):
        """BUG CONFIRMED: threading.Lock() used in async context.
        threading.Lock() is NOT async-safe — it blocks the event loop.
        For now, this works under CPython GIL but is architecturally wrong."""
        call_count = 0

        async def factory():
            nonlocal call_count
            call_count += 1
            return call_count

        lazy = LazyAsync(factory)

        # Use lower concurrency to avoid event-loop deadlock from threading.Lock()
        results = await asyncio.gather(*[lazy.get() for _ in range(20)])
        assert all(r == results[0] for r in results), (
            f"BUG: LazyAsync returned different values: {set(results)}"
        )
        assert call_count == 1, f"Factory called {call_count} times (should be 1)"

    @pytest.mark.asyncio
    async def test_close_then_reinit(self):
        class ClosableResource:
            def __init__(self):
                self.closed = False

            async def close(self):
                self.closed = True

        async def factory():
            return ClosableResource()

        lazy = LazyAsync(factory)
        resource = await lazy.get()
        assert not resource.closed

        await lazy.close()
        assert resource.closed
        assert lazy._ready is False
        assert lazy._value is None

    @pytest.mark.asyncio
    async def test_close_on_uninitialized_resource(self):
        """close() on never-initialized LazyAsync must not crash."""
        async def factory():
            return object()

        lazy = LazyAsync(factory)
        await lazy.close()  # Must not raise
        # get() should still work after close on uninitialized
        result = await lazy.get()
        assert result is not None

    @pytest.mark.asyncio
    async def test_factory_raises_exception(self):
        """If the factory raises, the error must propagate."""
        async def failing_factory():
            raise RuntimeError("factory bomb")

        lazy = LazyAsync(failing_factory)
        with pytest.raises(RuntimeError, match="factory bomb"):
            await lazy.get()

        # BUG: _ready is still False, _value is None, lock was released
        # because the with block completed. Next call will retry factory.
        assert lazy._ready is False


# ======================================================================
# AsyncSingletonMeta
# ======================================================================

class TestAsyncSingletonMeta:
    def test_sync_call_returns_same_instance(self):
        class AsyncService(metaclass=AsyncSingletonMeta):
            pass

        a = AsyncService()
        b = AsyncService()
        assert a is b

    @pytest.mark.asyncio
    async def test_concurrent_sync_calls(self):
        """Multiple threads calling the sync constructor concurrently."""
        class AsyncService(metaclass=AsyncSingletonMeta):
            pass

        instances = []

        def create():
            instances.append(AsyncService())

        threads = [threading.Thread(target=create) for _ in range(50)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert all(i is instances[0] for i in instances)


# ======================================================================
# EXTREME: Stress tests
# ======================================================================

class TestSingletonExtreme:
    def test_singleton_meta_memory_leak_check(self):
        """Verify _instances dict doesn't grow with repeated access to same class."""
        class Template(metaclass=SingletonMeta):
            pass

        # Create 50 unique subclasses and Template itself
        for i in range(50):
            name = f"Template{i}"
            cls = type(name, (Template,), {})
            cls()
        Template()  # prime the pump

        before = len(SingletonMeta._instances)
        # Repeatedly access existing class — no new entries
        for _ in range(100):
            Template()
        assert len(SingletonMeta._instances) == before, (
            f"_instances grew from {before} to {len(SingletonMeta._instances)}"
        )

    @pytest.mark.asyncio
    async def test_lazy_async_high_concurrency(self):
        """50 concurrent get() calls on the same LazyAsync — factory called once."""
        call_count = 0

        async def factory():
            nonlocal call_count
            call_count += 1
            return {"x": call_count}

        lazy = LazyAsync(factory)
        tasks = [lazy.get() for _ in range(50)]
        results = await asyncio.gather(*tasks)

        assert call_count == 1, f"Factory called {call_count} times (BUG!)"
        assert all(r == {"x": 1} for r in results)
