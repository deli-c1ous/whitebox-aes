from time import perf_counter
from contextlib import contextmanager


@contextmanager
def timer(task_name: str, pre_message: str):
    print(pre_message)
    start_time = perf_counter()
    yield
    end_time = perf_counter()
    print(f"{task_name}: {end_time - start_time:.4f} s")


if __name__ == '__main__':
    with timer("test", "Testing timer context manager"):
        for _ in range(1000000):
            pass
