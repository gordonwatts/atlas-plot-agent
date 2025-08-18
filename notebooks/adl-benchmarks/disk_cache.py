import functools
from diskcache import Cache
from pathlib import Path


def diskcache_decorator(cache: Cache | str | Path):
    """
    Decorator to cache function results using a diskcache.Cache instance.

    Args:
        cache (Cache | str | Path): An instance of diskcache.Cache or a path (str/Path) to create
                                    the cache.

    Returns:
        function: A decorator that caches the output of the decorated function.
            The cache key is based on the function name, positional arguments,
            and sorted keyword arguments. To bypass the cache, pass
            'ignore_cache=True' as a keyword argument.
    """
    # If cache is a str or Path, create a Cache instance
    if isinstance(cache, (str, Path)):
        cache = Cache(str(cache))

    def decorator(func):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            ignore_cache = kwargs.pop("ignore_cache", False)
            key = (func.__name__, args, tuple(sorted(kwargs.items())))
            if not ignore_cache and key in cache:
                return cache[key]
            result = func(*args, **kwargs)
            cache[key] = result
            return result

        return wrapper

    return decorator
