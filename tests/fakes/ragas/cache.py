"""Fake `ragas.cache.DiskCacheBackend`: no-op placeholder, offline smoke test
doesn't need real disk caching of judge responses."""


class DiskCacheBackend:
    def __init__(self, cache_dir: str = ".cache"):
        self.cache_dir = cache_dir
