class Dataset:
    def __init__(self, data: dict):
        self._data = data

    @classmethod
    def from_dict(cls, data: dict):
        return cls(data)

    def __len__(self):
        return len(next(iter(self._data.values())))

    def __getitem__(self, key):
        return self._data[key]
