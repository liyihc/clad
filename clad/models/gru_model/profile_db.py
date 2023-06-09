import torch
import numpy as np


class ProfileDB:
    def __init__(self):
        self.data = {}

    def clear_data(self):
        self.data.clear()

    def getProfiles(self, ids, default=None):
        return torch.stack([self.data.get(id, default) for id in ids])

    def setProfiles(self, ids, profiles):
        for id, profile in zip(ids, profiles):
            self.data[id] = profile.detach().cpu()

    def _getWrongProfiles_1_length(self, ids, default=None):
        key, value = next(iter(self.data.items()))
        return [default if id == key else value for id in ids]

    def _getWrongProfiles_more_length(self, ids):
        keys = list(self.data.keys())
        values = list(self.data.values())

        former: np.ndarray = np.arange(len(ids))
        new_key_id = former.copy()
        while True:
            eq = new_key_id == former
            s = eq.sum()
            if s > 0:
                new_key_id[eq] = np.random.randint(
                    low=0, high=len(keys), size=s)
            else:
                break

        return [values[key_id] for key_id in new_key_id]

    def getWrongProfiles(self, ids, default=None):
        if len(self.data) == 0:
            return [default] * len(ids)

        if len(self.data) == 1:
            return torch.stack(self._getWrongProfiles_1_length(ids, default=default))

        return torch.stack(self._getWrongProfiles_more_length(ids))
