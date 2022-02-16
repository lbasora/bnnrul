import os

import numpy as np

import torch


np.random.seed(seed=42)


class Dataset(torch.utils.data.Dataset):
    def __init__(
        self,
        root="data",
        dataset="FD001",
        split="train",
        normalization="min-max",
        channel_first=True,
        shuffle=None,
        max_rul=10000,
        quantity=1.0,
        cache_size=100000,
    ):
        """
        Parameters
        ----------
        root : str, optional
            Root directory of the dataset.
        dataset : str, optional
            Dataset to load.
        split : str, optional
            Portion of the data to load. Either 'train' or 'validation' or 'test'.
        normalization : str, optional
            Normalization strategy. Either 'min-max' or 'z-score'.
        channel_first : bool, optional
            True to load channels in the first dimension, False to load channels in the last one.
        max_rul : int, optional
            Label rectification threshold.
        quantity: float, optional
            Ratio of data to use. (1 - quantity) ratio of samples are randomly dropped.
        cache_size: int, optional
            Number of samples to cache.
        """
        assert split in ["train", "validation", "test"], (
            "'split' must be either 'train' or 'validation' or 'test', got '"
            + split
            + "'."
        )

        assert normalization in ["z-score", "min-max"], (
            "'normalization' must be either 'z-score' or 'min-max', got '"
            + normalization
            + "'."
        )

        assert 0 <= quantity <= 1.0, (
            "'quantity' must be a value within [0, 1], got %.2f" % quantity + "."
        )

        self.root = root
        tmp = dataset.split("/")
        #self.dataset = tmp[0] + "/data" if len(tmp) == 1 else tmp[0] + "/data/" + tmp[1]
        self.dataset = dataset
        self.split = split
        self.normalization = normalization
        self.channel_first = channel_first
        self.max_rul = max_rul

        # list of file_path
        self.datapath = (
            [
                os.path.join(
                    self.root, self.dataset, self.normalization, self.split, file_name
                )
                for file_name in os.listdir(
                    os.path.join(
                        self.root, self.dataset, self.normalization, self.split
                    )
                )
            ]
            if os.path.exists(
                os.path.join(self.root, self.dataset, self.normalization, self.split)
            )
            else []
        )

        # remove random elements from the list
        original_length = len(self.datapath)
        while len(self.datapath) > round(original_length * quantity):
            del self.datapath[np.random.randint(low=0, high=len(self.datapath))]

        # init data shape
        self.num_channels = 0
        self.window = 0
        self.num_features = 0

        # get data shape
        if len(self.datapath) > 0:
            with open(self.datapath[0], "r") as _:
                lines = _.readlines()
                self.num_channels = 1
                self.window = len(lines)
                self.num_features = len(lines[0].split())

        self.cache_size = cache_size  # how many data points to cache in memory
        self.cache = {}  # from index to (sample, label) tuple

    def _get_item(self, index):
        """Retrieve sample and label from either cache or source file.

        Parameters
        ----------
        index : int
            Sample index.

        Returns
        -------
        (ndarray, ndarray)
            4D array of `float` representing the data sample, 1D array of `float` (of size = 1) representing the label.
        """
        if index in self.cache:
            sample, label = self.cache[index]
        else:
            fn = self.datapath[index]
            label = float(fn.split("-")[-1].replace(".txt", ""))
            # rectify label
            if label > self.max_rul:
                label = self.max_rul
            label = np.array([label]).astype(np.float32)
            sample = np.loadtxt(fn).astype(np.float32)
            if self.channel_first:
                sample = sample.reshape(
                    self.num_channels, self.window, self.num_features
                )
            else:
                sample = sample.reshape(
                    self.window, self.num_features, self.num_channels
                )
            if len(self.cache) < self.cache_size:
                self.cache[index] = (sample, label)
        return sample, label

    def __getitem__(self, index):
        """Retrieve sample and label from either cache or source file.

        Parameters
        ----------
        index : int
            Sample index.

        Returns
        -------
        (ndarray, ndarray)
            4D array of `float` representing the data sample, 1D array of `float` (of size = 1) representing the label.
        """
        return self._get_item(index)

    def __len__(self):
        """Return total number of samples.

        Returns
        -------
        int
            Total number of samples.
        """
        return len(self.datapath)
