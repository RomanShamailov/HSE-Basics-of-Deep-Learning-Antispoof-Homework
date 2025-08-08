import kagglehub
import numpy as np
import pandas as pd
import torch
import torchaudio
import torchvision
from hydra.utils import instantiate
from tqdm.auto import tqdm

from src.datasets.base_dataset import BaseDataset
from src.utils.io_utils import ROOT_PATH, read_json, write_json

# Download latest version
path = kagglehub.dataset_download("awsaf49/asvpoof-2019-dataset")


class ASVspoof2019_LA(BaseDataset):
    """
    Logical access partition of the ASVspoof2019 dataset.
    """

    def __init__(self, dataset_length, name="train", *args, **kwargs):
        """
        Args:
            name (str): partition name
        """
        index_path = ROOT_PATH / "data" / "example" / name / "index.json"

        # each nested dataset class must have an index field that
        # contains list of dicts. Each dict contains information about
        # the object, including label, path, etc.
        if index_path.exists():
            index = read_json(str(index_path))
        else:
            index = self._create_index(dataset_length, name)

        super().__init__(index, *args, **kwargs)

    def _create_index(self, dataset_length, name):
        """
        Create index for the dataset. The function processes dataset metadata
        and utilizes it to get information dict for each element of
        the dataset.

        Args:
            name (str): partition name
        Returns:
            index (list[dict]): list, containing dict for each element of
                the dataset. The dict has required metadata information,
                such as label and object path.
        """
        index = []
        data_path = ROOT_PATH / "data" / "example" / name
        data_path.mkdir(exist_ok=True, parents=True)

        protocol = pd.read_csv(
            path
            + f"/LA/LA/ASVspoof2019_LA_cm_protocols/ASVspoof2019.LA.cm.{'train' if name=='train' else 'eval'}.{'trn' if name=='train' else 'trl'}.txt",
            sep=" ",
            header=None,
            usecols=[1, 4],
        )
        id = protocol[1]
        labels = protocol[4]

        print(str(torchaudio.list_audio_backends()))

        # to get pretty object names
        number_of_zeros = int(np.log10(dataset_length)) + 1
        print(f"Creating ASVspoof2019 {name} Dataset partition")
        for i in tqdm(range(dataset_length)):
            # create dataset
            loaded_path = data_path / f"{i:0{number_of_zeros}d}.pt"
            speech_path = (
                path
                + f"/LA/LA/ASVspoof2019_LA_{'train' if name=='train' else 'eval'}/flac/{id[i]}.flac"
            )
            data, sr = torchaudio.load(speech_path)
            label = 1 if labels[i] == "bonafide" else 0
            torch.save(data, loaded_path)

            # parse dataset metadata and append it to index
            index.append({"path": str(loaded_path), "label": label})

        # write index to disk
        write_json(index, str(data_path / "index.json"))

        return index
