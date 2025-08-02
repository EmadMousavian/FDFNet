from .crack_dataset import elpv
from torch.utils.data import DataLoader


def build_dataloader(data_path, batch_size, workers=4, mode="train", types='mono'):

    data = elpv(
        path=data_path,
        mode=mode,
        types=types
    )
    if mode == 'train':
        dataloader = DataLoader(data, batch_size=batch_size, num_workers=workers, shuffle=True)
    else:
        dataloader = DataLoader(data, batch_size=batch_size, num_workers=workers, shuffle=False)
    return dataloader
