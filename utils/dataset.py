from torch.utils.data import Dataset
import torch
import glob
import numpy as np
import biobox
import tqdm
import os

class FramesDataset(Dataset):
    def __init__(self, frames_path, filter=None, atoms="*", ignore_atoms=[], meanval=0, stdval=1):
        self.frames_path = frames_path
        self.filter = filter

        self.frames = glob.glob(f"{frames_path}/*.pdb")

        if self.filter is not None:
            self.frames = [
                frame
                for frame in self.frames
                if self.filter(frame)
            ]

        self.meanval = meanval
        self.stdval = stdval

        self.mol = biobox.Molecule()
        self.mol.import_pdb(self.frames[0])

        if atoms == "*":
            #list like default above of atom names
            atoms = list(np.unique(self.mol.data["name"].values))
            if ignore_atoms:
                for to_remove in ignore_atoms:
                    if to_remove in atoms:
                        atoms.remove(to_remove)

        # resids = self.mol.data['resid'].to_numpy()
        # _, unique = np.unique(resids, return_counts=True)

        self.atoms = atoms
        _, self.atom_idxs = self.mol.atomselect("*", "*", self.atoms, get_index=True)
        self.mol = self.mol.get_subset(self.atom_idxs)
        self.atom_names = self.mol.get_data(columns=['name', 'resname','resid'])

    def __len__(self):
        return len(self.frames)

    def __getitem__(self, item):
        frame_path = self.frames[item]

        mol = biobox.Molecule()
        mol.import_pdb(frame_path)
        mol = mol.get_subset(self.atom_idxs)

        frame = mol.coordinates[0]

        frame = torch.tensor(frame).float()
        frame = (frame - self.meanval) / self.stdval
        
        frame = frame.T

        return frame

def calculate_mean_std(dataset: FramesDataset):
    summ = 0
    sqrsum = 0

    dataloader = torch.utils.data.DataLoader(dataset, batch_size=32, num_workers=os.cpu_count())

    for item in tqdm.tqdm(dataloader, desc="Calculating dataset statistics..."):
        summ += item.sum(dim=(0, 2))
        sqrsum += (item**2).sum(dim=(0, 2))

    N = len(dataset) * dataset[0].shape[1]

    meanval = summ / N
    stdval = np.sqrt(sqrsum / N - 2 * meanval * summ / N + meanval**2)

    return meanval, stdval



if __name__ == "__main__":
    path = '/home/ebam/kacher1/doc/sum_traj/split/'

    data = FramesDataset(path)
    print(len(data))

    print(data[0].shape)
    print(data[10000].shape)

    print(data[0])
    print(data[10000])

    mean, std = calculate_mean_std(data)
    np.save("/home/ebam/kacher1/molearn/DATA/all_each200/mean_std.npy", (mean, std))
    print(mean, std)
