from torch.utils.data import Dataset
import torch
import glob
import numpy as np
import pandas as pd
import tqdm
import os
import json
import MDAnalysis as mda
from MDAnalysis.transformations import *
from MDAnalysis.coordinates.memory import MemoryReader
from MDAnalysis.analysis.base import AnalysisFromFunction
from MDAnalysis.analysis import diffusionmap, align, rms

class QDataset(Dataset):
    def __init__(self, frames, qcharge_path=None, topology=None, atoms='protein and not name H*', 
                        skip=1, start=0, stop=-1, calculate_statistics=True, meanval=0, stdval=1, in_memory=False, 
                        transform=False, plane='xy', root='/home/ebam/kacher1/molearn/DATA', exp_name='new_exp', extra=False):
        
        self.frames = frames
        self.qcharge_path = qcharge_path
        self.topology = topology
        self.atoms = atoms 
        self.skip = skip
        self.start = start
        self.stop = stop
        self.calculate_statistics = calculate_statistics 
        self.meanval = meanval
        self.stdval = stdval
        self.plane=plane
        self.in_memory = in_memory
        self.root = root
        self.exp_name = exp_name
        self.extra = extra
    
        if self.extra: 
            self.coordinates = np.load(f'{self.root}/{self.exp_name}/positions.npy')
            if self.topology is None: 
                self.u = mda.Universe(self.frames) 
            else: 
                self.u = mda.Universe(self.topology, self.frames)
            self.selection = self.u.select_atoms(self.atoms)
            self.n_residues = self.selection.n_residues
            self.n_atoms = self.selection.n_atoms

            self.atoms_names = self.selection.atoms.names 
            with open(f'{self.root}/{self.exp_name}/unique_atoms_to_labels.json', 'r') as file: 
                self.atomname_to_label =  json.load(file)
            labelled_atoms = []
            for atom in self.atoms_names:
                labelled_atoms.append(self.atomname_to_label[atom])
            self.labelled_atoms = torch.tensor(labelled_atoms)

            
    # reading trajectory
        else:
            if self.topology:
                if isinstance(self.frames, str):
                    u = mda.Universe(self.topology, self.frames) #, skip_timestep=self.skip)
                    print('I have just had a look at the trajectory')
                elif isinstance(self.frames, list) or isinstance(self.frames, tuple):
                    u = mda.Universe(self.topology, self.frames) #, skip_timestep=self.skip 
                    print('I have just had a look at the trajectory')

            elif self.topology is None: 
                    u = mda.Universe(self.frames) #, step=self.skip) 
                    print('I have just had a look at the trajectory')
            else: 
                raise ValueError('Sorry, cannot create a dataset, check the files or readme!') 

    # preparing trajectory if it was not done in advance         
            if transform and topology is None: # BE CAREFUL WITH PDB FORMAT!!!!
                workflow = (mda.transformations.center_in_box(u.select_atoms(self.atoms), center='mass'),
                            mda.transformations.fit_rot_trans(u.select_atoms(self.atoms), u.select_atoms(self.atoms), plane=self.plane))
                for ts in u.trajectory[self.start:self.stop:self.skip]:
                    for transformation in workflow:
                        ts = transformation._transform(ts)
                print('Transform is done')
            elif transform:
                workflow = (mda.transformations.unwrap(u.atoms), 
                        mda.transformations.center_in_box(u.select_atoms(self.atoms), center='mass'),
                        mda.transformations.wrap(u.select_atoms(self.atoms), compound='molecules'),
                        mda.transformations.fit_rot_trans(u.select_atoms(self.atoms), u.select_atoms(self.atoms), plane=self.plane)) 
                for ts in u.trajectory[self.start:self.stop:self.skip]: 
                    for transformation in workflow:
                        ts = transformation._transform(ts)
                print('Transform is done')  
                                      
    # getting rid of some atoms  
            selection = u.select_atoms(self.atoms)
            self.selection = selection

    # initializing self.Universe 
            if self.in_memory==True:
                u.transfer_to_memory(start=self.start, stop=self.stop, step=self.skip)
                coordinates = AnalysisFromFunction(lambda ag: ag.positions.copy(), self.selection).run().results
                self.coordinates = coordinates 
            else: 
                self.u = u
                #u.trajectory
                coord = []
                for i in selection.universe.trajectory[self.start:self.stop:self.skip]:
                    coord.append(selection.positions)
                self.coordinates = coord 
            #as in MDAnalysis, check https://docs.mdanalysis.org/2.1.0/documentation_pages/selections.html
            #self.positions = np.array([(self.u.trajectory.time, self.selection.atoms.positions) for ts in u.trajectory])
            
            np.save(f'{self.root}/{self.exp_name}/positions.npy', self.coordinates)
    # for practical info         
            self.atoms_names = self.selection.atoms.names  
            #np.save(f'{self.self.root}/{self.exp_name}/atoms_names.npy', self.atoms.names)
            self.n_atoms = self.selection.n_atoms
            self.n_residues = self.selection.n_residues 
    # preparing for positional embedding 
            self.atomname_to_label = dict()
            unique_names = np.unique(self.atoms_names) 
            for index, atom in enumerate(unique_names):
                self.atomname_to_label[atom] = index 
            
            if not os.path.exists(f'{self.root}/{self.exp_name}/'): 
                os.mkdir(f'{self.root}/{self.exp_name}/')
            else: 
                with open(f'{self.root}/{self.exp_name}/unique_atoms_to_labels.json', 'w') as file: 
                    json.dump(self.atomname_to_label, file)
            labelled_atoms = []
            for atom in self.atoms_names:
                labelled_atoms.append(self.atomname_to_label[atom])
            self.labelled_atoms = torch.tensor(labelled_atoms)
# extracting charges 
        if qcharge_path is None:
            self.qcharges = None 
            
        elif isinstance(qcharge_path, str): 
            qcharge_path = [qcharge_path]
            
            if qcharge_path[0].endswith('.csv'):
                qcharges = []
                for qfile in qcharge_path: 
                    qdf = pd.read_csv(qfile)
                    qcharges.append(qdf)
                qdf = pd.concat(qcharges)
                qdf = qdf.iloc[self.start:self.stop:self.skip]
                self.qcharges = qdf.q_charge.to_numpy()

            elif qcharge_path[0].endswith('.npy'):
                qcharges = []
                for qfile in qcharge_path: 
                    qcharges.append(np.load(qfile, allow_pickle=True))
                qcharges = np.array(np.hstack(qcharges))
                qcharges = qcharges[self.start:self.stop:self.skip]
                self.qcharges = qcharges
                
            elif (qcharge_path[-1]).split('/')[-1] == 'colvars_Q':
                qcharges = []
                for qfile in qcharge_path:  
                    with open(qfile) as file:
                        lines = file.readlines()
                        for line in lines[1+self.start:self.stop:self.skip]:
                            values = list(map(float, line.split()))
                            qcharges.append(values[1])
                self.qcharges = np.array(qcharges)
                np.save(f'{self.root}/{self.exp_name}_start{self.start}_stop{self.stop}_skip{self.skip}.npy', qcharges)
            
        else:
            raise ValueError('No information about qcharges was provided, check the positional arguments')
# calculating statistics    
        if os.path.exists(f'{root}/mean_std.npy'):
            self.meanval, self.stdval = np.load(f'{self.root}/{self.exp_name}_{self.start}_{self.stop}_{self.skip}.npy') 
        elif calculate_statistics:
            self.meanval, self.stdval = self.calculate_mean_std()
            np.save(f'{root}/{exp_name}/meanval_stdval.npy', (self.meanval, self.stdval))

# Get item!             
    def __getitem__(self, item):
        if self.in_memory==True:
            frame_xyz = torch.tensor(self.coordinates["timeseries"][item])
        else:
            #self.selection.universe.trajectory[item]
            frame_xyz = torch.tensor(self.coordinates[item])

        frame_xyz = (frame_xyz - self.meanval)/self.stdval
        
        if self.qcharges is None: 
            return frame_xyz, self.labelled_atoms 
        else: 
            qcharge = self.qcharges[item] 
            return frame_xyz, self.labelled_atoms, qcharge 
# functions 
    def __len__(self):
        if self.qcharge_path: 
            return min(len(self.u.trajectory[::self.skip]), len(self.qcharges))
        else: 
            return len(self.u.trajectory[::self.skip])
            
    def calculate_mean_std(self):
        summ = 0 
        sqrsum = 0
        dataloader = torch.utils.data.DataLoader(self, batch_size=32, num_workers=0) #os.cpu_count())
        #print(len(dataloader))

        # if self.qcharges is None:
        for item, *_ in tqdm.tqdm(dataloader, desc="Calculating dataset statistics..."):
            sqrsum += (item**2).sum(dim=(0,1))
            summ += item.sum(dim=(0,1))
        # else:s
        #     for item, _, _ in tqdm.tqdm(dataloader, desc="Calculating dataset statistics..."):
        #         sqrsum += (item**2).sum(dim=(0,1))
        #         summ += item.sum(dim=(0,1))
          
        N = len(self) * item.shape[1] 

        meanval = summ / N
        stdval = np.sqrt(sqrsum / N - meanval**2)

        np.save(f'{self.root}/{self.exp_name}_{self.start}_{self.stop}_{self.skip}.npy', (meanval, stdval))
       
        return meanval, stdval

    def min_max_av_qcharge(self): 
        min_q = np.min(self.qcharges)
        max_q = np.max(self.qcharges)
        av_q = np.mean(self.qcharges)
        return min_q, max_q, av_q 
    
    def check_rmsd(self):
        R = rms.RMSD(self.u,  # universe to align
                     self.u,  # reference universe or atomgroup
                     select=self.atoms,  # group to superimpose and calculate RMSD
                     ref_frame=0)  # frame index of the reference
        R.run(step=self.skip)

        df = pd.DataFrame(R.results.rmsd)  
        rmsd = np.array(df[2])
        return  rmsd 
    
    def max_rmsd_dif(self): 
        R = rms.RMSD(self.u,  # universe to align
                     self.u,  # reference universe or atomgroup
                     select=self.atoms,  # group to superimpose and calculate RMSD
                     ref_frame=0)  # frame index of the reference
        R.run(start=self.start, stop=self.stop, step=self.skip)

        df = pd.DataFrame(R.results.rmsd, columns=['frame_idx', 'frame_idx_copy', 'value'] )  
        test1 = int(np.argmax(df.value))
        test1_idx = int(df.iloc[test1].frame_idx)
        #print(df)
        
        R2 = rms.RMSD(self.u,  # universe to align
                     self.u,  # reference universe or atomgroup
                     select=self.atoms,  # group to superimpose and calculate RMSD
                     ref_frame=test1_idx)  # frame index of the reference
        R2.run(start=self.start, stop=self.stop, step=self.skip)

        df2 = pd.DataFrame(R2.results.rmsd, columns=['frame_idx', 'frame_idx_copy', 'value'])
        test0 = int(np.argmax(df2.value))
        test0_idx = int(df2.iloc[test0].frame_idx)
        #print(df2)

        value = np.max(df2.value)

        return test0_idx, test1_idx, value

if __name__ == "__main__":
    path = '/home/ebam/kacher1/Files/Kv1.2.VSD/preprocessed_pdb_traj'
    data = QDataset(path, skip=100)
    print(len(data))
    print(data[0].shape)
    print(data[1000].shape)
    print(data[0])
    print(data[1000])
    print(data.meanval, data.stdval)

