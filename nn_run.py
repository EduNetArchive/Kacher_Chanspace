import os
from copy import deepcopy
import pdb

import wandb
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim
import tensorflow as tf
from matplotlib import pyplot as plt
from tqdm import tqdm, trange
from argparse import ArgumentParser

import biobox
from molearn import Auto_potential, Autoencoder, load_data

import mdtraj as md
import nglview 

import MDAnalysis as mda
import MDAnalysis.tests.datafiles 
from MDAnalysis.analysis import align, rms
from MDAnalysis.tests.datafiles import TPR, XTC, PDB

import plotly as py
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from PIL import Image

from sklearn.cluster import DBSCAN, OPTICS
from sklearn import metrics
from sklearn.preprocessing import StandardScaler

from utils.dataset import FramesDataset, calculate_mean_std

# DATA_ORIGIN = "/home/ebam/kacher1/doc"

# ROOT = os.path.join(DATA, EXPERIMENT_NAME)

def charge_dif(df): 
    maximal_idx = np.argmax(df.q_charge)
    maximal_val = df.iloc[maximal_idx]

    minimal_idx = np.argmin(df.q_charge)
    minimal_val = df.iloc[minimal_idx]
    
    return maximal_idx, maximal_val, \
           minimal_idx, minimal_val 

def check_rmsd(traj): 
    R = rms.RMSD(traj,  # universe to align
                 traj,  # reference universe or atomgroup
                 select='backbone',  # group to superimpose and calculate RMSD
                 ref_frame=0)  # frame index of the reference
    R.run()
    
    df = pd.DataFrame(R.results.rmsd)
    
    test1_idx = int(np.argmax(df[2]))
    
    R2 = rms.RMSD(traj,  # universe to align
                 traj,  # reference universe or atomgroup
                 select='backbone',  # group to superimpose and calculate RMSD
                 ref_frame=test1_idx)  # frame index of the reference
    R2.run()
    
    df2 = pd.DataFrame(R2.results.rmsd)
    
    test0_idx = int(np.argmax(df2[2]))
    
    return test0_idx, test1_idx


def visualize_energy(network, dataset, num_atoms, lf, size=100, bounding_box=(0,0,1,1)):
    x1, y1, x2, y2 = bounding_box
    x = np.linspace(x1, x2, size)
    y = np.linspace(y1, y2, size)
    xx, yy = np.meshgrid(x, y) 
    z = np.stack((xx, yy), axis=2)
    z = z.reshape(size**2, 2, 1)

    losses_all = []
    for i in trange(0, len(z), 1): 
        z_batch = z[i:(i+1)]
        z_batch=torch.tensor(z_batch).float().to(device)
        out = network.decode(z_batch)[:, :, :num_atoms]
        out *= dataset.stdval
        bond_energy, angle_energy, torsion_energy, NB_energy = lf.get_loss(out)

        loss_f = (bond_energy + angle_energy + torsion_energy + NB_energy)
        losses_all.append(loss_f.item())
    
    img = np.array(losses_all)
    img = img.reshape(size, size)

    return img

def conformations_to_latent(network, train_loader, device):
    z_list=[]
    with torch.no_grad():
        for batch in tqdm(train_loader):
            x = batch.to(device)
            z = network.encode(x)
            z_list.append(z.cpu().squeeze(2))
    
    z_list = torch.cat(z_list)
    
    return z_list 

def visualization(network, dataset, num_atoms, lf, train_loader, device, size=100, bounding_box=None, log_scale=False):
    img_conf = conformations_to_latent(network, train_loader, device)
    
    if bounding_box is None:
        x1, y1 = img_conf.min(dim=0).values
        x2, y2 = img_conf.max(dim=0).values
        
        a = max(x2-x1, y2-y1)
        a *= 1.1
        x_av = (x1+x2)/2
        y_av = (y1+y2)/2
        
        x1 = x_av-a/2
        x2 = x_av+a/2
        y1 = y_av-a/2
        y2 = y_av+a/2
    
    else:
        x1, y1, x2, y2 = bounding_box

    img_array = visualize_energy(network, dataset, num_atoms, lf, size=size, bounding_box=(x1, y1, x2, y2))   
    
    plt.scatter(img_conf[:, 0], img_conf[:, 1], c=np.arange(len(img_conf)), alpha=0.5, s=1, cmap='PiYG')
    plt.imshow(img_array, extent=(x1, x2, y1, y2), origin='lower')
    plt.colorbar()

    plt.xlim(x1, x2)
    plt.ylim(y1, y2)

    plt.savefig("/tmp/fig1.png")
    plt.close()
    img = Image.open("/tmp/fig1.png")
    
            
    if log_scale:
        plt.scatter(img_conf[:, 0], img_conf[:,1], c=np.arange(len(img_conf)), alpha=0.5, s=1, cmap='PiYG')
        plt.imshow(np.log(img_array), extent=(x1, x2, y1, y2), origin='lower')
        plt.colorbar()

        plt.xlim(x1, x2)
        plt.ylim(y1, y2)

        plt.savefig("/tmp/fig2.png")
        plt.close()
        img2 = Image.open("/tmp/fig2.png")
        return img, img2, img_conf
    
    return img, img_conf 


def train_epoch(epoch, network, train_loader, loss_function, optimiser, num_atoms, dataset, device, verbose=False):
    network.train()
    
    batch_size = train_loader.batch_size // 2

    for batch in tqdm(train_loader, desc=f"Training epoch #{epoch}...", leave=False):
        x = batch.to(device)

        x0, x1 = x.split(batch_size)
        optimiser.zero_grad()

        #encode
        z0 = network.encode(x0)
        z1 = network.encode(x1)

        #interpolate
        alpha = torch.rand(batch_size, 1, 1).to(device)
        z_interpolated = (1-alpha)*z0 + alpha*z1

        #decode
        out0 = network.decode(z0)[:,:,:num_atoms]
        out1 = network.decode(z1)[:,:,:num_atoms]
        out_interpolated = network.decode(z_interpolated)[:,:,:num_atoms]

        #calculate MSE
        mse_loss_0 = ((x0-out0)**2).mean() # reconstructive loss (Mean square error)
        mse_loss_1 = ((x1-out1)**2).mean() # reconstructive loss (Mean square error)
        out0 *= dataset.stdval
        out1 *= dataset.stdval
        out_interpolated *= dataset.stdval
        
        mse_loss = (mse_loss_0 + mse_loss_1) / 2

        #calculate physics for interpolated samples
        bond_energy, angle_energy, torsion_energy, NB_energy = loss_function.get_loss(out_interpolated)
        
        #by being enclosed in torch.no_grad() torch autograd cannot see where this scaling
        #factor came from and hence although mathematically the physics cancels, no gradients
        #are found and the scale is simply redefined at each step
        #item  ~ torch.no_grad
        
        with torch.no_grad():
            scale = 0.1*mse_loss.item()/(bond_energy.item()+angle_energy.item()+torsion_energy.item()+NB_energy.item())

        network_loss = mse_loss + scale*(bond_energy + angle_energy + torsion_energy + NB_energy)
        
        if verbose:
            wandb.log(dict(
                mse_loss=mse_loss.item(),
                phys_loss=(bond_energy + angle_energy + torsion_energy + NB_energy).item(),
                bond_energy=bond_energy.item(),
                angle_energy=angle_energy.item(),
                torsion_energy=torsion_energy.item(),
                NB_energy=NB_energy.item(),
                network_loss=network_loss.item(),
            ))
        
        #determine gradients
        network_loss.backward()

        #advance the network weights
        optimiser.step()
        
    return network_loss


def validation(epoch, network, test0, test1, dataset, num_atoms, dataloader, 
               loss_function, device, mol, pdbs_dir, img_size=100):
    #encode test with each network
    #Not training so switch to eval mode
    network.eval()
    
    interpolation_out = torch.zeros(20, num_atoms, 3)
    interpolated_points = torch.zeros(20, 2)
    
    with torch.no_grad(): # don't need gradients for this bit
        test0_z = network.encode(test0.unsqueeze(0).float())
        test1_z = network.encode(test1.unsqueeze(0).float())

        #interpolate between the encoded Z space for each network between test0 and test1
        for idx, t in enumerate(np.linspace(0, 1, 20)):
            point = float(t)*test0_z + (1-float(t))*test1_z
            interpolated_points[idx] = point.squeeze().cpu().numpy()
            interpolation_out[idx] = network.decode(point)[:,:,:num_atoms].squeeze(0).permute(1,0).cpu().data
        interpolation_out *= dataset.stdval
        
    img, log_img, points = visualization(network, dataset, num_atoms, loss_function, dataloader, device, size=img_size, log_scale=True)
    wandb.log({"img": wandb.Image(img, caption=f"epoch_{epoch:0>4}")})
    wandb.log({"log_img": wandb.Image(log_img, caption=f"epoch_{epoch:0>4}")})
    np.save(f'{checkpoints_dir}/frames2D_epoch_{epoch:0>4}.pdb', points)

    #save interpolations
    mol = dataset.mol
    mol.coordinates = interpolation_out.numpy()
    mol.write_pdb(f'{pdbs_dir}/epoch_{epoch:0>4}_interpolation.pdb')
    np.save(f'{checkpoints_dir}/points_epoch_{epoch:0>4}.pdb', interpolated_points)

def create_dirs(args):
    root = os.path.join(args.output, args.experiment_name)

    if not os.path.exists(root):
        os.mkdir(root)

    checkpoints_dir = os.path.join(root, 'checkpoints')
    # checkpoints_extra_dir = os.path.join(ROOT, 'checkpoints_extra')
    # conformations_dir = os.path.join(ROOT, 'conformations')
    pdbs_dir = os.path.join(root, 'pdbs')
    # weights_dir = os.path.join(ROOT, 'weights')

    # if not os.path.exists(checkpoints_extra_dir):
    #     os.mkdir(checkpoints_extra_dir)

    if not os.path.exists(checkpoints_dir):
        os.mkdir(checkpoints_dir)
        
    # if not os.path.exists(conformations_dir):
    #     os.mkdir(conformations_dir)

    if not os.path.exists(pdbs_dir):
        os.mkdir(pdbs_dir)

    # if not os.path.exists(weights_dir):
    #     os.mkdir(weights_dir)

    return root, checkpoints_dir, pdbs_dir

def main(args):
    root, checkpoints_dir, pdbs_dir = create_dirs(args)

    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

    # dataset = torch.load(f'{conformations_dir}/dataset.pt') 
    # meanval = torch.load(f'{conformations_dir}/meanval.pt')
    # stdval = torch.load(f'{conformations_dir}/stdval.pt')
    #atom_names = torch.load(f'{conformations_dir}/atom_names.pt')

    # atom_names = torch.load(f'{DATA}/all_walkers_each200/conformations/atom_names.pt')
    batch_size = 32 # if this is too small, gpu utilization goes down
    epoch = 0
    method = 'roll'
    atoms = ["CA", "C", "N", "CB", "O"]

    if os.path.exists(f'{root}/mean_std.npy'):
        meanval, stdval = np.load(f'{root}/mean_std.npy')
    else:
        meanval, stdval = calculate_mean_std(dataset)
        np.save(f'{root}/mean_std.npy', (meanval, stdval))

    dataset = FramesDataset(
        args.dataset,
        atoms=atoms,
        
    )

    lf = Auto_potential(
        frame=dataset[0]*dataset.stdval, 
        pdb_atom_names=dataset.atom_names, 
        method=method, 
        device=device
    )

    num_atoms = dataset[0].shape[1]

    #with respect to rmsd: 
    # test0_idx, test1_idx = check_rmsd(args.traj)
    # test0, test1 = dataset[test0_idx], dataset[test1_idx]
    
    df = pd.read_csv(args.qcharge_file, index_col = None)
    q_max_idx, q_max_val, q_min_idx, q_min_val = charge_dif(df)
    test0, test1 = dataset[q_min_idx], dataset[q_max_idx]

    test0 = test0.to(device)
    test1 = test1.to(device)
    
    cfg = dict(
        batch_size=batch_size,
        learning_rate=0.001,
        num_epochs=200,
    )

    num_workers = os.cpu_count()

    train_loader = torch.utils.data.DataLoader(dataset,
                batch_size=2 * cfg["batch_size"], shuffle=True, drop_last=True, num_workers=num_workers)

    val_loader = torch.utils.data.DataLoader(dataset,
                batch_size=2 * cfg["batch_size"], shuffle=False, drop_last=False, num_workers=num_workers)


    network = Autoencoder(m=2.0, latent_z=2, r=2, sigmoid=False, BN=True).to(device)
    # network = torch.DataParallel(network)
    # network.encode = network.module.encode
    # network.decode = network.module.decode

    #Sending to W&B
    if args.wandb:
        wandb.init(project="molearn", entity="jk")
        wandb.config = cfg

    optimizer = torch.optim.Adam(network.parameters(), lr=cfg["learning_rate"], amsgrad=True)

    #training loop
    for epoch in trange(200, desc="Training..."):
        network_loss = train_epoch(
            epoch=epoch,
            network=network,
            train_loader=train_loader,
            loss_function=lf,
            optimiser=optimizer,
            num_atoms=num_atoms,
            dataset=dataset,
            device=device,
            verbose=args.wandb,
        )

        #save interpolations between test0 and test1 every 5 epochs
        if args.verbose and (epoch + 1) % 5 == 0:
            validation(
                epoch=epoch,
                network=network,
                test0=test0,
                test1=test1,
                dataset=dataset,
                num_atoms=num_atoms,
                dataloader=val_loader,
                loss_function=lf,
                device=device,
                img_size=100,
                pdbs_dir=pdbs_dir,
            )

        torch.save(network.state_dict(), f'{checkpoints_dir}/epoch_{epoch:0>4}_{network_loss.item():.5}.pth')
        
        #if extra training 
        #torch.save(network.state_dict(), f'{checkpoints_extra_dir}/epoch_{epoch:0>4}_{network_loss.item():.5}.pth')
        
        epoch+=1

if __name__ == "__main__":
    parser = ArgumentParser()

    parser.add_argument("--dataset", "-i", type=str, default="/home/ebam/kacher1/doc/sum_traj/split")
    parser.add_argument("--output", "-o", type=str, default="/home/ebam/kacher1/molearn/DATA")
    parser.add_argument("--experiment-name", "-e", type=str, default="all_kv1.2")
    parser.add_argument("--wandb", "-v", action="store_true", default=False)
    parser.add_argument("--qcharge_file", "-q", type=str, default=False)
    parser.add_argument("traj", "-t", type=str, default=False)

    args = parser.parse_args()
    main(args)