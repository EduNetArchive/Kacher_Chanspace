#Imports 
import os
import pdb
import wandb

import numpy as np
import pandas as pd

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim
import tensorflow as tf

from matplotlib import pyplot as plt
from PIL import Image
from tqdm import tqdm, trange
from argparse import ArgumentParser

import MDAnalysis as mda
from MDAnalysis.analysis import align, rms

import plotly as py
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from utils import chanset
from utils.positional_qpredictor import *
from molearn import Auto_potential, Autoencoder

#Defining functions 

@torch.no_grad() 
def visualize_energy(network, dataset, num_atoms, lf, device, size=100, bounding_box=(0,0,1,1)): #background = energy calculated by the decoder
    x1, y1, x2, y2 = bounding_box
    x = np.linspace(x1, x2, size)
    y = np.linspace(y1, y2, size)
    xx, yy = np.meshgrid(x, y) 
    z = np.stack((xx, yy), axis=2)
    z = z.reshape(size**2, 2, 1)

    losses_all = []
    for i in trange(len(z)):
        z_batch = z[i:(i+1)]
        z_batch=torch.tensor(z_batch).float().to(device)
        out = network.decode(z_batch)[:, :, :num_atoms]
        out *= dataset.stdval.reshape(3, 1).to(device)
        bond_energy, angle_energy, torsion_energy, NB_energy = lf.get_loss(out)

        loss_f = (bond_energy + angle_energy + torsion_energy + NB_energy)
        losses_all.append(loss_f.cpu().numpy())
    
    img = np.hstack(losses_all)
    img = img.reshape(size, size)

    return img

@torch.no_grad()
def visualize_qcharge(network, qpredictor, num_atoms, labelled_atoms, device, batchsize, size=100, bounding_box=(0,0,1,1)): #background = qcharges
    x1, y1, x2, y2 = bounding_box
    x = np.linspace(x1, x2, size)
    y = np.linspace(y1, y2, size)
    xx, yy = np.meshgrid(x, y) 
    z = np.stack((xx, yy), axis=2)
    z = z.reshape(size**2, 2, 1)

    qcharges_all = []
    for i in trange(0, len(z), batchsize):
        z_batch = z[i:(i+batchsize)]
        z_batch=torch.tensor(z_batch).float().to(device)
        
        out = network.decode(z_batch)[:, :, :num_atoms]
        # out *= dataset.stdval.reshape(3, 1).to(device)
        
        q_out = qpredictor(out, labelled_atoms[:out.shape[0]]) 
        qcharges_all.append(q_out.detach().cpu().numpy())
    
    q_array = np.hstack(qcharges_all)
    img_q = q_array.reshape(size, size)

    return img_q

def conformations_to_latent(network, train_loader, device): 
    z_list=[]
    with torch.no_grad():
        for batch in tqdm.tqdm(train_loader):
            x, *_ = batch
            x = x.to(device)
            z = network.encode(x)
            z_list.append(z.cpu().squeeze(2))
    
    z_list = torch.cat(z_list)
    
    return z_list 

def visualization(network, dataset, qpredictor, num_atoms, labelled_atoms, lf, train_loader, device, batchsize, size=100, bounding_box=None, log_scale=False):
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

        bb = (x1, x2, y1, y2)
    
    else:
        x1, y1, x2, y2 = bounding_box

    img_array = visualize_energy(network, dataset, num_atoms, lf, device,
                                 size=size, bounding_box=(x1, y1, x2, y2))

    plt.scatter(img_conf[:, 0], img_conf[:, 1], c=np.arange(len(img_conf)), alpha=0.5, s=1, cmap='PiYG')
    plt.imshow(img_array, extent=(x1, x2, y1, y2), origin='lower')
    plt.colorbar()

    plt.xlim(x1, x2)
    plt.ylim(y1, y2)

    plt.savefig("/tmp/fig1.png")
    plt.close()
    img = Image.open("/tmp/fig1.png")

    q_array = visualize_qcharge(network, qpredictor, num_atoms, labelled_atoms, device, batchsize, size=100, bounding_box=(0,0,1,1))
    plt.scatter(img_conf[:, 0], img_conf[:, 1], c=np.arange(len(img_conf)), alpha=0.5, s=1, cmap='PiYG')
    plt.imshow(q_array, extent=(x1, x2, y1, y2), origin='lower')
    plt.colorbar()

    plt.savefig("/tmp/fig3.png")
    plt.close()
    img_q = Image.open("/tmp/fig3.png")
    
    if log_scale:
        plt.scatter(img_conf[:, 0], img_conf[:,1], c=np.arange(len(img_conf)), alpha=0.5, s=1, cmap='PiYG')
        plt.imshow(np.log(img_array), extent=(x1, x2, y1, y2), origin='lower')
        plt.colorbar()

        plt.xlim(x1, x2)
        plt.ylim(y1, y2)

        plt.savefig("/tmp/fig2.png")
        plt.close()
        img2 = Image.open("/tmp/fig2.png")
        return img, img2, img_conf, img_array, bb, img_q
    
    return img, img_conf, img_array, bb, img_q 


# @profile
def train_epoch(epoch, network, qpredictor, train_loader, loss_function, optimiser, num_atoms, dataset, device, verbose=False):
    network.train()
    smoothed_loss = None
    smoothing_coef = 0.9 

    batch_size = train_loader.batch_size // 2
    print(len(train_loader))

    for batch in tqdm.tqdm(train_loader, desc=f"Training epoch #{epoch}...", leave=False):
        x, labelled_atoms, qcharges = [t.to(device) for t in batch]

        x0, x1 = x.split(batch_size)
        q0, q1 = qcharges.split(batch_size)
        labelled_atoms0, labelled_atoms1 = labelled_atoms.split(batch_size)
        optimiser.zero_grad()

        q0_predicted = qpredictor(x0, labelled_atoms0)
        q1_predicted = qpredictor(x1, labelled_atoms1)

        #enc
        #encode
        z0 = network.encode(x0)
        z1 = network.encode(x1)

        #interpolate
        alpha = torch.rand(batch_size, 1, 1).to(device)
        z_interpolated = (1-alpha)*z0 + alpha*z1
        q_interpolated = (1-alpha)*q0 + alpha*q1

        #decode
        out0 = network.decode(z0)[:,:,:num_atoms]
        out1 = network.decode(z1)[:,:,:num_atoms]
        out_interpolated = network.decode(z_interpolated)[:,:,:num_atoms]
        
        #q_out0 = qpredictor(out0, labelled_atoms0)
        #q_out1 = qpredictor(out1, labelled_atoms1)
        q_out_interpolated = qpredictor(out_interpolated, labelled_atoms0)
    
        #calculate MSE
        mse_loss_0 = ((x0-out0)**2).mean() # reconstructive loss (Mean square error)
        mse_loss_1 = ((x1-out1)**2).mean() # reconstructive loss (Mean square error)

        q0_mse = ((q0-q0_predicted)**2).mean() 
        q1_mse = ((q1-q1_predicted)**2).mean() 
        q_loss = (q0_mse + q1_mse) / 2
        
        out0 *= dataset.stdval.reshape(3, 1).to(device)
        out1 *= dataset.stdval.reshape(3, 1).to(device)
        out_interpolated *= dataset.stdval.reshape(3, 1).to(device)
        
        mse_loss = (mse_loss_0 + mse_loss_1) / 2
        q_mse_loss = ((q_interpolated-q_out_interpolated)**2).mean()

        #calculate physics for interpolated samples
        bond_energy, angle_energy, torsion_energy, NB_energy = loss_function.get_loss(out_interpolated)
        
        #by being enclosed in torch.no_grad() torch autograd cannot see where this scaling
        #factor came from and hence although mathematically the physics cancels, no gradients
        #are found and the scale is simply redefined at each step
        #item  ~ torch.no_grad
        
        with torch.no_grad():
            scale = args.scale*mse_loss.item()/(bond_energy.item()+angle_energy.item()+torsion_energy.item()+NB_energy.item())

        network_loss = mse_loss + args.q_scale*q_mse_loss + args.q_scale*q_loss + scale*(bond_energy + angle_energy + torsion_energy + NB_energy)

        if smoothed_loss is None: 
            smoothed_loss = network_loss.item()
        else: 
            smoothed_loss = (1-smoothing_coef)*network_loss.item() + smoothing_coef*smoothed_loss


        if verbose:
            wandb.log(dict(
                mse_loss=mse_loss.item(),
                q_mse = q_mse_loss.item(),
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
        
    return smoothed_loss, network_loss

# @profile
def validation(epoch, network, qpredictor, test0, test1, dataset, num_atoms, labelled_atoms, dataloader, 
               loss_function, device, checkpoints_dir, pdbs_dir, batchsize, verbose=False, img_size=100):
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
            interpolated_points[idx] = point.squeeze().cpu()
            interpolation_out[idx] = network.decode(point)[:,:,:num_atoms].squeeze(0).permute(1,0).cpu().data
        interpolation_out = interpolation_out.swapaxes(1,2)
        interpolation_out*= dataset.stdval
        interpolation_out = interpolation_out.numpy()

    interpolation_out = interpolation_out.clip(min=-999, max=9999)
    
    img, log_img, points, energy_array, bb, img_q = visualization(network, dataset, qpredictor, num_atoms, labelled_atoms, loss_function, dataloader, device, batchsize, size=img_size, log_scale=True)
    #np.save(f'{checkpoints_dir}/energy_{epoch}.npy', energy_array)
    #np.save(f'{checkpoints_dir}/bb_x1x2y1y2_{epoch}.npy', bb)
    
    if verbose:
        wandb.log({"img": wandb.Image(img, caption=f"epoch_{epoch:0>4}")})
        wandb.log({"log_img": wandb.Image(log_img, caption=f"epoch_{epoch:0>4}")})
        wandb.log({"img_q": wandb.Image(img_q, caption=f"epoch_{epoch:0>4}")})
    
    #np.save(f'{checkpoints_dir}/frames2D_epoch_{epoch:0>4}.npy', points)
    
    #save interpolations
    
    try: 
        dataset.write_pdb(interpolation_out)
    except ValueError: 
        print('Ooops ValueError, continue without writing coordinates')


    return points, interpolated_points, energy_array, bb, img_q, interpolation_out

def create_dirs(args):
    root = os.path.join(args.output, args.experiment_name)

    if not os.path.exists(root):
        os.mkdir(root)

    checkpoints_dir = os.path.join(root, 'checkpoints')
    checkpoints_extra_dir = os.path.join(root, 'checkpoints_extra')
    # conformations_dir = os.path.join(ROOT, 'conformations')
    pdbs_dir = os.path.join(root, 'pdbs')
    # weights_dir = os.path.join(ROOT, 'weights')

    if not os.path.exists(checkpoints_extra_dir):
        os.mkdir(checkpoints_extra_dir)

    if not os.path.exists(checkpoints_dir):
        os.mkdir(checkpoints_dir)
        
    # if not os.path.exists(conformations_dir):
    #     os.mkdir(conformations_dir)

    if not os.path.exists(pdbs_dir):
        os.mkdir(pdbs_dir)

    # if not os.path.exists(weights_dir):
    #     os.mkdir(weights_dir)

    return root, checkpoints_dir, checkpoints_extra_dir, pdbs_dir

def main(args):
    root, checkpoints_dir, checkpoints_extra_dir, pdbs_dir = create_dirs(args)
    if args.extra:
        checkpoints_dir=checkpoints_extra_dir
    
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    num_workers = args.num_workers

    batch_size = args.batchsize 
    print(f'batch_size is {batch_size}')

    epochs = args.epochs
    lr = args.lr

    epoch = 0
    method = 'roll'
    # atoms = ["CA", "C", "N", "CB", "O"] #args.atoms
    cfg = args.__dict__
   
    #autoencoder
    dataset = chanset.QDataset(frames = args.traj_path,
                                qcharge_path = args.qcharge_path,
                                topology = args.topology_path, 
                                atoms = args.atoms, 
                                skip = args.train_skip,
                                start = args.train_start,
                                stop = args.train_stop,
                                transform = args.transform,
                                plane = args.plane,
                                calculate_statistics=True, 
                                root = args.output,
                                exp_name = args.experiment_name,
                                extra = args.extra,
                                )

    print('Dataset is ', len(dataset), ' frames long')
    # if os.path.exists(f'{root}/mean_std.npy'):
    #     meanval, stdval = np.load(f'{root}/mean_std.npy', allow_pickle=True)
    # else:
    #     meanval, stdval = chanset.calculate_mean_std(dataset)
    #     np.save(f'{root}/mean_std.npy', (meanval, stdval), allow_pickle=True)

    num_unique_atoms = len(dataset.atomname_to_label)
    num_atoms = dataset.n_atoms
    labelled_atoms = dataset.labelled_atoms.to(device).unsqueeze(0).repeat_interleave(batch_size, dim=0)
    print(labelled_atoms.shape)
    
    if args.qcharge_path:
        test0, min_q, test1,  max_q, av_q = dataset.min_max_av_qcharge()
        print(f'Qmax is {max_q}, its index is {test0}; Qmin is {min_q}, its index is {test1}')
    
    else: 
        test0, test1, rmsd_value = dataset.max_rmsd_dif()
        print(f'RMSD max is {rmsd_value}, test frames are {test0} and {test1}')
    
    pdb_atom_names = np.vstack([dataset.atoms_names, dataset.resnames, dataset.resids])
    pdb_atom_names = pdb_atom_names.T

    lf = Auto_potential(
        frame=dataset[0][0]*dataset.stdval.reshape(3, 1),
        pdb_atom_names=pdb_atom_names, 
        method=method, 
        device=device
    )

    test0, test1 = dataset[test0][0], dataset[test1][0]
    test0 = test0.to(device)
    test1 = test1.to(device)
    
    train_loader = torch.utils.data.DataLoader(dataset,
                batch_size=2 * batch_size, shuffle=True, drop_last=True, num_workers=num_workers)
    print(len(train_loader))

    val_loader = torch.utils.data.DataLoader(dataset,
                batch_size=2 * batch_size, shuffle=False, drop_last=False, num_workers=num_workers)

    network = Autoencoder(m=2.0, latent_z=2, r=2, sigmoid=False, BN=True, parallel_mode=args.parallel).to(device)
    
    qpredictor = PositionalQPredictor(num_unique_atoms=num_unique_atoms, depth=4, scale=2, channels=32, res_n=2, droprate=None, batch_norm=True).to(device)
    

    if args.extra_path:
        network.load_state_dict(torch.load(args.extra_path),strict=False)
    
    #Sending to W&B
    if args.verbose:
        wandb.init(project="molearn", entity="jk", name=args.experiment_name)
        wandb.config = cfg

    optimizer = torch.optim.Adam(network.parameters(), lr=cfg["lr"], amsgrad=True)

    best_loss = torch.inf
    best_loss_name = None
    
    #training loop
    for epoch in trange(epochs, desc="Training..."):
        smoothed_loss, network_loss = train_epoch(
            epoch=epoch,
            network=network,
            qpredictor=qpredictor,
            train_loader=train_loader,
            loss_function=lf,
            optimiser=optimizer,
            num_atoms=num_atoms,
            dataset=dataset,
            device=device,
            verbose=args.verbose,
        )

        (
            points, 
            interpolated_points, 
            energy_array, 
            bb,
            img_q, 
            interpolation_out

        ) = validation(
            epoch=epoch,
            network=network,
            qpredictor=qpredictor,
            test0=test0,
            test1=test1,
            dataset=dataset,
            num_atoms=num_atoms,
            labelled_atoms=labelled_atoms,
            dataloader=val_loader,
            loss_function=lf,
            device=device,
            pdbs_dir=pdbs_dir,
            checkpoints_dir=checkpoints_dir,
            batchsize=batch_size,
            img_size=100,
            verbose=args.verbose,
        )

        if smoothed_loss < best_loss: 
            best_loss = smoothed_loss
            if best_loss_name is not None: 
                os.remove(best_loss_name)
            best_loss_name = f'{checkpoints_dir}/epoch_{epoch:0>4}_{smoothed_loss:.5}.pth'
            torch.save(network.state_dict(), best_loss_name)
            np.save(f'{checkpoints_dir}/frames2D.npy', points)
            np.save(f'{checkpoints_dir}/points.npy', interpolated_points)
            np.save(f'{checkpoints_dir}/bb_x1x2y1y2.npy', bb)
            np.save(f'{checkpoints_dir}/energy.npy', energy_array)
            np.save(f'{checkpoints_dir}/qcharges_background.npy', img_q)
            np.save(f'{checkpoints_dir}/decoded_coord.npy', interpolation_out)

        torch.save(network.state_dict(), f'{checkpoints_dir}/last_checkpoint.pth')
        
        #if extra training 
        #torch.save(network.state_dict(), f'{checkpoints_extra_dir}/epoch_{epoch:0>4}_{network_loss.item():.5}.pth')
        
        #epoch+=1

if __name__ == "__main__":
    parser = ArgumentParser()


    parser.add_argument("--output", "-o", type=str, default="/home/ebam/kacher1/molearn/DATA")
    parser.add_argument("--experiment_name", "-e", type=str, default="Kv1_qpositional")
    parser.add_argument("--verbose", "-v", action="store_true", default=False)
    parser.add_argument("--extra", "-d", action="store_true", default=False)
    parser.add_argument("--extra_path", type=str, default=None)

    parser.add_argument("--traj_path", "-i", type=str, default="/home/ebam/kacher1/Files/Kv1.2.VSD/preprocessed_pdb_traj/all_kv1.2.pdb")
    parser.add_argument("--topology_path", "-t", type=str, default=None)
    parser.add_argument("--qcharge_path", "-q", type=str, default="/home/ebam/kacher1/Files/Kv1.2.VSD/qcharges/qcharge.all.walkers.each200.csv")
    parser.add_argument("--atoms", "-a", type=str, default="protein and not name H*")

    parser.add_argument("--train_start", type=int, default=0)
    parser.add_argument("--train_stop", type=int, default=-1)
    parser.add_argument("--train_skip", "-s", type=int, default=1)
    parser.add_argument("--transform", action="store_true", default=False)
    parser.add_argument("--plane", type=str, default="xy")
    parser.add_argument("--val_start", type=int, default=0)
    parser.add_argument("--val_stop", type=int, default=-1)
    parser.add_argument("--val_skip", "-k", type=int, default=1)

    parser.add_argument("--num_workers", "-w", type=int, default=8)
    parser.add_argument("--batchsize", "-b", type=int, default=8)
    parser.add_argument("--lr", type=float, default=0.001)
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--parallel", "-p", action="store_true", default=False)
    parser.add_argument("--scale", "-l", type=float, default=0.1)
    parser.add_argument("--q_scale", "-c", type=float, default=0.1)


    args = parser.parse_args()
    main(args)