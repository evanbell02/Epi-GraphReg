import sys
from pathlib import Path
sys.path.append(str(Path().parent.resolve().parent.resolve()))
import torch
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
import pytorch_lightning as pl
import random

def make_epi(num_bins=1000, silencers=False):
    num_genes = num_bins//50
    num_regulators = num_bins//50
    
    dnase_seq = torch.zeros(num_bins)
    
    rand_indices = torch.randperm(num_bins)
    gene_indices = rand_indices[:num_genes]
    enhancer_indices = rand_indices[-num_regulators:]
    silencer_indices = torch.tensor([], dtype=torch.int32)
    if silencers:
        enhancer_indices = rand_indices[-num_regulators:-num_regulators//2]
        silencer_indices = rand_indices[-num_regulators//2:]
    
    all_indices = torch.cat([gene_indices, enhancer_indices, silencer_indices])
    
    while torch.min(all_indices) <= 5 or torch.max(all_indices) >= num_bins-5:
        rand_indices = torch.randperm(num_bins)
        gene_indices = rand_indices[:num_genes]
        enhancer_indices = rand_indices[-num_regulators:]
        silencer_indices = torch.tensor([], dtype=torch.int32)
        if silencers:
            enhancer_indices = rand_indices[-num_regulators:-num_regulators//2]
            silencer_indices = rand_indices[-num_regulators//2:]
        
        all_indices = torch.cat([gene_indices, enhancer_indices, silencer_indices])
    
    dnase_seq[gene_indices] = 1
    dnase_seq[enhancer_indices] = 1
    
    for idx in gene_indices:
        dnase_seq[idx-2:idx+3] += (2 * torch.rand(1))**2 * torch.rand(5)
    
    for idx in enhancer_indices:
        dnase_seq[idx-2:idx+3] += (2 * torch.rand(1))**2 * torch.rand(5)

    for idx in silencer_indices:
        dnase_seq[idx-2:idx+3] += (torch.rand(1))**2 * torch.rand(5)

    dnase_seq = F.conv1d(dnase_seq.unsqueeze(0), torch.ones(1,1,5)/5, padding=2).squeeze()
    
    dnase_seq += 0.5 * torch.rand_like(dnase_seq) * (torch.rand_like(dnase_seq) > 0.9).float()

    return dnase_seq, gene_indices, enhancer_indices, silencer_indices, num_genes, num_regulators

def make_graph(dnase_seq, gene_indices, enhancer_indices, silencer_indices, num_genes, num_regulators):
    edges = []
    for g in gene_indices:
        enhancers = random.sample([*enhancer_indices], 3)
        for e in enhancers:
            edges.append([e,g])
        if len(silencer_indices) >= 2:
            silencers = random.sample([*silencer_indices], 2)
            for s in silencers:
                edges.append([s,g])
    edges = torch.tensor(edges).T
    return edges

def make_cage(dnase_seq, gene_indices, enhancer_indices, silencer_indices, num_genes, num_regulators, edges):
    cage_seq = torch.zeros_like(dnase_seq)
    for offset in range(-5, 6):
        for e in edges.T:
            if e[0] in enhancer_indices:
                cage_seq[e[1]+offset] += dnase_seq[e[0]+offset]
            if e[0] in silencer_indices:
                cage_seq[e[1]+offset] -= dnase_seq[e[0]+offset]
    cage_seq = F.relu(cage_seq)
    cage_seq += 0.15 * cage_seq.max() * torch.rand_like(cage_seq) * (torch.rand_like(cage_seq) > 0.9).float()
    return cage_seq

def main():
    torch.manual_seed(0)

    for i in range(1,21):
        dnase_seq, gene_indices, enhancer_indices, silencer_indices, num_genes, num_regulators = make_epi()
        edges = make_graph(dnase_seq, gene_indices, enhancer_indices, silencer_indices, num_genes, num_regulators)
        cage_seq = make_cage(dnase_seq, gene_indices, enhancer_indices, silencer_indices, num_genes, num_regulators, edges)
        
        torch.save({'dnase_seq': dnase_seq,
                   'gene_indices': gene_indices,
                   'enhancer_indices': enhancer_indices,
                   'silencer_indices': silencer_indices,
                   'num_genes': num_genes,
                   'num_regulators': num_regulators,
                   'edges': edges,
                   'cage_seq': cage_seq},
                  f'./data/enhancers_only/train_data/chromosome_{i}.pt')
        
    for i in range(1,6):
        dnase_seq, gene_indices, enhancer_indices, silencer_indices, num_genes, num_regulators = make_epi()
        edges = make_graph(dnase_seq, gene_indices, enhancer_indices, silencer_indices, num_genes, num_regulators)
        cage_seq = make_cage(dnase_seq, gene_indices, enhancer_indices, silencer_indices, num_genes, num_regulators, edges)
        
        torch.save({'dnase_seq': dnase_seq,
                   'gene_indices': gene_indices,
                   'enhancer_indices': enhancer_indices,
                   'silencer_indices': silencer_indices,
                   'num_genes': num_genes,
                   'num_regulators': num_regulators,
                   'edges': edges,
                   'cage_seq': cage_seq},
                  f'./data/enhancers_only/val_data/chromosome_{i}.pt')
    
    for i in range(1,6):
        dnase_seq, gene_indices, enhancer_indices, silencer_indices, num_genes, num_regulators = make_epi()
        edges = make_graph(dnase_seq, gene_indices, enhancer_indices, silencer_indices, num_genes, num_regulators)
        cage_seq = make_cage(dnase_seq, gene_indices, enhancer_indices, silencer_indices, num_genes, num_regulators, edges)
        
        torch.save({'dnase_seq': dnase_seq,
                   'gene_indices': gene_indices,
                   'enhancer_indices': enhancer_indices,
                   'silencer_indices': silencer_indices,
                   'num_genes': num_genes,
                   'num_regulators': num_regulators,
                   'edges': edges,
                   'cage_seq': cage_seq},
                  f'./data/enhancers_only/test_data/chromosome_{i}.pt')

    for i in range(1,21):
        dnase_seq, gene_indices, enhancer_indices, silencer_indices, num_genes, num_regulators = make_epi(silencers=True)
        edges = make_graph(dnase_seq, gene_indices, enhancer_indices, silencer_indices, num_genes, num_regulators)
        cage_seq = make_cage(dnase_seq, gene_indices, enhancer_indices, silencer_indices, num_genes, num_regulators, edges)
        
        torch.save({'dnase_seq': dnase_seq,
                   'gene_indices': gene_indices,
                   'enhancer_indices': enhancer_indices,
                   'silencer_indices': silencer_indices,
                   'num_genes': num_genes,
                   'num_regulators': num_regulators,
                   'edges': edges,
                   'cage_seq': cage_seq},
                  f'./data/enhancers_and_silencers/train_data/chromosome_{i}.pt')
        
    for i in range(1,6):
        dnase_seq, gene_indices, enhancer_indices, silencer_indices, num_genes, num_regulators = make_epi(silencers=True)
        edges = make_graph(dnase_seq, gene_indices, enhancer_indices, silencer_indices, num_genes, num_regulators)
        cage_seq = make_cage(dnase_seq, gene_indices, enhancer_indices, silencer_indices, num_genes, num_regulators, edges)
        
        torch.save({'dnase_seq': dnase_seq,
                   'gene_indices': gene_indices,
                   'enhancer_indices': enhancer_indices,
                   'silencer_indices': silencer_indices,
                   'num_genes': num_genes,
                   'num_regulators': num_regulators,
                   'edges': edges,
                   'cage_seq': cage_seq},
                  f'./data/enhancers_and_silencers/val_data/chromosome_{i}.pt')
    
    for i in range(1,6):
        dnase_seq, gene_indices, enhancer_indices, silencer_indices, num_genes, num_regulators = make_epi(silencers=True)
        edges = make_graph(dnase_seq, gene_indices, enhancer_indices, silencer_indices, num_genes, num_regulators)
        cage_seq = make_cage(dnase_seq, gene_indices, enhancer_indices, silencer_indices, num_genes, num_regulators, edges)
        
        torch.save({'dnase_seq': dnase_seq,
                   'gene_indices': gene_indices,
                   'enhancer_indices': enhancer_indices,
                   'silencer_indices': silencer_indices,
                   'num_genes': num_genes,
                   'num_regulators': num_regulators,
                   'edges': edges,
                   'cage_seq': cage_seq},
                  f'./data/enhancers_and_silencers/test_data/chromosome_{i}.pt')

if __name__ == "__main__":
    main()