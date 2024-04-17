import sys
sys.path.append('/egr/research-slim/belleva1/EpiGraphReg/')
import torch
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
import torch_geometric
from torch_geometric.data import Data
import pytorch_lightning as pl
import random

def make_epi(num_bins=1000):
    num_genes = num_bins//50
    num_promoters = num_bins//50
    
    dnase_seq = torch.zeros(num_bins)
    
    rand_indices = torch.randperm(num_bins)
    gene_indices = rand_indices[:num_genes]
    promoter_indices = rand_indices[-num_promoters:]
    all_indices = torch.cat([gene_indices, promoter_indices])

    
    while torch.min(all_indices) <= 5 or torch.max(all_indices) >= num_bins-5:
        rand_indices = torch.randperm(num_bins)
        gene_indices = rand_indices[:num_genes]
        promoter_indices = rand_indices[-num_promoters:]
        all_indices = torch.cat([gene_indices, promoter_indices])
    
    dnase_seq[gene_indices] = 1
    dnase_seq[promoter_indices] = 1
    
    for idx in gene_indices:
        dnase_seq[idx-2:idx+3] += (2 * torch.rand(1))**2 * torch.rand(5)
    
    for idx in promoter_indices:
        dnase_seq[idx-2:idx+3] += (2 * torch.rand(1))**2 * torch.rand(5)

    dnase_seq = F.conv1d(dnase_seq.unsqueeze(0), torch.ones(1,1,5)/5, padding=2).squeeze()
    
    dnase_seq += 0.5 * torch.rand_like(dnase_seq) * (torch.rand_like(dnase_seq) > 0.9).float()

    return dnase_seq, gene_indices, promoter_indices, num_genes, num_promoters

def make_epi(num_bins=1000):
    num_genes = num_bins//50
    num_promoters = num_bins//50
    
    dnase_seq = torch.zeros(num_bins)
    
    rand_indices = torch.randperm(num_bins)
    gene_indices = rand_indices[:num_genes]
    promoter_indices = rand_indices[-num_promoters:]
    all_indices = torch.cat([gene_indices, promoter_indices])

    
    while torch.min(all_indices) <= 5 or torch.max(all_indices) >= num_bins-5:
        rand_indices = torch.randperm(num_bins)
        gene_indices = rand_indices[:num_genes]
        promoter_indices = rand_indices[-num_promoters:]
        all_indices = torch.cat([gene_indices, promoter_indices])
    
    dnase_seq[gene_indices] = 1
    dnase_seq[promoter_indices] = 1
    
    for idx in gene_indices:
        dnase_seq[idx-2:idx+3] += (2 * torch.rand(1))**2 * torch.rand(5)
    
    for idx in promoter_indices:
        dnase_seq[idx-2:idx+3] += (2 * torch.rand(1))**2 * torch.rand(5)

    dnase_seq = F.conv1d(dnase_seq.unsqueeze(0), torch.ones(1,1,5)/5, padding=2).squeeze()
    
    dnase_seq += 0.5 * torch.rand_like(dnase_seq) * (torch.rand_like(dnase_seq) > 0.9).float()

    return dnase_seq, gene_indices, promoter_indices, num_genes, num_promoters

def make_graph(dnase_seq, gene_indices, promoter_indices, num_genes, num_promoters):
    edges = []
    for g in gene_indices:
        promoters = random.sample([*promoter_indices], 3)
        for p in promoters:
            edges.append([p,g])
    edges = torch.tensor(edges).T
    edge_attr = torch.tensor([[0,1] for _ in range(len(gene_indices))])
    return edges, edge_attr

def make_cage(dnase_seq, gene_indices, promoter_indices, num_genes, edges):
    cage_seq = torch.zeros_like(dnase_seq)
    for offset in range(-5, 6):
        for e in edges.T:
            cage_seq[e[1]+offset] += dnase_seq[e[0]+offset]
    cage_seq += 0.15 * cage_seq.max() * torch.rand_like(cage_seq) * (torch.rand_like(cage_seq) > 0.9).float()
    return cage_seq

def main():
    torch.manual_seed(0)

    for i in range(1,21):
        dnase_seq, gene_indices, promoter_indices, num_genes, num_promoters = make_epi()
        edges, edge_attr = make_graph(dnase_seq, gene_indices, promoter_indices, num_genes, num_promoters)
        cage_seq = make_cage(dnase_seq, gene_indices, promoter_indices, num_genes, edges)
        
        torch.save({'dnase_seq': dnase_seq,
                   'gene_indices': gene_indices,
                   'promoter_indices': promoter_indices,
                   'num_genes': num_genes,
                   'num_promoters': num_promoters,
                   'edges': edges,
                   'edge_attr': edge_attr,
                   'cage_seq': cage_seq},
                  f'../data/train_data/chromosome_{i}.pt')
        
    for i in range(1, 6):
        dnase_seq, gene_indices, promoter_indices, num_genes, num_promoters = make_epi()
        edges, edge_attr = make_graph(dnase_seq, gene_indices, promoter_indices, num_genes, num_promoters)
        cage_seq = make_cage(dnase_seq, gene_indices, promoter_indices, num_genes, edges)
        
        torch.save({'dnase_seq': dnase_seq,
                   'gene_indices': gene_indices,
                   'promoter_indices': promoter_indices,
                   'num_genes': num_genes,
                   'num_promoters': num_promoters,
                   'edges': edges,
                   'edge_attr': edge_attr,
                   'cage_seq': cage_seq},
                  f'../data/val_data/chromosome_{i}.pt')
    
    for i in range(1, 6):
        dnase_seq, gene_indices, promoter_indices, num_genes, num_promoters = make_epi()
        edges, edge_attr = make_graph(dnase_seq, gene_indices, promoter_indices, num_genes, num_promoters)
        cage_seq = make_cage(dnase_seq, gene_indices, promoter_indices, num_genes, edges)
        
        torch.save({'dnase_seq': dnase_seq,
                   'gene_indices': gene_indices,
                   'promoter_indices': promoter_indices,
                   'num_genes': num_genes,
                   'num_promoters': num_promoters,
                   'edges': edges,
                   'edge_attr': edge_attr,
                   'cage_seq': cage_seq},
                  f'../data/test_data/chromosome_{i}.pt')

if __name__ == "__main__":
    main()