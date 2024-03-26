import torch
import torch.nn.functional as F

num_bins = 100
dnase_seq = torch.rand(num_bins)

for _ in range(10):
  dnase_seq = F.conv1d(dnase_seq, torch.ones(10)/10, padding=5)

dnase_seq += 0.5 * torch.rand(num_bins)

torch.save(dnase_seq, '../data/dnase_seq.pt')