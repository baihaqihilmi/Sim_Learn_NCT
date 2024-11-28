import torch
import torch.nn as nn


class ContrastiveLoss(nn.Module):
    def __init__(self, margin=1.0):
        super(ContrastiveLoss, self).__init__()
        self.margin = margin
 
    def forward(self, output1, output2, label):
        # Compute Euclidean distance
        euclidean_distance = torch.nn.functional.pairwise_distance(output1, output2)

        # Contrastive loss formula
        loss = torch.mean((label) * torch.pow(euclidean_distance, 2) +
                          (1 - label) * torch.pow(torch.clamp(self.margin - euclidean_distance, min=0.0), 2))
        return loss

if __name__ == "__main__":
    ContrastiveLoss()