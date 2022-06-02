import torch
import torch.nn as nn
import torch.nn.functional
from torch.autograd import Variable

import utility


class Flatten(nn.Module):
    def __init__(self):
        super(Flatten, self).__init__()

    def forward(self, x):
        return x.view(x.size(0), -1)


class ProtoNet(nn.Module):
    def __init__(self, encoder):
        """
        Args:
            encoder : CNN encoding the data in sample
        """
        super(ProtoNet, self).__init__()
        self.encoder = encoder.cuda()
        print(self.encoder)

    def set_forward_loss(self, sample):
        """
        Computes loss, accuracy and output for classification task
        Args:
            sample (torch.Tensor): shape (n_way, n_support+n_query, (dim))
                n_way (int): number of classes in a classification task
                n_support (int): number of labeled examples per class in the support set
                n_query (int): number of labeled examples per class in the query set
        Returns:
            torch.Tensor: shape(2), loss, accuracy and y_hat
        """
        sample_data = sample["sample_data"].cuda()
        n_way = sample["n_way"]
        n_support = sample["n_support"]
        n_query = sample["n_query"]

        x_support = sample_data[:, :n_support]
        x_query = sample_data[:, n_support:]

        # target indices are 0 ... n_way-1
        target_inds = torch.arange(0, n_way).view(n_way, 1, 1).expand(n_way, n_query, 1).long()
        # target_inds = Variable(target_inds, requires_grad=False)
        target_inds = target_inds.cuda()
        target_inds.requires_grad_(False)

        # encode data of the support and the query set
        example_dimension = x_support.size()[2:]
        x = torch.cat([x_support.contiguous().view(n_way * n_support, *example_dimension),
                       x_query.contiguous().view(n_way * n_query, *example_dimension)], 0)

        z = self.encoder.forward(x)
        z_dim = z.size(-1)  # usually 64
        z_proto = z[:n_way * n_support].view(n_way, n_support, z_dim).mean(1)
        z_query = z[n_way * n_support:]

        # compute distances
        dists = utility.euclidean_dist(z_query, z_proto)

        # compute probabilities
        log_p_y = torch.nn.functional.log_softmax(-dists, dim=1).view(n_way, n_query, -1)

        loss_val = -log_p_y.gather(2, target_inds).squeeze().view(-1).mean()
        _, y_hat = log_p_y.max(2)
        acc_val = torch.eq(y_hat, target_inds.squeeze()).float().mean()

        return loss_val, {
            "loss": loss_val.item(),
            "acc": acc_val.item(),
            "y_hat": y_hat
        }
