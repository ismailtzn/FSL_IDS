import torch
import torch.nn as nn
import torch.nn.functional

import utility


class Flatten(nn.Module):
    def __init__(self):
        super(Flatten, self).__init__()

    # noinspection PyMethodMayBeStatic
    def forward(self, x):
        return x.view(x.size(0), -1)


class ProtoNet(nn.Module):
    def __init__(self, encoder, n_way=5, n_support=5, n_query=5):
        super(ProtoNet, self).__init__()
        self.encoder = encoder.cuda()
        self.n_way = n_way
        self.n_support = n_support
        self.n_query = n_query

    def pre_process_meta_sample(self, meta_sample):
        sample_data = meta_sample.cuda()

        with torch.no_grad():
            x_support = sample_data[:, :self.n_support]
            x_query = sample_data[:, self.n_support:]

            # encode data of the support and the query set
            example_dimension = x_support.size()[2:]
            x = torch.cat([x_support.contiguous().view(self.n_way * self.n_support, *example_dimension),
                           x_query.contiguous().view(self.n_way * self.n_query, *example_dimension)], 0)
        return x

    def forward(self, x):
        return self.encoder.forward(x)

    def get_protonet_loss_accuracy(self, outputs):
        outputs_dim = outputs.size(-1)
        z_proto = outputs[:self.n_way * self.n_support].view(self.n_way, self.n_support, outputs_dim).mean(1)
        z_query = outputs[self.n_way * self.n_support:]

        # compute distances
        dists = utility.euclidean_dist(z_query, z_proto)

        # compute probabilities
        log_p_y = torch.nn.functional.log_softmax(-dists, dim=1).view(self.n_way, self.n_query, -1)

        # target indices are 0 ... self.n_way-1
        target_inds = torch.arange(0, self.n_way).view(self.n_way, 1, 1).expand(self.n_way, self.n_query, 1).long()
        target_inds = target_inds.cuda()
        target_inds.requires_grad_(False)
        loss_val = -log_p_y.gather(2, target_inds).squeeze().view(-1).mean()
        _, y_hat = log_p_y.max(2)
        acc_val = torch.eq(y_hat, target_inds.squeeze()).float().mean()

        return loss_val, acc_val
