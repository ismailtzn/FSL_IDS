import torch
import torch.nn as nn
import torch.nn.functional
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix

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
        self.z_proto = None
        self.z_labels = None
        self.support_set = None

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

    def forward(self, x_query, support_set=None):
        if support_set is None:
            support_set = self.support_set
        else:
            self.support_set = support_set

        if support_set is None:
            raise ValueError("support_set should not be None!")

        with torch.no_grad():
            # encode data of the support and the query set
            example_dimension = support_set.size()[2:]
            x = torch.cat([support_set.contiguous().view(self.n_way * self.n_support, *example_dimension),
                           x_query.contiguous().view(self.n_way * self.n_query, *example_dimension)], 0)

        return self.encoder.forward(x)

    def get_protonet_loss_metrics(self, outputs, y_true, support_set_included=True, class_labels=None):
        outputs_dim = outputs.size(-1)
        if support_set_included:
            z_proto = outputs[:self.n_way * self.n_support].view(self.n_way, self.n_support, outputs_dim).mean(1)
            z_query = outputs[self.n_way * self.n_support:]
        else:
            z_proto = self.z_proto
            z_query = outputs

        # compute distances
        dists = utility.euclidean_dist(z_query, z_proto)

        # compute probabilities
        log_p_y = torch.nn.functional.log_softmax(-dists, dim=1).view(self.n_way, self.n_query, -1)
        loss_val = -log_p_y.gather(2, y_true).squeeze().view(-1).mean()
        _, y_hat = log_p_y.max(2)

        y_true_np = y_true.detach().cpu().squeeze().numpy().flatten()
        y_hat_np = y_hat.detach().cpu().numpy().flatten()

        metrics = classification_report(y_true_np, y_hat_np, target_names=class_labels, output_dict=True, zero_division=0)
        cf_matrix = confusion_matrix(y_true_np, y_hat_np)
        # acc_val = torch.eq(y_hat, y_true.squeeze()).float().mean()
        metrics["loss"] = loss_val.item()

        return loss_val, metrics, cf_matrix

    def set_encoded_prototypes(self, support_set, labels):
        n_way = support_set.shape[0]
        n_support = support_set.shape[1]
        support_set = support_set.cuda()
        with torch.no_grad():
            example_dimension = support_set.size()[2:]
            support_set = support_set.contiguous().view(n_way * n_support, *example_dimension)

        outputs = self.encoder.forward(support_set)
        outputs_dim = outputs.size(-1)
        self.z_proto = outputs.view(n_way, n_support, outputs_dim).mean(1)
        self.z_labels = labels

    def predict(self, query):
        """
        Requires batch of queries and returns predicted label indexes
        Args:
            query: Queried samples (q, *FeatureDim)

        Returns:
            predicted indexes
        """
        if (self.z_proto is None) or (self.z_labels is None):
            raise Exception("Prototype encodings is not ready!")

        z_query = self.encoder.forward(query)

        dists = utility.euclidean_dist(z_query, self.z_proto)
        # compute probabilities
        log_p_y = torch.nn.functional.log_softmax(-dists, dim=1)
        _, y_hats = log_p_y.max(1)
        return y_hats
