import torch
import torch.nn as nn
import torch.nn.functional
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix


class RelNetEncoder(nn.Module):
    def __init__(self, x_dim0=1, hid_dim=128, out_dim=64):
        super(RelNetEncoder, self).__init__()

        self.layer1 = nn.Sequential(
            nn.Conv1d(x_dim0, hid_dim, 3, padding=1),
            nn.BatchNorm1d(hid_dim),
            nn.ReLU(),
            nn.MaxPool1d(2)
        )
        self.layer2 = nn.Sequential(
            nn.Conv1d(hid_dim, hid_dim, 3, padding=1),
            nn.BatchNorm1d(hid_dim),
            nn.ReLU(),
            nn.MaxPool1d(2)
        )
        self.layer3 = nn.Sequential(
            nn.Conv1d(hid_dim, hid_dim, 3, padding=1),
            nn.BatchNorm1d(hid_dim),
            nn.MaxPool1d(2)
        )
        self.layer4 = nn.Sequential(
            nn.Conv1d(hid_dim, out_dim, 3, padding=1),
            nn.BatchNorm1d(out_dim),
            nn.MaxPool1d(2)
        )

    def forward(self, x):
        out = self.layer1(x)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        return out


class RelationNetwork(nn.Module):

    def __init__(self, input_size=128, hidden_size=64):
        super(RelationNetwork, self).__init__()

        self.layer1 = nn.Sequential(
            nn.Conv1d(input_size, hidden_size, kernel_size=3, padding=1),
            nn.BatchNorm1d(hidden_size, momentum=1, affine=True),
            nn.ReLU(),
            nn.MaxPool1d(2))
        self.layer2 = nn.Sequential(
            nn.Conv1d(hidden_size, hidden_size, kernel_size=3, padding=1),
            nn.BatchNorm1d(hidden_size, momentum=1, affine=True),
            nn.ReLU(),
            nn.MaxPool1d(2))
        self.fc1 = nn.Linear(hidden_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, 1)

    def forward(self, x):
        out = self.layer1(x)
        out = self.layer2(out)
        out = out.view(out.size(0), -1)
        out = torch.nn.functional.relu(self.fc1(out))
        out = torch.sigmoid(self.fc2(out))
        return out


class RelationNet(nn.Module):
    def __init__(self, feature_encoder, relation_network, n_way=5, n_support=5, n_query=5):
        super(RelationNet, self).__init__()
        self.feature_encoder = feature_encoder.cuda()
        self.relation_network = relation_network.cuda()
        self.n_way = n_way
        self.n_support = n_support
        self.n_query = n_query
        self.support_set = None

    def forward(self, x_query, support_set=None):
        if support_set is None:
            support_set = self.support_set
        else:
            self.support_set = support_set

        if support_set is None:
            raise ValueError("support_set should not be None!")

        # TODO::: Maybe use no grad for support set encoding
        # with torch.no_grad():
        example_dimension = support_set.size()[2:]
        self.support_set = self.support_set.view(self.n_way * self.n_support, *example_dimension)
        support_set_encoded = self.feature_encoder.forward(self.support_set)
        encoded_dimension = support_set_encoded.size()[1:]
        support_set_encoded = support_set_encoded.view(self.n_way, self.n_support, *encoded_dimension)
        support_set_encoded = torch.sum(support_set_encoded, 1)
        support_set_encoded = torch.mul(support_set_encoded, (1 / self.n_support))

        x_query = x_query.view(self.n_way * self.n_query, *example_dimension)
        query_set_encoded = self.feature_encoder(x_query)

        support_set_encoded_ext = support_set_encoded.unsqueeze(0).repeat(self.n_query * self.n_way, 1, 1, 1)
        query_set_encoded_ext = query_set_encoded.unsqueeze(0).repeat(self.n_way, 1, 1, 1)
        query_set_encoded_ext = torch.transpose(query_set_encoded_ext, 0, 1)

        relation_pairs = torch.cat((support_set_encoded_ext, query_set_encoded_ext), 2).view(-1, encoded_dimension[0] * 2, *encoded_dimension[1:])
        relations = self.relation_network(relation_pairs).view(-1, self.n_way)

        return relations

    def get_relation_net_loss_metrics(self, outputs, y_true, support_set_included=True, class_labels=None):
        mse = nn.MSELoss().cuda()
        one_hot_labels = torch.zeros(self.n_query * self.n_way, self.n_way).cuda().scatter_(1, y_true.view(-1, 1), 1)
        loss = mse(outputs, one_hot_labels)

        _, y_hat = outputs.max(1)

        y_true_np = y_true.detach().cpu().squeeze().numpy().flatten()
        y_hat_np = y_hat.detach().cpu().numpy().flatten()

        metrics = classification_report(y_true_np, y_hat_np, target_names=class_labels, output_dict=True, zero_division=0)
        cf_matrix = confusion_matrix(y_true_np, y_hat_np)
        metrics["loss"] = loss.item()

        return loss, metrics, cf_matrix
