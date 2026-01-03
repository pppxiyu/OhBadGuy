import numpy as np
import torch
import torch.nn as nn
import torch_geometric as pyG
from torch_geometric.nn import GCNConv
from torch_geometric.nn import ChebConv
import random
import networkx as nx

import heapq
from collections import defaultdict
from typing import Dict, List, Tuple

class CrimeTorchDataset(torch.utils.data.Dataset):
    def __init__(self, x, y, standardize=False):
        super().__init__()
        self.x = torch.from_numpy(x).to(torch.float32)
        self.y = torch.from_numpy(y).to(torch.float32)
        self.standardize = standardize
        if self.standardize:
            self.mean_max_x = np.max(x[:, -1, :, 0], axis=1).mean()

    def __len__(self):
        return len(self.x)

    def __getitem__(self, idx):
        x = self.x[idx]
        y = self.y[idx]
        if self.standardize:
            assert ~(x.max(dim=1, keepdim=True)[0] == 0).any(), 'No crime in a week. 0-dividing issue.'
            assert ~(y.max(dim=1, keepdim=True)[0] == 0).any(), 'No crime in a week. 0-dividing issue.'
            x = x / x.max(dim=1, keepdim=True)[0]
            y = y / y.max(dim=1, keepdim=True)[0]
        return x, y


class CrimePyGDataset(pyG.data.Dataset):
    def __init__(self, graphs):
        super().__init__()
        self.graphs = graphs

    def len(self):
        return len(self.graphs)

    def get(self, idx):
        return self.graphs[idx]


class CrimeCountSoftRound(torch.nn.Module):
    def __init__(self, alpha, eps=1e-3):
        super(CrimeCountSoftRound, self).__init__()
        self.alpha = alpha
        self.eps = eps
        assert self.alpha >= self.eps, 'alpha must be >= eps'
        self.b = torch.nn.Parameter(torch.randint(0, 3, (1,), dtype=torch.float32))

    def forward(self, x):
        alpha_bounded = torch.tensor(self.alpha, dtype=x.dtype, device=x.device)
        m = torch.floor(x) + 0.5
        r = x - m
        z = torch.tanh(alpha_bounded / 2) * 2
        y = m + torch.tanh(alpha_bounded * r) / z
        return y


class CrimePred:
    def __init__(self, ):
        self.polygon = None
        self.adj_matrix = None
        self.edge_index = None
        self.edge_weight = None
        self.adj_mode = None
        self.node_count = None

        self.train_dataset = None
        self.val_dataset = None
        self.test_dataset = None

        self.model = None
        self.readout = None

        self.device = (
            "cuda"
            if torch.cuda.is_available()
            else "mps"
            if torch.backends.mps.is_available()
            else "cpu"
        )

    def build_adj_from_polygons(self, polygon, mode='graph', knn_n=None, road_geo=None):
        from scipy.sparse import csr_matrix
        from torch_geometric.utils import from_scipy_sparse_matrix

        self.polygon = polygon
        if mode == 'Queen':
            adj_matrix = self.polygon.geometry.apply(lambda geom: self.polygon.geometry.touches(geom)).values
            adj_matrix = np.array(adj_matrix, dtype=int)

        elif mode == 'KNN':
            assert knn_n is not None, 'Missing knn_n'
            from sklearn.neighbors import kneighbors_graph
            coordinates = np.array(list(self.polygon.geometry.centroid.apply(lambda p: (p.x, p.y))))
            adj_matrix = kneighbors_graph(coordinates, n_neighbors=knn_n, mode='connectivity', include_self=False)

        elif mode == 'graph':
            assert road_geo is not None, 'Missing road_geo'
            from shapely.geometry import LineString, Point
            
            # Create a mapping from id to index for quick lookup
            id_to_idx = {row['id']: idx for idx, row in self.polygon.iterrows()}
            
            # Initialize adjacency matrix
            n = len(self.polygon)
            adj_matrix = np.zeros((n, n), dtype=int)
            
            # Build a dictionary mapping road endpoints to road ids
            endpoint_to_roads = {}
            for idx, row in road_geo.iterrows():
                road_id = row['id']
                geom = row['geometry']
                
                if isinstance(geom, LineString):
                    # Get the start and end points of the road
                    start_point = Point(geom.coords[0])
                    end_point = Point(geom.coords[-1])
                    
                    # Round coordinates to avoid floating point precision issues
                    start_key = (round(start_point.x, 8), round(start_point.y, 8))
                    end_key = (round(end_point.x, 8), round(end_point.y, 8))
                    
                    # Add road_id to the endpoint dictionary
                    if start_key not in endpoint_to_roads:
                        endpoint_to_roads[start_key] = []
                    endpoint_to_roads[start_key].append(road_id)
                    
                    if end_key not in endpoint_to_roads:
                        endpoint_to_roads[end_key] = []
                    endpoint_to_roads[end_key].append(road_id)
            
            # Build adjacency based on shared endpoints
            for endpoint, road_ids in endpoint_to_roads.items():
                if len(road_ids) > 1:
                    # These roads are connected at this endpoint
                    for i in range(len(road_ids)):
                        for j in range(i + 1, len(road_ids)):
                            road_i = road_ids[i]
                            road_j = road_ids[j]
                            
                            # Get polygon indices
                            if road_i in id_to_idx and road_j in id_to_idx:
                                idx_i = id_to_idx[road_i]
                                idx_j = id_to_idx[road_j]
                                
                                # Mark as adjacent (undirected graph)
                                adj_matrix[idx_i, idx_j] = 1
                                adj_matrix[idx_j, idx_i] = 1
        else:
            raise NotImplementedError

        self.adj_matrix = adj_matrix
        edge_index, edge_weight = from_scipy_sparse_matrix(csr_matrix(adj_matrix))
        self.edge_index = edge_index.long()
        self.edge_weight = edge_weight.float()
        self.adj_mode = mode
        self.node_count = adj_matrix.shape[0]
        return

    def build_dataset(self, train_x_ar, train_y_ar, val_x_ar, val_y_ar, test_x_ar, test_y_ar):
        self.train_dataset = CrimeTorchDataset(train_x_ar, train_y_ar, True)
        self.val_dataset = CrimeTorchDataset(val_x_ar, val_y_ar, True)
        self.test_dataset = CrimeTorchDataset(test_x_ar, test_y_ar, True)
        return

    def build_model(self, in_channels, out_channels, model_name='TGCN', **kwargs):
        self.reset_model()
        if model_name == 'TGCN':
            self.model = ResTGCN(in_channels=in_channels, out_channels=out_channels, ).to(self.device)
        elif model_name == 'GConvGRU':
            self.model = ResGConvGRU(in_channels=in_channels, out_channels=out_channels, K=kwargs['K']).to(self.device)
        elif model_name == 'GConvGRU_NoRes':
            self.model = GConvGRU(in_channels=in_channels, out_channels=out_channels, K=kwargs['K']).to(self.device)
        elif model_name == 'GConvGRU_NoConv':
            self.model = ResGRU(in_channels=in_channels, out_channels=out_channels,).to(self.device)
        else:
            raise NotImplementedError
        self.readout = torch.nn.Sequential(
            torch.nn.Linear(1, 1, bias=True),
        ).to(self.device)
        return

    def reset_model(self, ):
        del self.model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    def _crime_seq_collate_fn(self, batch):
        x_full = torch.stack([item[0] for item in batch])
        y_full = torch.stack([item[1] for item in batch])
        x_full = torch.transpose(x_full, 0, 1)
        y_full = torch.transpose(y_full, 0, 1)

        x_temporal = [t for t in x_full]
        temporal_dataset = [
            CrimePyGDataset([
                pyG.data.Data(x=x, edge_index=self.edge_index, edge_attr=self.edge_weight, y=y) for x, y in
                zip(t, y_full[0])
            ]) for t in x_temporal
        ]
        temporal_dataloader = [
            pyG.loader.DataLoader(t, batch_size=len(t), shuffle=False) for t in temporal_dataset
        ]
        temporal_batchdata = [next(iter(loader)) for loader in temporal_dataloader]
        return temporal_batchdata  # a list of BatchData

    def train_model(self, batch_size=32, lr=0.001, epoch=1000, dir_cache='./', test_loc=999):
        import os
        from torch.utils.data import DataLoader
        train_dataloader = DataLoader(
            self.train_dataset, batch_size=batch_size, collate_fn=self._crime_seq_collate_fn, shuffle=True
        )
        val_dataloader = DataLoader(self.val_dataset, batch_size=batch_size, collate_fn=self._crime_seq_collate_fn)

        loss_func = torch.nn.MSELoss()
        optimizer = torch.optim.Adam(self.model.parameters(), lr=lr)
        early_stopper = EarlyStopper(patience=5, min_delta=1e-5)

        for e in range(epoch):
            print(f"Epoch {e + 1}\n-------------------------------")
            self.model.train()
            self.readout.train()
            for i, batch in enumerate(train_dataloader):
                h = None
                y = batch[0].y.to(self.device)
                for snapshot in batch:
                    x = snapshot.x.to(self.device)
                    h = self.model(x, snapshot.edge_index.to(self.device), snapshot.edge_attr.to(self.device), h)
                out = self.readout(h)
                loss = loss_func(out, y)
                loss.backward()
                optimizer.step()
                optimizer.zero_grad()
                if i % 3 == 0:
                    loss, current = loss.item(), (i + 1) * batch_size
                    print(f"loss: {loss:>7f}  [{current:>5d}/{len(train_dataloader.dataset):>5d}]")

            self.model.eval()
            self.readout.eval()
            val_loss = 0
            with torch.no_grad():
                for _, val_batch in enumerate(val_dataloader):
                    val_h = None
                    val_y = val_batch[0].y.to(self.device)
                    for val_snapshot in val_batch:
                        val_x = val_snapshot.x.to(self.device)
                        val_h = self.model(
                            val_x, val_snapshot.edge_index.to(self.device), val_snapshot.edge_attr.to(self.device),
                            val_h
                        )
                    pred = self.readout(val_h)
                    val_loss += loss_func(pred, val_y).item()
            val_loss /= len(val_dataloader)
            print(f"val loss: {val_loss:>8f} \n")

            early_stopper.stopper(val_loss)
            if early_stopper.stop:
                break
        print("Done!")

        best_val_loss = 1e8
        for file in os.listdir(dir_cache):
            if file.startswith("model_main_") and file.endswith(f"{test_loc}.pth"):
                best_val_loss = float(file.split("model_main_")[1].split("_")[0])
                break
        if val_loss < best_val_loss:
            for filename in os.listdir(dir_cache):
                if filename.startswith('model_') and filename.endswith(".pth"):
                    if float(filename.split("model_")[1].split("_")[2].replace('.pth', '')) == test_loc:
                        file_path = os.path.join(dir_cache, filename)
                        os.remove(file_path)
            torch.save(self.model, f'{dir_cache}/model_main_{val_loss:>8f}_{test_loc}.pth')
            torch.save(self.readout, f'{dir_cache}/model_readout_{val_loss:>8f}_{test_loc}.pth')
        return val_loss

    def pred_crime_test_set(self, dir_cache, test_loc='', verbose=1):
        import os
        from torch.utils.data import DataLoader
        test_dataloader = DataLoader(
            self.test_dataset, batch_size=len(self.test_dataset), collate_fn=self._crime_seq_collate_fn
        )

        for filename in os.listdir(dir_cache):
            if filename.startswith('model_main') and filename.endswith(f"{test_loc}.pth"):
                model = torch.load(f'{dir_cache}/{filename}', weights_only=False).to(self.device)
            if filename.startswith('model_readout') and filename.endswith(f"{test_loc}.pth"):
                readout = torch.load(f'{dir_cache}/{filename}', weights_only=False).to(self.device)

        with torch.no_grad():
            for _, test_batch in enumerate(test_dataloader):
                test_h = None
                test_y = test_batch[0].y
                for test_snapshot in test_batch:
                    test_x = test_snapshot.x.to(self.device)
                    test_h = model(
                        test_x, test_snapshot.edge_index.to(self.device), test_snapshot.edge_attr.to(self.device),
                        test_h
                    )
                pred = readout(test_h)
        if verbose > 0:
            print(f"Test set predicted.")

        num_graphs = test_batch[0].batch.max().item() + 1
        pred_out = np.array([pred.cpu().numpy()[test_batch[0].batch == i] for i in range(num_graphs)])
        test_y_out = np.array([test_y.cpu().numpy()[test_batch[0].batch == i] for i in range(num_graphs)])
        return pred_out, test_y_out

    @staticmethod
    def convert_pred_2_int(pred, scaler):
        pred_scaled = pred * scaler
        return pred_scaled


class EarlyStopper:
    def __init__(self, patience=5, min_delta=0):
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.min_val_loss = np.inf
        self.stop = False

    def stopper(self, val_loss):
        if val_loss < (self.min_val_loss - self.min_delta):
            self.min_val_loss = val_loss
            self.counter = 0
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.stop = True


class ResTGCN(torch.nn.Module):
    r"""
    Obtained from Pytorch Geometric Temporal. Refer to their docs for annotations.
    Graph Convolutional Gated Recurrent Cell.
    """

    def __init__(
            self,
            in_channels: int,
            out_channels: int,
            improved: bool = False,
            cached: bool = False,
            add_self_loops: bool = True,
    ):
        super(ResTGCN, self).__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.improved = improved
        self.cached = cached
        self.add_self_loops = add_self_loops

        self._create_parameters_and_layers()

    def _create_update_gate_parameters_and_layers(self):
        self.conv_z = GCNConv(
            in_channels=self.in_channels,
            out_channels=self.out_channels,
            improved=self.improved,
            cached=self.cached,
            add_self_loops=self.add_self_loops,
        )

        self.linear_z = torch.nn.Linear(2 * self.out_channels, self.out_channels)

    def _create_reset_gate_parameters_and_layers(self):
        self.conv_r = GCNConv(
            in_channels=self.in_channels,
            out_channels=self.out_channels,
            improved=self.improved,
            cached=self.cached,
            add_self_loops=self.add_self_loops,
        )

        self.linear_r = torch.nn.Linear(2 * self.out_channels, self.out_channels)

    def _create_candidate_state_parameters_and_layers(self):
        self.conv_h = GCNConv(
            in_channels=self.in_channels,
            out_channels=self.out_channels,
            improved=self.improved,
            cached=self.cached,
            add_self_loops=self.add_self_loops,
        )

        self.linear_h = torch.nn.Linear(2 * self.out_channels, self.out_channels)

    def _create_parameters_and_layers(self):
        self._create_update_gate_parameters_and_layers()
        self._create_reset_gate_parameters_and_layers()
        self._create_candidate_state_parameters_and_layers()

    def _set_hidden_state(self, X, H):
        if H is None:
            H = torch.zeros(X.shape[0], self.out_channels).to(X.device)
        return H

    def _calculate_update_gate(self, X, edge_index, edge_weight, H):
        X_conv_z = self.conv_z(X, edge_index, edge_weight)
        Z = torch.cat([X_conv_z, H], axis=1)
        Z = self.linear_z(Z) + X_conv_z
        Z = torch.sigmoid(Z)
        return Z

    def _calculate_reset_gate(self, X, edge_index, edge_weight, H):
        X_conv_r = self.conv_r(X, edge_index, edge_weight)
        R = torch.cat([X_conv_r, H], axis=1)
        R = self.linear_r(R) + X_conv_r
        R = torch.sigmoid(R)
        return R

    def _calculate_candidate_state(self, X, edge_index, edge_weight, H, R):
        H_last = self.conv_h(X, edge_index, edge_weight)
        H_tilde = torch.cat([H_last, H * R], axis=1)
        H_tilde = self.linear_h(H_tilde)
        H_tilde = torch.tanh(H_tilde) + H_last
        return H_tilde

    def _calculate_hidden_state(self, Z, H, H_tilde):
        H = Z * H + (1 - Z) * H_tilde
        return H

    def forward(
            self,
            X: torch.FloatTensor,
            edge_index: torch.LongTensor,
            edge_weight: torch.FloatTensor = None,
            H: torch.FloatTensor = None,
    ) -> torch.FloatTensor:
        H = self._set_hidden_state(X, H)
        Z = self._calculate_update_gate(X, edge_index, edge_weight, H)
        R = self._calculate_reset_gate(X, edge_index, edge_weight, H)
        H_tilde = self._calculate_candidate_state(X, edge_index, edge_weight, H, R)
        H = self._calculate_hidden_state(Z, H, H_tilde)
        return H


class ResGConvGRU(torch.nn.Module):
    r"""
    Obtained from Pytorch Geometric Temporal. Refer to their docs for annotations.
     Chebyshev Graph Convolutional Gated Recurrent Unit Cell.
    """

    def __init__(
            self,
            in_channels: int,
            out_channels: int,
            K: int,
            normalization: str = "sym",
            bias: bool = True,
    ):
        super(ResGConvGRU, self).__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.K = K
        self.normalization = normalization
        self.bias = bias
        self._create_parameters_and_layers()

    def _create_update_gate_parameters_and_layers(self):
        self.conv_x_z = ChebConv(
            in_channels=self.in_channels,
            out_channels=self.out_channels,
            K=self.K,
            normalization=self.normalization,
            bias=self.bias,
        )

        self.conv_h_z = ChebConv(
            in_channels=self.out_channels,
            out_channels=self.out_channels,
            K=self.K,
            normalization=self.normalization,
            bias=self.bias,
        )

    def _create_reset_gate_parameters_and_layers(self):
        self.conv_x_r = ChebConv(
            in_channels=self.in_channels,
            out_channels=self.out_channels,
            K=self.K,
            normalization=self.normalization,
            bias=self.bias,
        )

        self.conv_h_r = ChebConv(
            in_channels=self.out_channels,
            out_channels=self.out_channels,
            K=self.K,
            normalization=self.normalization,
            bias=self.bias,
        )

    def _create_candidate_state_parameters_and_layers(self):
        self.conv_x_h = ChebConv(
            in_channels=self.in_channels,
            out_channels=self.out_channels,
            K=self.K,
            normalization=self.normalization,
            bias=self.bias,
        )

        self.conv_h_h = ChebConv(
            in_channels=self.out_channels,
            out_channels=self.out_channels,
            K=self.K,
            normalization=self.normalization,
            bias=self.bias,
        )

    def _create_parameters_and_layers(self):
        self._create_update_gate_parameters_and_layers()
        self._create_reset_gate_parameters_and_layers()
        self._create_candidate_state_parameters_and_layers()

    def _set_hidden_state(self, X, H):
        if H is None:
            H = torch.zeros(X.shape[0], self.out_channels).to(X.device)
        return H

    def _calculate_update_gate(self, X, edge_index, edge_weight, H, lambda_max):
        Z = self.conv_x_z(X, edge_index, edge_weight, lambda_max=lambda_max)
        Z = Z + self.conv_h_z(H, edge_index, edge_weight, lambda_max=lambda_max)
        Z = torch.sigmoid(Z)
        return Z

    def _calculate_reset_gate(self, X, edge_index, edge_weight, H, lambda_max):
        R = self.conv_x_r(X, edge_index, edge_weight, lambda_max=lambda_max)
        R = R + self.conv_h_r(H, edge_index, edge_weight, lambda_max=lambda_max)
        R = torch.sigmoid(R)
        return R

    def _calculate_candidate_state(self, X, edge_index, edge_weight, H, R, lambda_max):
        # H_tilde = self.conv_x_h(X, edge_index, edge_weight, lambda_max=lambda_max)
        # H_tilde = H_tilde + self.conv_h_h(H * R, edge_index, edge_weight, lambda_max=lambda_max)
        # H_tilde = torch.tanh(H_tilde)
        H_tilde_1 = self.conv_x_h(X, edge_index, edge_weight, lambda_max=lambda_max)
        H_tilde_2 = self.conv_h_h(H * R, edge_index, edge_weight, lambda_max=lambda_max)
        H_tilde = H_tilde_1 + H_tilde_2
        H_tilde = torch.tanh(H_tilde) + H_tilde_1 + H_tilde_2
        return H_tilde

    def _calculate_hidden_state(self, Z, H, H_tilde):
        H = Z * H + (1 - Z) * H_tilde
        return H

    def forward(
            self,
            X: torch.FloatTensor,
            edge_index: torch.LongTensor,
            edge_weight: torch.FloatTensor = None,
            H: torch.FloatTensor = None,
            lambda_max: torch.Tensor = None,
    ) -> torch.FloatTensor:
        H = self._set_hidden_state(X, H)
        Z = self._calculate_update_gate(X, edge_index, edge_weight, H, lambda_max)
        R = self._calculate_reset_gate(X, edge_index, edge_weight, H, lambda_max)
        H_tilde = self._calculate_candidate_state(X, edge_index, edge_weight, H, R, lambda_max)
        H = self._calculate_hidden_state(Z, H, H_tilde)
        return H


class GConvGRU(torch.nn.Module):
    r"""
    A downgraded version of ResGConvGRU without residual connections.
    """

    def __init__(
            self,
            in_channels: int,
            out_channels: int,
            K: int,
            normalization: str = "sym",
            bias: bool = True,
    ):
        super(GConvGRU, self).__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.K = K
        self.normalization = normalization
        self.bias = bias
        self._create_parameters_and_layers()

    def _create_update_gate_parameters_and_layers(self):
        self.conv_x_z = ChebConv(
            in_channels=self.in_channels,
            out_channels=self.out_channels,
            K=self.K,
            normalization=self.normalization,
            bias=self.bias,
        )

        self.conv_h_z = ChebConv(
            in_channels=self.out_channels,
            out_channels=self.out_channels,
            K=self.K,
            normalization=self.normalization,
            bias=self.bias,
        )

    def _create_reset_gate_parameters_and_layers(self):
        self.conv_x_r = ChebConv(
            in_channels=self.in_channels,
            out_channels=self.out_channels,
            K=self.K,
            normalization=self.normalization,
            bias=self.bias,
        )

        self.conv_h_r = ChebConv(
            in_channels=self.out_channels,
            out_channels=self.out_channels,
            K=self.K,
            normalization=self.normalization,
            bias=self.bias,
        )

    def _create_candidate_state_parameters_and_layers(self):
        self.conv_x_h = ChebConv(
            in_channels=self.in_channels,
            out_channels=self.out_channels,
            K=self.K,
            normalization=self.normalization,
            bias=self.bias,
        )

        self.conv_h_h = ChebConv(
            in_channels=self.out_channels,
            out_channels=self.out_channels,
            K=self.K,
            normalization=self.normalization,
            bias=self.bias,
        )

    def _create_parameters_and_layers(self):
        self._create_update_gate_parameters_and_layers()
        self._create_reset_gate_parameters_and_layers()
        self._create_candidate_state_parameters_and_layers()

    def _set_hidden_state(self, X, H):
        if H is None:
            H = torch.zeros(X.shape[0], self.out_channels).to(X.device)
        return H

    def _calculate_update_gate(self, X, edge_index, edge_weight, H, lambda_max):
        Z = self.conv_x_z(X, edge_index, edge_weight, lambda_max=lambda_max)
        Z = Z + self.conv_h_z(H, edge_index, edge_weight, lambda_max=lambda_max)
        Z = torch.sigmoid(Z)
        return Z

    def _calculate_reset_gate(self, X, edge_index, edge_weight, H, lambda_max):
        R = self.conv_x_r(X, edge_index, edge_weight, lambda_max=lambda_max)
        R = R + self.conv_h_r(H, edge_index, edge_weight, lambda_max=lambda_max)
        R = torch.sigmoid(R)
        return R

    def _calculate_candidate_state(self, X, edge_index, edge_weight, H, R, lambda_max):
        H_tilde = self.conv_x_h(X, edge_index, edge_weight, lambda_max=lambda_max)
        H_tilde = H_tilde + self.conv_h_h(H * R, edge_index, edge_weight, lambda_max=lambda_max)
        H_tilde = torch.tanh(H_tilde)
        # H_tilde_1 = self.conv_x_h(X, edge_index, edge_weight, lambda_max=lambda_max)
        # H_tilde_2 = self.conv_h_h(H * R, edge_index, edge_weight, lambda_max=lambda_max)
        # H_tilde = H_tilde_1 + H_tilde_2
        # H_tilde = torch.tanh(H_tilde) + H_tilde_1 + H_tilde_2
        return H_tilde

    def _calculate_hidden_state(self, Z, H, H_tilde):
        H = Z * H + (1 - Z) * H_tilde
        return H

    def forward(
            self,
            X: torch.FloatTensor,
            edge_index: torch.LongTensor,
            edge_weight: torch.FloatTensor = None,
            H: torch.FloatTensor = None,
            lambda_max: torch.Tensor = None,
    ) -> torch.FloatTensor:
        H = self._set_hidden_state(X, H)
        Z = self._calculate_update_gate(X, edge_index, edge_weight, H, lambda_max)
        R = self._calculate_reset_gate(X, edge_index, edge_weight, H, lambda_max)
        H_tilde = self._calculate_candidate_state(X, edge_index, edge_weight, H, R, lambda_max)
        H = self._calculate_hidden_state(Z, H, H_tilde)
        return H


class ResGRU(torch.nn.Module):
    r"""
    A downgraded version of ResGConvGRU without graph convolutions.
    """

    def __init__(
            self,
            in_channels: int,
            out_channels: int,
            normalization: str = "sym",
            bias: bool = True,
    ):
        super(ResGRU, self).__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.normalization = normalization
        self.bias = bias
        self._create_parameters_and_layers()

    def _create_update_gate_parameters_and_layers(self):
        self.conv_x_z = nn.Linear(self.in_channels, self.out_channels, bias=self.bias)
        self.conv_h_z = nn.Linear(self.out_channels, self.out_channels, bias=self.bias)

    def _create_reset_gate_parameters_and_layers(self):
        self.conv_x_r = nn.Linear(self.in_channels, self.out_channels, bias=self.bias)
        self.conv_h_r = nn.Linear(self.out_channels, self.out_channels, bias=self.bias)

    def _create_candidate_state_parameters_and_layers(self):
        self.conv_x_h = nn.Linear(self.in_channels, self.out_channels, bias=self.bias)
        self.conv_h_h = nn.Linear(self.out_channels, self.out_channels, bias=self.bias)

    def _create_parameters_and_layers(self):
        self._create_update_gate_parameters_and_layers()
        self._create_reset_gate_parameters_and_layers()
        self._create_candidate_state_parameters_and_layers()

    def _set_hidden_state(self, X, H):
        if H is None:
            H = torch.zeros(X.shape[0], self.out_channels).to(X.device)
        return H

    def _calculate_update_gate(self, X, H,):
        Z = self.conv_x_z(X,)
        Z = Z + self.conv_h_z(H,)
        Z = torch.sigmoid(Z)
        return Z

    def _calculate_reset_gate(self, X, H,):
        R = self.conv_x_r(X,)
        R = R + self.conv_h_r(H,)
        R = torch.sigmoid(R)
        return R

    def _calculate_candidate_state(self, X, H, R,):
        H_tilde_1 = self.conv_x_h(X,)
        H_tilde_2 = self.conv_h_h(H * R,)
        H_tilde = H_tilde_1 + H_tilde_2
        H_tilde = torch.tanh(H_tilde) + H_tilde_1 + H_tilde_2
        return H_tilde

    def _calculate_hidden_state(self, Z, H, H_tilde):
        H = Z * H + (1 - Z) * H_tilde
        return H

    def forward(
            self,
            X: torch.FloatTensor,
            edge_index: torch.LongTensor,
            edge_weight: torch.FloatTensor = None,
            H: torch.FloatTensor = None,
            lambda_max: torch.Tensor = None,
    ) -> torch.FloatTensor:
        H = self._set_hidden_state(X, H)
        Z = self._calculate_update_gate(X, H,)
        R = self._calculate_reset_gate(X, H,)
        H_tilde = self._calculate_candidate_state(X, H, R,)
        H = self._calculate_hidden_state(Z, H, H_tilde)
        return H


class CrimePredTuner(CrimePred):
    def __init__(self, model_name, storage_path="sqlite:///optuna_study.db", model_save='./'):
        import datetime
        super().__init__()
        self.storage_path = storage_path
        self.model_name = model_name
        self.study_name = f"{model_name}_{datetime.datetime.now().strftime('%m%d%H%M')}"
        self.model_save = model_save

    def objective(self, trial):
        if self.model_name == 'TGCN':
            batch_size = trial.suggest_categorical("batch_size", [8, 16, 32, 64])
            lr = trial.suggest_float("lr", 1e-4, 1e-2, log=True)
            self.build_model(1, 1, model_name=self.model_name)
        elif self.model_name == 'GConvGRU':
            batch_size = 8  # trial.suggest_categorical("batch_size", [8, 16])
            lr = trial.suggest_float("lr", 1e-3, 1e-2, log=True)
            K = trial.suggest_int("K", 1, 3)
            self.build_model(1, 1, model_name=self.model_name, K=K)
        elif self.model_name == 'GConvGRU_NoRes':
            batch_size = 8  # trial.suggest_categorical("batch_size", [8, 16])
            lr = trial.suggest_float("lr", 1e-3, 1e-1, log=True)
            K = trial.suggest_int("K", 1, 3)
            self.build_model(1, 1, model_name=self.model_name, K=K)
        elif self.model_name == 'GConvGRU_NoConv':
            batch_size = 8  # trial.suggest_categorical("batch_size", [8, 16])
            lr = trial.suggest_float("lr", 1e-4, 1e-2, log=True)
            self.build_model(1, 1, model_name=self.model_name,)
        else:
            raise NotImplementedError
        val_loss = self.train_model(batch_size=batch_size, lr=lr, dir_cache=self.model_save, test_loc=999)
        return val_loss

    def run_study(self, n_trials=50, ):
        import optuna
        study = optuna.create_study(
            direction="minimize", storage=self.storage_path, study_name=self.study_name
        )
        study.optimize(self.objective, n_trials=n_trials)
        print("Best hyperparameters:", study.best_params)
        return study.best_params


class SensorPlacement_legacy:
    def __init__(self, crime, road_data, road_graph, sim_iter, random_seed):
        self.crime = crime
        self.crime_loc = self.build_start_points(crime)
        self.road_data = road_data
        self.road_graph = road_graph
        self.sim_iter_count = sim_iter
        random.seed(random_seed)

        self.neighbors_by_direction = None
        self.records_by_roads = None
        self.records_by_iter = None

    @staticmethod
    def change_random_seed_to(random_seed):
        random.seed(random_seed)

    @staticmethod
    def build_start_points(origins):
        start_points = []
        loc = 1
        for item in origins:
            if item is not None and item > 0:
                start_points.append(loc)
            loc += 1
        return start_points

    def _move_to_crime(self, node_num):
        # OUTPUT: list of nodes
        # randomly choose road according to population
        roads = [item[0] for item in list(self.road_graph.nodes(data='pop'))]
        pop = [item[1] for item in list(self.road_graph.nodes(data='pop'))]
        pops_dist = pop / sum(pop)
        random_node = random.choices(roads, pops_dist)[0]
        route = nx.shortest_path(self.road_graph, weight='weight', source=random_node, target=node_num)
        return route

    def _split_neighbors_by_direction(self, connected_roads, buffer=0, lost_warning=False):
        # USE: assign the connected roads to the two end points of the road for analysis
        # INPUT: dict, the connected roads for a road
        # OUTPUT: a dict
        from shapely import Point

        geo = self.road_data[self.road_data['id'] == list(connected_roads.keys())[0]].geometry.values[0]
        try:
            coords = geo.coords
        except:  # When lines are disconnected, data problem
            coords = [geo.geoms[0].coords[0], geo.geoms[-1].coords[-1]]
            if lost_warning:
                print(
                    "Remedial measures are taken to extract end points from road segments. "
                    "Please note there are problems (i.e., lines are not connected in the same group) "
                    "in the data, although they are solved for this time."
                )

        # assign the connected roads to the two end points of the road in analysis
        c1 = []
        c2 = []
        for road in list(connected_roads.values())[0]:
            connected_geo = self.road_data[self.road_data['id'] == road].geometry.values[0]
            if connected_geo.distance(Point(coords[0])) <= buffer:  # if the connected road is closer to the cor 0
                c1.append(road)
            elif connected_geo.distance(
                    Point(coords[-1])) <= buffer:  # if the connected road is closer to the cor 1
                c2.append(road)
            else:
                if lost_warning:
                    print('Road', road, 'is not attached to any vetice of the target road',
                          list(connected_roads.keys())[0], '.')
        return {'v1': c1, 'v2': c2}

    def _split_inward_outward_neighbors(self):
        road_clusters_by_direction = {k + 1: None for k in range(self.road_graph.number_of_nodes())}
        for road in range(self.road_graph.number_of_nodes()):
            road_neighbor = list(nx.neighbors(self.road_graph, road + 1))
            roads_connected = {road + 1: road_neighbor}
            road_clusters = self._split_neighbors_by_direction(roads_connected)
            road_clusters_by_direction[road + 1] = road_clusters
            self.neighbors_by_direction = road_clusters_by_direction
        return road_clusters_by_direction

    def _placement_elimination(
            self, placements, records_by_roads, records_by_iterations, with_effect_ratio=False
    ):
        # Get movements from the placement.
        inflow = [[i, placements['Road']] for i in placements['Inflow roads']]
        outflow = [[placements['Road'], i] for i in placements['Outflow roads']]
        valid_flows = inflow + outflow

        # Check if any of the movements exist in records_by_iterations.
        # If so, get the index of simulation iteration and the index of suspect vehicle.
        delete_list = []
        for iteration in range(len(records_by_iterations)):
            reco_iteration = records_by_iterations[iteration + 1]
            for origin in list(reco_iteration.keys()):
                for vehicle, route in zip(
                        list(reco_iteration[origin].keys()),
                        list(reco_iteration[origin].values())
                ):
                    # tell if the route contains the valid flows indicated by the placement plan
                    for flow in valid_flows:
                        if flow[0] in route:
                            next_step = []
                            next_step_index = [index for index, x in enumerate(route) if x == flow[0]]
                            for k in next_step_index:
                                if k == len(route) - 1:
                                    continue
                                else:
                                    next_step.append(route[k + 1])
                            if flow[1] in next_step:
                                if not with_effect_ratio:
                                    delete_list.append({'iteration_num': iteration + 1, 'origin': origin,
                                                        'vehicle_num': vehicle})
                                # if consider pre-located cameras
                                elif with_effect_ratio:
                                    if random.random() <= placements['EffectRatio']:
                                        delete_list.append({'iteration_num': iteration + 1, 'origin': origin,
                                                            'vehicle_num': vehicle})

        # a vehicle from an origin in an iteration may travel through multiple valid flows
        delete_list_short = []
        [delete_list_short.append(x) for x in delete_list if x not in delete_list_short]
        ave_delete_per_inter = len(delete_list_short) / self.sim_iter_count

        # According to the indexes obtained, delete corresponding items
        for delete in delete_list_short:
            del records_by_iterations[delete['iteration_num']][delete['origin']][delete['vehicle_num']]

        # According to the indexes obtained, delete corresponding items from records_by_roads
        def _delete_judge(delete_info, record):
            i_iteration = record[0]
            i_origin = record[1]
            i_vehicle = record[2]
            captured_vehicle = [
                [i['origin'], i['vehicle_num']] for i in delete_info if i['iteration_num'] == i_iteration
            ]
            if [i_origin, i_vehicle] in captured_vehicle:
                return True
            else:
                return False

        updated_records_by_roads = {k + 1: [] for k in range(self.road_data.shape[0])}
        for road_index in range(len(records_by_roads)):
            updated_road_reco = [
                i for i in records_by_roads[road_index + 1]
                if not _delete_judge(
                    delete_list_short, [i['iteration'], i['loc'], i['sequence on loc']]
                )
            ]
            updated_records_by_roads[road_index + 1] = updated_road_reco
        return updated_records_by_roads, records_by_iterations, ave_delete_per_inter

    def _placement_weight(self, records_by_roads):
        # USE: Calculate the weight of each placement using simulation records and road cluster information

        placement_weight = []
        for i in range(len(records_by_roads)):
            record = records_by_roads[i + 1]
            neighbors = self.neighbors_by_direction[i + 1]

            # Calculate average detection number for each road pair
            # (# of vehicles detected among all iteration / total # of iteration)
            roads_v1 = neighbors['v1']
            roads_v2 = neighbors['v2']
            meet_num_v1 = [
                [i['iteration'], i['loc'], i['sequence on loc']]
                for i in record
                if (i['step1'] in roads_v1 or i['step2'] in roads_v2)
            ]  # start is on v1 or end is on v2
            meet_num_v2 = [
                [i['iteration'], i['loc'], i['sequence on loc']]
                for i in record
                if (i['step1'] in roads_v2 or i['step2'] in roads_v1)
            ]  # start is on v2 or end is on v1
            # divided by 2 as records are counted twice for each road (in and out)
            ave_meet_num_v1 = len(meet_num_v1) / (2 * self.sim_iter_count)
            ave_meet_num_v2 = len(meet_num_v2) / (2 * self.sim_iter_count)

            out1 = {'Road': i + 1, 'Inflow roads': roads_v1, 'Outflow roads': roads_v2,
                    'Number of agents met': ave_meet_num_v1}
            out2 = {'Road': i + 1, 'Inflow roads': roads_v2, 'Outflow roads': roads_v1,
                    'Number of agents met': ave_meet_num_v2}

            placement_weight.append(out1)
            placement_weight.append(out2)
        placement_weight_sorted = sorted(
            placement_weight, reverse=True, key=lambda x: x['Number of agents met'])
        return placement_weight_sorted

    def run_sim(self):
        # USE: Calculate the weight of each "placement"
        #     Each "placement" is the combination of a road segment and a direction,
        #     representing the placement of a camera
        # USE: run the simulation and record all the data for further analysis
        # OUTPUT: records_by_roads - simulation record on each roads
        #         records_by_iter - simulation record by each iteration, each iteration contains routes of each vehicle
        #         road_clusters_by_direction - road cluster information for each road

        records_by_roads = {k + 1: [] for k in range(len(self.road_data))}
        records_by_iter = {k + 1: None for k in range(self.sim_iter_count)}
        for iteration in range(self.sim_iter_count):
            routes_by_crime_loc = {k: None for k in self.crime_loc}
            for loc in self.crime_loc:
                routes_by_crime = {k + 1: None for k in range(int(self.crime[loc - 1]))}
                for c in range(int(self.crime[loc - 1])):

                    route = self._move_to_crime(loc)
                    routes_by_crime[c + 1] = route
                    for start, end in zip(route[:-1], route[1:]):
                        snapshot = {
                            'iteration': iteration + 1,
                            'loc': loc,  # the loc of the crime
                            'sequence on loc': c + 1,  # the label of crime at that loc
                            'step1': start, 'step2': end
                        }

                        records_by_roads[start].append(snapshot)
                        records_by_roads[end].append(snapshot)
                routes_by_crime_loc[loc] = routes_by_crime
            records_by_iter[iteration + 1] = routes_by_crime_loc

        self._split_inward_outward_neighbors()
        self.records_by_roads = records_by_roads
        self.records_by_iter = records_by_iter
        return

    def place_multiple_cameras(self, cam_count, pre_located_cam=None, verbose=0):
        layout = {k + 1: [] for k in range(cam_count)}
        records_by_roads = self.records_by_roads.copy()
        records_by_iterations = self.records_by_iter.copy()

        # Consider the existing cameras
        if pre_located_cam is not None:
            for k in range(len(pre_located_cam)):
                records_by_roads, records_by_iterations, ave_delete_per_inter = self._placement_elimination(
                    pre_located_cam[k], records_by_roads, records_by_iterations,
                )

        # First camera
        placements_sorted = self._placement_weight(records_by_roads)
        layout[1] = placements_sorted[0]

        # Multiple camera
        for i in list(layout.keys())[1:]:
            records_by_roads, records_by_iterations, _ = \
                self._placement_elimination(layout[i - 1], records_by_roads, records_by_iterations)
            placements_sorted = self._placement_weight(records_by_roads)
            layout[i] = placements_sorted[0]

        if verbose > 0:
            print(layout)
        return layout


import heapq
from collections import defaultdict
from typing import Dict, List

class WeightedBetweennessCentrality:
    def __init__(self, graph: nx.Graph, 
                 p_start: Dict, 
                 p_end: Dict,
                 weight: str = 'weight',
                 node_directions: Dict = None
                 ):
        """
        Initialize the weighted betweenness centrality calculator for NetworkX graphs.
        
        Parameters:
        -----------
        graph : nx.Graph or nx.DiGraph
            NetworkX graph object with weighted edges
        p_start : dict
            Probability that a path starts at each node.
            Key: node, Value: probability
        p_end : dict
            Probability that a path ends at each node.
            Key: node, Value: probability
        weight : str, optional
            Name of the edge attribute to use as weight (default: 'weight')
        """
        self.graph = graph
        self.p_start = p_start
        self.p_end = p_end
        self.weight_attr = weight
        self.nodes = set(graph.nodes())
        self.betweenness = {node: 0.0 for node in self.nodes}
        self.excluded_nodes = None
        self.node_directions = node_directions
    
    def set_excluded_nodes(self, excluded_nodes: List):
        """
        Set the list of nodes to be excluded from the betweenness calculation.
        
        Parameters:
        -----------
        excluded_nodes : list
            List of nodes to exclude
        """
        self.excluded_nodes = set(excluded_nodes)

    def _dijkstra_shortest_paths(self, source):
        """
        Run Dijkstra's algorithm from source to find shortest paths.
        
        Returns:
        --------
        dist : dict
            Shortest distance from source to each node
        sigma : dict
            Number of shortest paths from source to each node
        pred : dict
            List of predecessors for each node on shortest paths
        """
        dist = {node: float('inf') for node in self.nodes}
        sigma = {node: 0 for node in self.nodes}
        pred = {node: [] for node in self.nodes}
        
        dist[source] = 0
        sigma[source] = 1
        
        # Priority queue: (distance, node)
        pq = [(0, source)]
        visited = set()
        
        while pq:
            d, u = heapq.heappop(pq)
            
            if u in visited:
                continue
            visited.add(u)
            
            # Explore neighbors
            for v in self.graph.neighbors(u):
                # Get edge weight
                edge_data = self.graph.get_edge_data(u, v)
                edge_weight = edge_data.get(self.weight_attr, 1.0) if edge_data else 1.0
                
                # Calculate distance to v through u
                alt_dist = dist[u] + edge_weight
                
                if alt_dist < dist[v]:
                    # Found a shorter path to v
                    dist[v] = alt_dist
                    sigma[v] = sigma[u]
                    pred[v] = [u]
                    heapq.heappush(pq, (alt_dist, v))
                
                elif alt_dist == dist[v]:
                    # Found another shortest path to v
                    sigma[v] += sigma[u]
                    pred[v].append(u)
        
        return dist, sigma, pred
    
    def calculate(self) -> Dict:
        """
        Calculate weighted betweenness centrality for all nodes using a modified Brandes algorithm.
        
        Algorithm:
        ----------
        For each source node s in the graph:
        
        1. Single-Source Shortest Paths (Forward Pass):
           - Run Dijkstra's algorithm from s to find:
             * dist[w]: shortest distance from s to each node w
             * sigma[w]: number of shortest paths from s to w
             * pred[w]: list of predecessor nodes of w on shortest paths from s
        
        2. Dependency Accumulation (Backward Pass):
           - Process nodes in reverse order of distance from s (farthest first)
           - For each node w that could be a destination:
             * Calculate contribution = P_start[s] × P_end[w]
             * This weights the importance of paths from s to w
           - Propagate contributions backward along shortest paths:
             * For each predecessor u of w:
               * Accumulate delta[u] += (sigma[u] / sigma[w]) × (contribution + delta[w])
             * The ratio sigma[u]/sigma[w] represents the fraction of shortest paths
               to w that pass through predecessor u
        
        3. Update Betweenness:
           - Add delta[v] to the betweenness score of each node v (except source s)
           - delta[v] captures all the weighted flow passing through v from source s
        
        The algorithm efficiently computes the weighted sum over all (s,t) pairs by
        processing each source once and accumulating contributions in a single backward pass.
        
        Returns:
        --------
        betweenness : dict
            Weighted betweenness centrality for each node
        """
        # Reset betweenness scores
        self.betweenness = {node: 0.0 for node in self.nodes}
        
        # Step 2: Iterate over each source node
        for s in self.nodes:
            # Step 2a: Find all shortest paths from s
            dist, sigma, pred = self._dijkstra_shortest_paths(s)
            
            # Step 2b: Backward accumulation
            delta = {node: 0.0 for node in self.nodes}
            
            # Sort nodes by distance (farthest first)
            nodes_by_dist = sorted(self.nodes, key=lambda x: dist[x], reverse=True)
            
            for w in nodes_by_dist:
                if w != s and dist[w] != float('inf'):
                    # Calculate contribution for this (s, w) pair
                    contribution = self.p_start.get(s, 0) * self.p_end.get(w, 0)
                    
                    # Propagate to predecessors
                    for u in pred[w]:
                        if sigma[w] > 0:
                            delta[u] += (sigma[u] / sigma[w]) * (contribution + delta[w])
            
            # Step 2c: Update betweenness scores
            for v in self.nodes:
                if v != s:
                    self.betweenness[v] += delta[v]
        
        return self.betweenness

    def calculate_w_exclusion(self) -> Dict:
        """
        Calculate weighted betweenness centrality for all nodes with excluded nodes.
        Flows passing through any excluded node are disregarded.
        
        Algorithm:
        ----------
        For each source node s in the graph:
        
        1. Single-Source Shortest Paths (Forward Pass):
        - Run Dijkstra's algorithm from s to find:
            * dist[w]: shortest distance from s to each node w
            * sigma[w]: number of shortest paths from s to w
            * pred[w]: list of predecessor nodes of w on shortest paths from s
        
        2. Compute Valid Path Counts:
        - For each node w, compute sigma_valid[w]:
            * Number of shortest paths from s to w that do NOT pass through any excluded node
        - Process nodes in increasing order of distance from s
        - sigma_valid[w] = sum of sigma_valid[u] for all predecessors u not in excluded set
        
        3. Dependency Accumulation (Backward Pass):
        - Process nodes in reverse order of distance from s (farthest first)
        - For each node w that could be a destination (not excluded, has valid paths):
            * Calculate contribution = P_start[s] × P_end[w] × (sigma_valid[w] / sigma[w])
            * The ratio sigma_valid[w]/sigma[w] accounts for fraction of paths still valid
        - Propagate contributions backward along valid shortest paths:
            * For each predecessor u of w (not excluded):
            * Accumulate delta[u] += (sigma_valid[u] / sigma_valid[w]) × (contribution + delta[w])
            * Uses sigma_valid ratio to properly distribute flow among valid paths only
        
        4. Update Betweenness:
        - Add delta[v] to betweenness score of each node v (except source and excluded nodes)
        - Excluded nodes receive zero betweenness contribution
        
        Returns:
        --------
        betweenness : dict
            Weighted betweenness centrality for each node, excluding flows through excluded nodes
        """

        assert self.excluded_nodes is not None, "Excluded nodes set must be defined before calling this method."

        # Reset betweenness scores
        self.betweenness = {node: 0.0 for node in self.nodes}
        
        # Step 2: Iterate over each source node
        for s in self.nodes:
            # Step 2a: Find all shortest paths from s
            dist, sigma, pred = self._dijkstra_shortest_paths(s)
            
            # Step 2b: Compute valid path counts (paths not through excluded nodes)
            sigma_valid = {node: 0 for node in self.nodes}
            
            # Sort nodes by distance (closest first for forward computation)
            nodes_by_dist_asc = sorted(self.nodes, key=lambda x: dist[x])
            
            for w in nodes_by_dist_asc:
                if w in self.excluded_nodes:
                    sigma_valid[w] = 0
                elif w == s:
                    sigma_valid[w] = 1
                elif dist[w] != float('inf'):
                    # Sum sigma_valid from all predecessors not in excluded set
                    sigma_valid[w] = sum(sigma_valid[u] for u in pred[w] 
                                        if u not in self.excluded_nodes)
            
            # Step 2c: Backward accumulation
            delta = {node: 0.0 for node in self.nodes}
            
            # Sort nodes by distance (farthest first for backward accumulation)
            nodes_by_dist_desc = sorted(self.nodes, key=lambda x: dist[x], reverse=True)
            
            for w in nodes_by_dist_desc:
                if w != s and w not in self.excluded_nodes and dist[w] != float('inf'):
                    if sigma_valid[w] > 0:  # Only count if there are valid paths
                        # Calculate contribution for this (s, w) pair
                        # Scale by fraction of paths that are valid
                        contribution = (self.p_start.get(s, 0) * self.p_end.get(w, 0) * 
                                    sigma_valid[w] / sigma[w])
                        
                        # Propagate to predecessors
                        for u in pred[w]:
                            if u not in self.excluded_nodes and sigma_valid[w] > 0:
                                delta[u] += (sigma_valid[u] / sigma_valid[w]) * (contribution + delta[w])
            
            # Step 2d: Update betweenness scores
            for v in self.nodes:
                if v != s and v not in self.excluded_nodes:
                    self.betweenness[v] += delta[v]
        
        return self.betweenness

    def calculate_w_direction(self) -> Dict:
        """
        Calculate weighted betweenness centrality for all nodes with directional tracking.
        Each node's centrality is split into two directions based on node_directions.
        
        Returns:
        --------
        betweenness : dict
            Nested dict: {node: {direction: centrality_score}}
            For example: {271: {'v1': 15.3, 'v2': 8.7}}
        """
        assert self.node_directions is not None, "node_directions must be provided for directional calculation"
        
        # **CHANGED**: Initialize betweenness with directional structure
        self.betweenness = {node: {'v1': 0.0, 'v2': 0.0} for node in self.nodes}
        
        # Step 2: Iterate over each source node
        for s in self.nodes:
            # Step 2a: Find all shortest paths from s
            dist, sigma, pred = self._dijkstra_shortest_paths(s)
            
            # **CHANGED**: Initialize delta with directional structure
            delta = {node: {'v1': 0.0, 'v2': 0.0} for node in self.nodes}
            
            # Sort nodes by distance (farthest first)
            nodes_by_dist = sorted(self.nodes, key=lambda x: dist[x], reverse=True)
            
            for w in nodes_by_dist:
                if w != s and dist[w] != float('inf'):
                    # Calculate contribution for this (s, w) pair
                    contribution = self.p_start.get(s, 0) * self.p_end.get(w, 0)
                    
                    # Propagate to predecessors
                    for u in pred[w]:
                        if sigma[w] > 0:
                            # **CHANGED**: Determine directional flow
                            # Find which direction of u has w in its neighbor list (outgoing direction)
                            if w in self.node_directions[u]['v1']:
                                outgoing_direction_of_u = 'v1'
                            else:  # w must be in 'v2'
                                outgoing_direction_of_u = 'v2'
                            
                            # Find which direction of w has u in its neighbor list (incoming direction)
                            if u in self.node_directions[w]['v1']:
                                incoming_direction_of_w = 'v1'
                            else:  # u must be in 'v2'
                                incoming_direction_of_w = 'v2'
                            
                            # **CHANGED**: Accumulate to specific direction with aligned delta
                            delta[u][outgoing_direction_of_u] += (sigma[u] / sigma[w]) * (
                                contribution + delta[w][incoming_direction_of_w]
                            )
            
            # **CHANGED**: Update betweenness scores for both directions
            for v in self.nodes:
                if v != s:
                    self.betweenness[v]['v1'] += delta[v]['v1']
                    self.betweenness[v]['v2'] += delta[v]['v2']
        
        return self.betweenness

    def calculate_w_exclusion_direction(self) -> Dict:
        """
        Calculate directional weighted betweenness centrality for all nodes with excluded nodes.
        Flows passing through any excluded node are disregarded.
        Returns betweenness centrality split by direction for each node.
        
        Algorithm:
        ----------
        For each source node s in the graph:
        
        1. Single-Source Shortest Paths (Forward Pass):
        - Run Dijkstra's algorithm from s to find:
            * dist[w]: shortest distance from s to each node w
            * sigma[w]: number of shortest paths from s to w
            * pred[w]: list of predecessor nodes of w on shortest paths from s
        
        2. Compute Valid Path Counts:
        - For each node w, compute sigma_valid[w]:
            * Number of shortest paths from s to w that do NOT pass through any excluded node
        - Process nodes in increasing order of distance from s
        - sigma_valid[w] = sum of sigma_valid[u] for all predecessors u not in excluded set
        
        3. Dependency Accumulation with Directional Tracking (Backward Pass):
        - Process nodes in reverse order of distance from s (farthest first)
        - For each node w that could be a destination (not excluded, has valid paths):
            * Calculate contribution = P_start[s] × P_end[w] × (sigma_valid[w] / sigma[w])
        - Propagate contributions backward along valid shortest paths with direction tracking:
            * For each predecessor u of w (not excluded):
            * Determine outgoing_direction_of_u: which direction of u contains w in its neighbors
            * Determine incoming_direction_of_w: which direction of w contains u in its neighbors
            * Accumulate delta[u][outgoing_direction_of_u] += 
                (sigma_valid[u] / sigma_valid[w]) × (contribution + delta[w][incoming_direction_of_w])
        
        4. Update Betweenness:
        - Add delta[v][direction] to betweenness score for each node v and direction
        - Excluded nodes receive zero betweenness contribution
        
        Returns:
        --------
        betweenness_directional : dict
            Nested dict: {node: {direction_name: score}} for each node and its directions
        """

        assert self.excluded_nodes is not None, "Excluded nodes set must be defined before calling this method."
        assert self.node_directions is not None, "Node directions must be defined before calling this method."

        # CHANGED: Initialize directional betweenness scores
        self.betweenness_directional = {}
        for node in self.nodes:
            if node in self.node_directions:
                directions = list(self.node_directions[node].keys())
                self.betweenness_directional[node] = {d: 0.0 for d in directions}
        
        # Step 2: Iterate over each source node
        for s in self.nodes:
            # Step 2a: Find all shortest paths from s
            dist, sigma, pred = self._dijkstra_shortest_paths(s)
            
            # Step 2b: Compute valid path counts (paths not through excluded nodes)
            sigma_valid = {node: 0 for node in self.nodes}
            
            # Sort nodes by distance (closest first for forward computation)
            nodes_by_dist_asc = sorted(self.nodes, key=lambda x: dist[x])
            
            for w in nodes_by_dist_asc:
                if w in self.excluded_nodes:
                    sigma_valid[w] = 0
                elif w == s:
                    sigma_valid[w] = 1
                elif dist[w] != float('inf'):
                    # Sum sigma_valid from all predecessors not in excluded set
                    sigma_valid[w] = sum(sigma_valid[u] for u in pred[w] 
                                        if u not in self.excluded_nodes)
            
            # CHANGED: Step 2c: Backward accumulation with directional tracking
            delta = {}
            for node in self.nodes:
                if node in self.node_directions:
                    directions = list(self.node_directions[node].keys())
                    delta[node] = {d: 0.0 for d in directions}
            
            # Sort nodes by distance (farthest first for backward accumulation)
            nodes_by_dist_desc = sorted(self.nodes, key=lambda x: dist[x], reverse=True)
            
            for w in nodes_by_dist_desc:
                if w != s and w not in self.excluded_nodes and dist[w] != float('inf'):
                    if sigma_valid[w] > 0:  # Only count if there are valid paths
                        # Calculate contribution for this (s, w) pair
                        # Scale by fraction of paths that are valid
                        contribution = (self.p_start.get(s, 0) * self.p_end.get(w, 0) * 
                                    sigma_valid[w] / sigma[w])
                        
                        # CHANGED: Propagate to predecessors with direction tracking
                        for u in pred[w]:
                            if u not in self.excluded_nodes and sigma_valid[w] > 0:
                                # CHANGED: Determine which direction of u this flow exits from
                                # (direction of u that contains w in its neighbor list)
                                outgoing_direction_of_u = None
                                if u in self.node_directions:
                                    directions_u = list(self.node_directions[u].keys())
                                    for dir_name in directions_u:
                                        if w in self.node_directions[u][dir_name]:
                                            outgoing_direction_of_u = dir_name
                                            break
                                
                                # CHANGED: Determine which direction of w this flow enters from
                                # (direction of w that contains u in its neighbor list)
                                incoming_direction_of_w = None
                                if w in self.node_directions:
                                    directions_w = list(self.node_directions[w].keys())
                                    for dir_name in directions_w:
                                        if u in self.node_directions[w][dir_name]:
                                            incoming_direction_of_w = dir_name
                                            break
                                
                                # CHANGED: Accumulate to the specific direction using only aligned delta
                                if outgoing_direction_of_u is not None and incoming_direction_of_w is not None:
                                    delta[u][outgoing_direction_of_u] += (
                                        (sigma_valid[u] / sigma_valid[w]) * 
                                        (contribution + delta[w][incoming_direction_of_w])
                                    )
            
            # CHANGED: Step 2d: Update directional betweenness scores
            for v in self.nodes:
                if v != s and v not in self.excluded_nodes and v in self.betweenness_directional:
                    for direction in self.betweenness_directional[v]:
                        self.betweenness_directional[v][direction] += delta[v][direction]
        
        return self.betweenness_directional

    def get_betweenness(self) -> Dict:
        """
        Get the calculated betweenness centrality scores.
        
        Returns:
        --------
        betweenness : dict
            Weighted betweenness centrality for each node
        """
        return self.betweenness
    
    def get_normalized_betweenness(self) -> Dict:
        """
        Get normalized betweenness centrality scores.
        Normalizes by the sum of all probability pairs.
        
        Returns:
        --------
        normalized_betweenness : dict
            Normalized weighted betweenness centrality for each node
        """
        # Calculate total possible weight
        total_weight = sum(self.p_start.get(s, 0) * self.p_end.get(t, 0) 
                          for s in self.nodes for t in self.nodes if s != t)
        
        if total_weight == 0:
            return {node: 0.0 for node in self.nodes}
        
        return {node: score / total_weight 
                for node, score in self.betweenness.items()}


from typing import Set
import pandas as pd

class SensorPlacement:
    def __init__(self, 
                 graph: nx.Graph,
                 geo: pd.DataFrame,
                 p_start: Dict,
                 p_end: Dict,
                 weight: str = 'weight'):
        """
        Initialize the sensor placement optimizer.
        
        Parameters:
        -----------
        graph : nx.Graph or nx.DiGraph
            NetworkX graph object with weighted edges
        p_start : dict
            Probability that a path starts at each node.
            Key: node, Value: probability
        p_end : dict
            Probability that a path ends at each node.
            Key: node, Value: probability
        weight : str, optional
            Name of the edge attribute to use as weight (default: 'weight')
        """
        self.graph = graph
        self.geometries = geo
        self.p_start = p_start
        self.p_end = p_end
        self.weight = weight
        self.selected_sensors = []
        self.iteration_scores = []
        
        self.node_directions = self._split_inward_outward_neighbors()

    def _split_neighbors_by_direction(self, connected_roads, buffer=0, lost_warning=False):
        # USE: assign the connected roads to the two end points of the road for analysis
        # INPUT: dict, the connected roads for a road
        # OUTPUT: a dict
        from shapely import Point

        node_geo = self.geometries[self.geometries['id'] == list(connected_roads.keys())[0]].geometry.values[0]
        try:
            coords = node_geo.coords
        except:  # When lines are disconnected, data problem
            coords = [node_geo.geoms[0].coords[0], node_geo.geoms[-1].coords[-1]]
            if lost_warning:
                print(
                    "Remedial measures are taken to extract end points from road segments. "
                    "Please note there are problems (i.e., lines are not connected in the same group) "
                    "in the data, although they are solved for this time."
                )

        # assign the connected roads to the two end points of the road in analysis
        v1 = []
        v2 = []
        for neighbor in list(connected_roads.values())[0]:
            neighbor_geo = self.geometries[self.geometries['id'] == neighbor].geometry.values[0]
            if neighbor_geo.distance(Point(coords[0])) <= buffer:  # if the connected road is closer to the cor 0
                v1.append(neighbor)
            elif neighbor_geo.distance(
                    Point(coords[-1])) <= buffer:  # if the connected road is closer to the cor 1
                v2.append(neighbor)
            else:
                if lost_warning:
                    print('Road', road, 'is not attached to any vetice of the target road',
                          list(neighbor_roads.keys())[0], '.')
        return {'v1': v1, 'v2': v2}

    def _split_inward_outward_neighbors(self):
        node_names = list(self.graph.nodes())
        node_directions = {k: None for k in node_names}
        for n in node_names:
            neighbors = list(nx.neighbors(self.graph, n))
            node_directions[n] = self._split_neighbors_by_direction({n: neighbors})
        return node_directions

    def place_sensors(
        self, num_sensors: int, verbose: bool = True, get_iterations = False, preset_sensors: Set = None
        ) -> List:
        """
        Greedily select sensor locations to maximize coverage.
        
        Algorithm:
        ----------
        1. Initialize empty exclusion set
        2. For each iteration (up to num_sensors):
           a. Calculate weighted betweenness centrality with current exclusions
           b. Select node with highest centrality (that's not already selected)
           c. Add selected node to exclusion set
           d. Record the selection and its score
        
        Parameters:
        -----------
        num_sensors : int
            Number of sensors to place
        verbose : bool, optional
            If True, print progress information (default: True)
            
        Returns:
        --------
        selected_sensors : list
            List of selected sensor locations in order of selection
        """
        self.selected_sensors = []
        self.iteration_scores = []
        excluded_nodes = set()
        
        calculator = WeightedBetweennessCentrality(
            self.graph, 
            self.p_start, 
            self.p_end, 
            self.weight,
            self.node_directions,
        )
        
        candidate_nodes_over_iterations = []
        for iteration in range(num_sensors):
            if verbose:
                print(f"\n{'='*60}")
                print(f"Iteration {iteration + 1}/{num_sensors}")
                print(f"{'='*60}")
            
            # Set excluded nodes (previously selected sensors)
            if excluded_nodes:
                calculator.set_excluded_nodes(list(excluded_nodes))
                betweenness = calculator.calculate_w_exclusion()
            else:
                # First iteration: no exclusions
                betweenness = calculator.calculate()
            
            # Find node with highest centrality (excluding already selected)
            available_nodes = {node: score for node, score in betweenness.items() 
                             if node not in excluded_nodes}
            
            if not available_nodes:
                if verbose:
                    print("No more available nodes to select.")
                break
            
            # Select node with maximum betweenness
            best_node = max(available_nodes.items(), key=lambda x: x[1])
            candidate_nodes = dict(sorted(
                available_nodes.items(), key=lambda x: x[1], reverse=True
                )[:num_sensors])
            selected_node, selected_score = best_node
            
            # Record selection
            self.selected_sensors.append(selected_node)
            self.iteration_scores.append(selected_score)
            excluded_nodes.add(selected_node)
            candidate_nodes_over_iterations.append(candidate_nodes)
            
            if verbose:
                print(f"Selected node: {selected_node}")
                print(f"Betweenness score: {selected_score:.6f}")
                print(f"Total sensors placed: {len(self.selected_sensors)}")
        
        if verbose:
            print(f"\n{'='*60}")
            print("Sensor Placement Complete")
            print(f"{'='*60}")
            print(f"Selected sensors (in order): {self.selected_sensors}")
            print(f"Scores: {[f'{s:.6f}' for s in self.iteration_scores]}")
        
        if get_iterations:
            return self.selected_sensors, candidate_nodes_over_iterations
        return self.selected_sensors

    def place_sensors_w_directions(
        self, num_sensors: int, verbose: bool = True, get_iterations = False, preset_sensors: Set = None
        ) -> List:
        """
        Greedily select sensor locations to maximize coverage.
        
        Algorithm:
        ----------
        1. Initialize empty exclusion set
        2. For each iteration (up to num_sensors):
           a. Calculate weighted betweenness centrality with current exclusions
           b. Select node with highest centrality (that's not already selected)
           c. Add selected node to exclusion set
           d. Record the selection and its score
        
        Parameters:
        -----------
        num_sensors : int
            Number of sensors to place
        verbose : bool, optional
            If True, print progress information (default: True)
            
        Returns:
        --------
        selected_sensors : list
            List of selected sensor locations in order of selection
        """
        self.selected_sensors = []
        self.iteration_scores = []
        if preset_sensors is not None:
            excluded_nodes = preset_sensors.copy()
        else:
            excluded_nodes = set()
        
        calculator = WeightedBetweennessCentrality(
            self.graph, 
            self.p_start, 
            self.p_end, 
            self.weight,
            self.node_directions,
        )
        
        candidate_nodes_over_iterations = []
        for iteration in range(num_sensors):
            if verbose:
                print(f"\n{'='*60}")
                print(f"Iteration {iteration + 1}/{num_sensors}")
                print(f"{'='*60}")
            
            # Set excluded nodes (previously selected sensors)
            if excluded_nodes:
                calculator.set_excluded_nodes(list(excluded_nodes))
                betweenness = calculator.calculate_w_exclusion_direction()
            else:
                # First iteration: no exclusions
                betweenness = calculator.calculate_w_direction()
            
            # Find node-direction pair with highest centrality (excluding already selected nodes)
            available_node_directions = {}
            for node, directions in betweenness.items():
                if node not in excluded_nodes:
                    # Add both directions as separate candidates
                    available_node_directions[(node, 'v1')] = directions['v1']
                    available_node_directions[(node, 'v2')] = directions['v2']
            
            if not available_node_directions:
                if verbose:
                    print("No more available nodes to select.")
                break
            
            # Select node-direction pair with maximum betweenness
            best_node_direction = max(available_node_directions.items(), key=lambda x: x[1])
            candidate_node_directions = dict(sorted(
                available_node_directions.items(), key=lambda x: x[1], reverse=True
                )[:num_sensors])
            (selected_node, selected_direction), selected_score = best_node_direction
            
            # Record selection
            self.selected_sensors.append((selected_node, selected_direction))
            self.iteration_scores.append(selected_score)
            excluded_nodes.add(selected_node)
            candidate_nodes_over_iterations.append(candidate_node_directions)
            
            if verbose:
                print(f"Selected node: {selected_node}, direction: {selected_direction}")
                print(f"Betweenness score: {selected_score:.6f}")
                print(f"Total sensors placed: {len(self.selected_sensors)}")
        
        if verbose:
            print(f"\n{'='*60}")
            print("Sensor Placement Complete")
            print(f"{'='*60}")
            print(f"Selected sensors (in order): {self.selected_sensors}")
            print(f"Scores: {[f'{s:.6f}' for s in self.iteration_scores]}")
        
        if get_iterations:
            return self.selected_sensors, candidate_nodes_over_iterations
        return self.selected_sensors

    def get_selected_sensors(self) -> List:
        """
        Get the list of selected sensor locations.
        
        Returns:
        --------
        selected_sensors : list
            List of selected sensor locations in order of selection
        """
        return self.selected_sensors
    
    def get_iteration_scores(self) -> List:
        """
        Get the betweenness scores at each iteration.
        
        Returns:
        --------
        iteration_scores : list
            List of betweenness scores for each selected sensor
        """
        return self.iteration_scores
    
    def get_summary(self) -> Dict:
        """
        Get a summary of the sensor placement results.
        
        Returns:
        --------
        summary : dict
            Dictionary containing:
            - 'sensors': list of selected sensors
            - 'scores': list of scores at each iteration
            - 'num_sensors': total number of sensors placed
        """
        return {
            'sensors': self.selected_sensors,
            'scores': self.iteration_scores,
            'num_sensors': len(self.selected_sensors)
        }


