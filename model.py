import numpy as np
import torch
import torch_geometric as pyG
from torch_geometric.nn import GCNConv
from torch_geometric.nn import ChebConv


class CrimeTorchDataset(torch.utils.data.Dataset):
    def __init__(self, x, y, standardize=False):
        super().__init__()
        self.x = torch.from_numpy(x).to(torch.float32)
        self.y = torch.from_numpy(y).to(torch.float32)
        self.standardize = standardize
        if self.standardize:
            self.mean_max = np.max(x[:, 0, :, 0], axis=1).mean()

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
    def __init__(self,):
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

    def build_adj_from_polygons(self, polygon, mode='Queen', knn_n=None):
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
        else:
            raise NotImplementedError
        self.readout = torch.nn.Sequential(
            torch.nn.Linear(1, 1, bias=True),
        ).to(self.device)
        return

    def reset_model(self,):
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
                pyG.data.Data(x=x, edge_index=self.edge_index, edge_attr=self.edge_weight, y=y) for x, y in zip(t, y_full[0])
            ]) for t in x_temporal
        ]
        temporal_dataloader = [
            pyG.loader.DataLoader(t, batch_size=len(t), shuffle=False) for t in temporal_dataset
        ]
        temporal_batchdata = [next(iter(loader)) for loader in temporal_dataloader]
        return temporal_batchdata  # a list of BatchData

    def train_model(self, batch_size=32, lr=0.001, epoch=1000, dir_cache='./', test_loc=''):
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
                            val_x, val_snapshot.edge_index.to(self.device), val_snapshot.edge_attr.to(self.device), val_h
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

    def pred_crime_test_set(self, dir_cache, test_loc=''):
        import os
        from torch.utils.data import DataLoader
        test_dataloader = DataLoader(
            self.test_dataset, batch_size=len(self.test_dataset), collate_fn=self._crime_seq_collate_fn
        )

        for filename in os.listdir(dir_cache):
            if filename.startswith('model_main') and filename.endswith(f"{test_loc}.pth"):
                model = torch.load(f'{dir_cache}/{filename}', weights_only=False).to(self.device)
            if filename.startswith('model_readout')  and filename.endswith(f"{test_loc}.pth"):
                readout = torch.load(f'{dir_cache}/{filename}', weights_only=False).to(self.device)

        with torch.no_grad():
            for _, test_batch in enumerate(test_dataloader):
                test_h = None
                test_y = test_batch[0].y
                for test_snapshot in test_batch:
                    test_x = test_snapshot.x.to(self.device)
                    test_h = model(
                        test_x, test_snapshot.edge_index.to(self.device), test_snapshot.edge_attr.to(self.device), test_h
                    )
                pred = readout(test_h)
        print(f"Test set predicted.")

        num_graphs = test_batch[0].batch.max().item() + 1
        pred_out = np.array([pred.cpu().numpy()[test_batch[0].batch == i] for i in range(num_graphs)])
        test_y_out = np.array([ test_y.cpu().numpy()[test_batch[0].batch == i] for i in range(num_graphs)])
        return pred_out, test_y_out


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


class CrimePredTuner(CrimePred):
    def __init__(self, model_name, storage_path="sqlite:///optuna_study.db", model_save='./'):
        import datetime
        super().__init__()
        self.storage_path = storage_path
        self.model_name = model_name
        self.study_name = f"{model_name}_{datetime.datetime.now().strftime("%m%d%H%M")}"
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
        else:
            raise NotImplementedError
        val_loss = self.train_model(batch_size=batch_size, lr=lr, dir_cache=self.model_save)
        return val_loss

    def run_study(self, n_trials=50,):
        import optuna
        study = optuna.create_study(
            direction="minimize", storage=self.storage_path, study_name=self.study_name
        )
        study.optimize(self.objective, n_trials=n_trials)
        print("Best hyperparameters:", study.best_params)
        return study.best_params


