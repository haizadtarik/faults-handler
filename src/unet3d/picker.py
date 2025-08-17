import os
import numpy as np
import torch
import torch.optim as optim
import optuna
import docker
from .models import UNet3D, UNet3D_OSV
from .loss import BCELossWrapper, DiceLoss, BalancedBCELoss, BalancedDiceLoss, FocalLoss
import shutil


class FaultsPicker:
    def __init__(self, use_osv=False):
        self.use_osv = use_osv
        self.model = None
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.best_params = None

        if self.use_osv:
            self.client = docker.from_env(timeout=300)

        # Map string to actual loss class
        self.loss_dict = {
            "bce": BCELossWrapper(),
            "dice": DiceLoss(),
            "balanced_bce": BalancedBCELoss(),
            "balanced_dice": BalancedDiceLoss(),
            "focal": FocalLoss(),
        }

    def load_model(self, depth=4, model_path=None):
        self.model = UNet3D(depth)  # default architecture if not known
        if model_path is not None:
            self.model.load_state_dict(torch.load(model_path, map_location=self.device))

    def load_osv_model(self, depth=4, model_path=None):
        self.model = UNet3D_OSV(depth)
        if model_path is not None:
            self.model.load_state_dict(torch.load(model_path, map_location=self.device))

    def _compute_mean_iou(self, model, loader):
        """Compute mean IoU over the given loader using threshold=0.5."""
        model.to(self.device)
        model.eval()
        iou_sum = 0.0
        eps = 1e-6
        with torch.no_grad():
            for data, target in loader:
                data, target = data.to(self.device), target.to(self.device)
                output = model(data)
                if self.use_osv:
                    output = torch.sigmoid(output)
                pred = (output > 0.5).float()
                intersection = (pred * target).sum()
                union = pred.sum() + target.sum() - intersection
                iou_sum += (intersection / (union + eps)).item()
        return iou_sum / len(loader)

    def _compute_f1_score(self, model, loader):
        """
        Compute mean F1 score (equivalent to Dice for binary segmentation)
        over the given loader using threshold=0.5.
        """
        model.to(self.device)
        model.eval()
        f1_sum = 0.0
        eps = 1e-6  # to avoid division by zero
        with torch.no_grad():
            for data, target in loader:
                data, target = data.to(self.device), target.to(self.device)

                output = model(data)
                if self.use_osv:
                    output = torch.sigmoid(output)
                pred = (output > 0.5).float()

                intersection = (pred * target).sum()
                # Compute precision and recall
                precision = intersection / (pred.sum() + eps)
                recall = intersection / (target.sum() + eps)
                # Compute F1 = 2 * (precision * recall) / (precision + recall)
                f1 = 2 * precision * recall / (precision + recall + eps)

                f1_sum += f1.item()

        return f1_sum / len(loader)

    def optimize_hyperparams(self, train_loader, val_loader, n_trials=10):
        """
        Runs Optuna hyperparameter optimization for:
          - number of layers in [4..7]
          - learning rate in [1e-5..1e-2]
          - loss function in [bce, dice, balanced_bce, balanced_dice, focal]
        Maximizes mean IoU on the validation set.
        """

        def objective(trial):
            # Suggest hyperparameters
            depth = trial.suggest_int("num_layers", 4, 7)
            lr = trial.suggest_float("lr", 1e-7, 1e-3, log=True)
            loss_name = trial.suggest_categorical(
                "loss_func", ["bce", "dice", "balanced_bce", "balanced_dice", "focal"]
            )

            # Create model with chosen depth
            if self.use_osv:
                model = UNet3D_OSV(depth).to(self.device)
            else:
                model = UNet3D(depth).to(self.device)
            # Pick the loss
            loss_fn = self.loss_dict[loss_name]
            # Optimizer
            optimizer = optim.Adam(model.parameters(), lr=lr)

            num_epochs = 30  # Shorter training for HPO
            best_val_iou = 0.0

            for epoch in range(num_epochs):
                # ---- Training ----
                model.train()
                for data, target in train_loader:
                    data, target = data.to(self.device), target.to(self.device)
                    optimizer.zero_grad()
                    output = model(data)
                    if self.use_osv:  # or loss_name in ['dice', 'balanced_dice']:
                        probabilities = torch.sigmoid(output)
                        loss = loss_fn(probabilities, target)
                    else:
                        loss = loss_fn(output, target)
                    loss.backward()
                    optimizer.step()

                # ---- Validation IoU ----
                val_iou = self._compute_mean_iou(model, val_loader)
                if val_iou > best_val_iou:
                    best_val_iou = val_iou

                # Report best IoU so far to Optuna
                trial.report(best_val_iou, epoch)
                if trial.should_prune():
                    raise optuna.TrialPruned()

            return best_val_iou

        # Create the study to maximize IoU
        study = optuna.create_study(direction="maximize")
        study.optimize(objective, n_trials=n_trials)

        # Store best params for later use
        self.best_params = study.best_trial.params
        print("Hyperparameter Optimization Complete.")
        print("Best params:", self.best_params)
        print("Best mean IoU:", study.best_value)

    def train(
        self,
        train_loader,
        valid_loader,
        num_epochs=20,
        hpo=False,
        n_trials=10,
        num_layers=4,
        lr=1e-4,
        loss_name="bce",
        save_path="model/best_model.pth",
        return_model=False,
    ):
        """
        If hpo=True, first run hyperparam optimization to maximize mean IoU,
        then re-train using the best found hyperparameters (for num_epochs).
        Otherwise, train with the given (or default) settings.
        """

        if hpo:
            print("Running hyperparameter optimization (maximize IoU)...")
            self.optimize_hyperparams(train_loader, valid_loader, n_trials=n_trials)
            # Build final model using best hyperparams
            depth = self.best_params["num_layers"]
            best_lr = self.best_params["lr"]
            loss_name = self.best_params["loss_func"]

            if self.use_osv:
                self.model = UNet3D_OSV(depth).to(self.device)
            else:
                self.model = UNet3D(depth).to(self.device)
            criterion = self.loss_dict[loss_name]
            optimizer = optim.Adam(self.model.parameters(), lr=best_lr)
        else:
            if self.use_osv:
                self.model = UNet3D_OSV(num_layers).to(self.device)
            else:
                self.model = UNet3D(num_layers).to(self.device)
            criterion = self.loss_dict[loss_name]
            optimizer = optim.Adam(self.model.parameters(), lr=lr)

        print(f"Training on {self.device}")
        best_valid_iou = 0.0
        for epoch in range(num_epochs):
            # ---- TRAIN ----
            self.model.train()
            train_loss = 0.0
            for batch_idx, (data, target) in enumerate(train_loader):
                data, target = data.to(self.device), target.to(self.device)
                optimizer.zero_grad()
                output = self.model(data)
                # if self.use_osv:
                #     output = torch.sigmoid(output)
                # loss = criterion(output, target)
                # Apply sigmoid to get probabilities if using OSV or dice-based losses
                if self.use_osv:  # or loss_name in ['dice', 'balanced_dice']:
                    probabilities = torch.sigmoid(output)
                    loss = criterion(probabilities, target)
                else:
                    loss = criterion(output, target)
                loss.backward()
                optimizer.step()
                train_loss += loss.item()
            train_loss /= len(train_loader)

            # ---- VALIDATE (compute IoU + loss) ----
            self.model.eval()
            valid_loss = 0.0
            with torch.no_grad():
                for data, target in valid_loader:
                    data, target = data.to(self.device), target.to(self.device)
                    output = self.model(data)
                    if self.use_osv:
                        output = torch.sigmoid(output)
                    valid_loss += criterion(output, target).item()
            valid_loss /= len(valid_loader)

            valid_iou = self._compute_mean_iou(self.model, valid_loader)

            print(
                f"Epoch {epoch+1}/{num_epochs} "
                f"- Train Loss: {train_loss:.4f}, "
                f"Valid Loss: {valid_loss:.4f}, "
                f"Valid IoU: {valid_iou:.4f}"
            )

            # Save best model (based on IoU)
            if valid_iou > best_valid_iou:
                best_valid_iou = valid_iou
                if not os.path.exists(os.path.dirname(save_path)):
                    os.makedirs(os.path.dirname(save_path))
                torch.save(self.model.state_dict(), save_path)

        if return_model:
            return self.model

    def predict(self, seismic_data):
        if self.model is None:
            raise Exception("Model not loaded. Use load_model or train a model first.")
        seismic_data = torch.FloatTensor(seismic_data).to(self.device)
        self.model.to(self.device)
        self.model.eval()
        with torch.no_grad():
            prediction = self.model(seismic_data).cpu().numpy()[0, 0]
        return prediction

    def predict_osv(self, seismic_data, root_dir, output_dir="data/prediction/"):
        if self.model is None:
            raise Exception("Model not loaded. Use load_model or train a model first.")
        # Create output directory if it doesn't exist and remove if it exist before creating
        output_path = os.path.join(root_dir, output_dir)
        if os.path.exists(output_path):
            shutil.rmtree(output_path)
        os.makedirs(output_path, exist_ok=True)
        seismic_data = torch.FloatTensor(seismic_data).to(self.device)
        self.model.to(self.device)
        self.model.eval()
        with torch.no_grad():
            prediction = self.model(seismic_data).cpu().numpy()[0, 0]
        prediction.tofile(root_dir + "/" + output_dir + "ep.dat")
        seismic_data.cpu().numpy()[0, 0].tofile(root_dir + "/" + output_dir + "xs.dat")
        print(f"Prediction Min = {prediction.min()}, Prediction Max = {prediction.max()}")
        print("Running OSV...")
        self.client.containers.run(
            "osv:latest", volumes={root_dir + "/" + output_dir: {"bind": "/app/data", "mode": "rw"}}
        )
        print("Run Completed")
        osv = np.fromfile(root_dir + "/" + output_dir + "fv.dat", dtype=">f4").reshape(128, 128, 128)
        osv_thin = np.fromfile(root_dir + "/" + output_dir + "fvt.dat", dtype=">f4").reshape(128, 128, 128)
        prediction = 1 / (1 + np.exp(-prediction))
        return prediction, osv, osv_thin

    def evaluate(self, test_loader):
        """Compute mean IoU and mean F1 on the test set."""
        if self.model is None:
            raise Exception("Model not loaded. Use load_model or train a model first.")

        iou = self._compute_mean_iou(self.model, test_loader)
        f1 = self._compute_f1_score(self.model, test_loader)
        return iou, f1
