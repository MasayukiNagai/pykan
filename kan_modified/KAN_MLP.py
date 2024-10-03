import torch
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
import random

from .LBFGS import *
from .KANLayer2 import KANLayer_efficient, KANLayer_original
# from .Symbolic_KANLayer import Symbolic_KANLayer
# from .MultKAN import MultKAN


KANLayer = [KANLayer_original, KANLayer_efficient][0]


class BaseKAN(torch.nn.Module):
    '''
    KAN with the first layer being a nn.Linear
    '''
    def __init__(
            self,
            width,
            grid_size=3,
            spline_order=3,
            mult_arity=2,
            scale_noise=0.1,
            scale_base=0.0,
            scale_spline=1.0,
            base_fun='silu',
            grid_eps=0.02,
            grid_range=[-1, 1],
            seed=1,
            # save_act=True,
            device='cpu'
    ):
        super().__init__()

        torch.manual_seed(seed)
        np.random.seed(seed)
        random.seed(seed)

        ### initializeing the numerical front ###
        self.depth = len(width) - 1

        for i in range(len(width)):
            if isinstance(width[i], int):
                width[i] = [width[i],0]
        self.width = width

        # if mult_arity is just a scalar, we extend it to a list of lists
        # e.g, mult_arity = [[2,3],[4]] means that in the first hidden layer, 2 mult ops have arity 2 and 3, respectively;
        # in the second hidden layer, 1 mult op has arity 4.
        if isinstance(mult_arity, int):
            self.mult_homo = True # when homo is True, parallelization is possible
        else:
            self.mult_homo = False # when home if False, for loop is required.
        self.mult_arity = mult_arity

        self.base_fun_name = base_fun
        if base_fun == 'silu':
            base_fun = torch.nn.SiLU()
        elif base_fun == 'identity':
            base_fun = torch.nn.Identity()
        elif base_fun == 'zero':
            base_fun = lambda x: x*0.

        self.layers = torch.nn.ModuleList()

        self.grid_size = grid_size
        self.spline_order = spline_order
        self.scale_noise=scale_noise
        self.scale_base=scale_base
        self.scale_spline=scale_spline
        self.base_fun = base_fun
        self.grid_eps = grid_eps
        self.grid_range = grid_range

        self.device = device
        self.to(device)

    def _create_layers(self):
        raise NotImplementedError('Layers must be defined in the subclasses.')

    def to(self, device):
        self.device = device
        super().to(device)
        for layer in self.layers:
            layer.to(device)
        return self

    @property
    def width_in(self):
        '''
        The number of input nodes for each layer
        '''
        width = self.width
        width_in = [width[l][0]+width[l][1] for l in range(len(width))]
        return width_in

    @property
    def width_out(self):
        '''
        The number of output subnodes for each layer
        '''
        width = self.width
        if self.mult_homo is True:
            width_out = [width[l][0]+self.mult_arity*width[l][1] for l in range(len(width))]
        else:
            width_out = [width[l][0]+int(np.sum(self.mult_arity[l])) for l in range(len(width))]
        return width_out

    @property
    def n_sum(self):
        '''
        The number of addition nodes for each layer
        '''
        width = self.width
        n_sum = [width[l][0] for l in range(1,len(width)-1)]
        return n_sum

    @property
    def n_mult(self):
        '''
        The number of multiplication nodes for each layer
        '''
        width = self.width
        n_mult = [width[l][1] for l in range(1,len(width)-1)]
        return n_mult

    def update_grid(self, x):
        '''
        call update_grid_from_samples. This seems unnecessary but we retain it for the sake of classes that might inherit from MultKAN
        '''
        self.update_grid_from_samples(x)

    def forward(self, x: torch.Tensor, update_grid=False, save_act=False):
        for layer in self.layers:
            # Update grid for KAN layers
            if isinstance(layer, KANLayer) and update_grid:
                layer.update_grid(x)

            # Forward pass
            x = layer(x)

            # Save activation values
            if save_act:
                self.activations.append(x)

        return x

    def get_activations(self, x):
        '''
        Get the activations of all layers
        '''
        self.preacts = []
        self.postacts = []
        for layer in self.layers:
            if isinstance(layer, KANLayer):
                x, preacts, postacts, _ = layer.get_activations(x)
                self.preacts.append(preacts)
                self.postacts.append(postacts)
            else:
                x = layer(x)
                self.preacts.append(None)
                self.postacts.append(None)

        return self.preacts, self.postacts

    def get_function(self, l, i, j):
        '''
        Visualize the learned function at (l,i,j) based on pre- and post-activation values of input x
        Must run get_activations(x) first to save activation values
        '''
        # preacts, postacts = self.get_activations(x)
        assert self.preacts is not None and self.postacts is not None,\
            'Save activation values first by running "get_activations(x)"'

        assert self.preacts[l] is not None, f'Activation values do not exist for layer {l}.'

        inputs = self.preacts[l][:,j,i].cpu().detach().numpy()
        outputs = self.postacts[l][:,j,i].cpu().detach().numpy()
        # they are not ordered yet
        rank = np.argsort(inputs)
        inputs = inputs[rank]
        outputs = outputs[rank]
        plt.figure(figsize=(3,3))
        plt.plot(inputs, outputs, marker="o")
        return inputs, outputs

    def get_all_functions(self, l):
        '''
        Plot all the activation functions for layer l
        '''
        assert isinstance(self.layers[l], KANLayer), \
            f'layer {l} is not a KAN layer.'

        ncols = self.layers[l].in_features
        nrows = self.layers[l].out_features

        fig, axes = plt.subplots(nrows=nrows, ncols=ncols, figsize=(ncols*4, nrows*4), squeeze=False)

        # Loop through each subplot and call the get_function for each (i, j)
        for i in range(ncols):
            for j in range(nrows):
                ax = axes[j, i]
                inputs, outputs = self.get_function(l, i, j)
                ax.plot(inputs, outputs, marker="o")
                ax.set_title(f'in_node {i}, out_node {j}')
                ax.set_xlabel('Pre-activation')
                if i == 0:
                    ax.set_ylabel('Post-activation')
                ax.grid(True)

        fig.subplots_adjust(top=0.85)
        fig.suptitle(f'Activation values for layer {l}', fontsize=12)
        # plt.tight_layout()

        return fig

    # def regularization_loss(self, regularize_activation=1.0, regularize_entropy=1.0):
    #     return sum(
    #         layer.regularization_loss(regularize_activation, regularize_entropy)
    #         for layer in self.layers
    #     )

    def regularization_loss(self, **kwargs):
        # Default implementation (no-op) — subclasses can override this
        return torch.tensor(0.0, requires_grad=True)

    def fit(
        self,
        train_loader,
        valid_loader,
        loss_fn,
        optimizer,
        num_epochs=10,
        update_grid=True,
        save_act=True,
        device='cpu',
        **kwargs
    ):

        self.to(device)

        losses = {'train': [], 'valid': []}
        for epoch in range(num_epochs):
            # Set model to training mode
            self.train()
            running_loss = 0.0
            ct_train = 0

            ### Define the frequency of updating the grid
            n_freq = 5
            update_grid_epoch = update_grid and (epoch + 1) % n_freq == 0
            if update_grid_epoch:
                print(f'epoch {epoch+1}: update grid')

            # Training loop with progress bar
            train_loop = tqdm(
                train_loader,
                desc=f'Epoch [{epoch+1}/{num_epochs}] Training',
                leave=False,
                ncols=100)

            for i, (inputs, labels) in enumerate(train_loop):
                inputs, labels = inputs.to(device), labels.to(device)

                # Zero the gradients
                optimizer.zero_grad()

                # Forward pass
                self.activations = []
                update_grid_now = update_grid_epoch and i == 0

                outputs = self.forward(
                    inputs,
                    update_grid=update_grid_now,
                    save_act=save_act)
                loss = loss_fn(outputs, labels)

                # Custom regularization loss
                reg_loss = self.regularization_loss(**kwargs)
                loss += reg_loss

                # Backward pass and optimize
                loss.backward()
                optimizer.step()

                # Accumulate the loss
                batch_size = inputs.size(0)
                running_loss += loss.item() * batch_size
                ct_train += batch_size

                # Update progress bar
                train_loss = running_loss / ct_train
                # train_loop.set_postfix({'loss': f'{train_loss:.4f}'})
                train_loop.set_description(
                    f'Epoch [{epoch+1}/{num_epochs}] Training '
                    f'| loss: {train_loss:.4f} | reg: {reg_loss:.4f}'
                )

            # Validation phase
            self.eval()
            valid_loss = 0.0
            ct_valid = 0

            with torch.no_grad():
                valid_loop = tqdm(
                    valid_loader,
                    desc=f'Epoch [{epoch+1}/{num_epochs}] Validation',
                    leave=False,
                    ncols=100)

                for inputs, labels in valid_loop:
                    inputs, labels = inputs.to(device), labels.to(device)

                    # Forward pass
                    outputs = self.forward(inputs)
                    loss = loss_fn(outputs, labels)

                    # Accumulate validation loss
                    batch_size = inputs.size(0)
                    valid_loss += loss.item() * batch_size
                    ct_valid += batch_size

                    # Update progress bar
                    valid_loss_avg = valid_loss / ct_valid
                    valid_loop.set_postfix({'val_loss': f'{valid_loss_avg:.4f}'})

            # Calculate average training and validation loss
            train_loss = running_loss / ct_train
            valid_loss /= ct_valid
            losses['train'].append(train_loss**(1/2))
            losses['valid'].append(valid_loss**(1/2))

            # Print statistics for the epoch
            print(f'Epoch [{epoch+1}/{num_epochs}], '
                  f'Train Loss: {train_loss:.4f}, '
                  f'Valid Loss: {valid_loss:.4f}')

        print("Training complete.")

        return losses


class LinearKAN(BaseKAN):
    '''
    KAN with the first layer being a nn.Linear
    '''
    def __init__(
        self,
        width,
        grid_size=3,
        spline_order=3,
        mult_arity=2,
        scale_noise=0.1,
        scale_base=0.0,
        scale_spline=1.0,
        base_fun='silu',
        grid_eps=0.02,
        grid_range=[-1, 1],
        seed=1,
        device='cpu'
    ):
        super().__init__(
            width=width,
            grid_size=grid_size,
            spline_order=spline_order,
            mult_arity=mult_arity,
            scale_noise=scale_noise,
            scale_base=scale_base,
            scale_spline=scale_spline,
            base_fun=base_fun,
            grid_eps=grid_eps,
            grid_range=grid_range,
            seed=seed,
            device=device
        )
        self._create_layers()

    def _create_layers(self):
        # The first layer is a linear layer
        l = 0
        self.layers.append(
            torch.nn.Linear(
                in_features=self.width_in[l],
                out_features=self.width_out[l+1],
                bias=True
            )
        )
        # The rest of the layers are KAN layers
        for l in range(1, self.depth):
            self.layers.append(
                KANLayer(
                    in_features=self.width_in[l],
                    out_features=self.width_out[l+1],
                    grid_size=self.grid_size,
                    spline_order=self.spline_order,
                    scale_noise=self.scale_noise,
                    scale_base=self.scale_base,
                    scale_spline=self.scale_spline,
                    base_activation=self.base_fun,
                    grid_eps=self.grid_eps,
                    grid_range=self.grid_range,
                )
            )

    def fit(
        self,
        train_loader,
        valid_loader,
        loss_fn,
        optimizer,
        num_epochs=10,
        update_grid=True,
        # save_act=True,
        device='cpu',
        l1_lambda=0.0,
        group_lambda=0.0,
        ortho_lambda=0.0,
        kan_lambda=0.0,
        act_lambda=0.0
    ):
        # Save activation values if activation regularization is enabled
        save_act = act_lambda > 0

        losses = super().fit(
            train_loader,
            valid_loader,
            loss_fn=loss_fn,
            optimizer=optimizer,
            num_epochs=num_epochs,
            update_grid=update_grid,
            save_act=save_act,
            device=device,
            l1_lambda=l1_lambda,
            group_lambda=group_lambda,
            ortho_lambda=ortho_lambda,
            kan_lambda=kan_lambda,
            act_lambda=act_lambda,
        )
        return losses

    def regularization_loss(
        self,
        l1_lambda=0.0,
        group_lambda=0.0,
        ortho_lambda=0.0,
        kan_lambda=0.0,
        act_lambda=0.0,
    ):
        # L1 regularization for all linear layers
        l1_reg = 0.0
        if l1_lambda > 0:
            for layer in self.layers:
                if isinstance(layer, torch.nn.Linear):
                    l1_reg += l1_lambda * torch.norm(layer.weight, p=1)
            l1_reg *= l1_lambda

        # Group lasso regularzation
        group_reg = 0.0
        if group_lambda > 0:
            # for name, param in self.named_parameters():
            #     if 'weight' in name:
            #         group_reg += group_lambda * torch.norm(param, p=2)
            for weights in self.layers[0].weight:
                group_reg += torch.norm(weights, p=2)
            group_reg *= group_lambda

        # Orthogonality regularization for the first linear layer
        ortho_reg = 0.0
        if ortho_lambda > 0:
            layer = self.layers[0]
            assert isinstance(layer, torch.nn.Linear)
            weights = layer.weight

            # A) Strict, sensitive to the magnitude
            # wt_w = torch.mm(weights, weights.t())
            # identity = torch.eye(wt_w.size(0)).to(weights.device)
            # ortho_reg = torch.norm(wt_w - identity, p='fro')  # Frobenius

            # B) Focuses purely on the direction
            w_normalized = F.normalize(weights, p=2, dim=1)
            wt_w = torch.mm(w_normalized, w_normalized.t())
            mask = torch.eye(wt_w.size(0), device=weights.device)
            ortho_reg = torch.sum(torch.abs(wt_w * (1 - mask)))

            ortho_reg *= ortho_lambda

        # KAN l1 regularization
        kan_reg = 0.0
        if kan_lambda > 0:
            for layer in self.layers:
                if isinstance(layer, KANLayer):
                    # lambda_act = 1.0, lambda_entropy = 1.0 by default
                    kan_reg += layer.regularization_loss()
            kan_reg *= kan_lambda

        # Activation regularization
        act_reg = 0.0
        if act_lambda > 0:
            assert len(self.activations) > 0, 'No activation values saved.'
            layer_idx = 1 ### FIX: don't hard code
            activations = self.activations[layer_idx]
            num_features, batch_size = activations.shape

            # 1a) l1 regularization on activations
            # act_reg = torch.norm(activations, p=1) / batch_size

            # 1b) l1 regularization on feature activations
            feature_acts = torch.mean(activations, dim=0)
            act_reg = torch.norm(feature_acts, p=1)  # / num_features

            # 2) Diversity regularization
            # cosine_sim = torch.matmul(activations.T, activations)
            # cosine_sim.fill_diagonal_(0)  # Ignore self-similarity
            # act_reg +=  torch.norm(cosine_sim, 1)

            act_reg *= act_lambda

        return l1_reg + group_reg + ortho_reg + kan_reg + act_reg


class ConvKAN(BaseKAN):
    '''
    KAN with the first layer being a nn.Conv2d
    '''
    def __init__(
        self,
        width,
        num_kernels,
        kernel_size,
        pool_size,
        grid_size=3,
        spline_order=3,
        mult_arity=2,
        scale_noise=0.1,
        scale_base=0.0,
        scale_spline=1.0,
        base_fun='silu',
        grid_eps=0.02,
        grid_range=[-1, 1],
        seed=1,
        # save_act=True,
        device='cpu'
    ):
        super().__init__(
            width=width,
            grid_size=grid_size,
            spline_order=spline_order,
            mult_arity=mult_arity,
            scale_noise=scale_noise,
            scale_base=scale_base,
            scale_spline=scale_spline,
            base_fun=base_fun,
            grid_eps=grid_eps,
            grid_range=grid_range,
            seed=seed,
            device=device
        )
        self.num_kernels = num_kernels
        self.kernel_size = kernel_size
        self.pool_size = pool_size
        self._create_layers()

    def _create_layers(self):
        # The first layer is a linear layer
        l = 0
        conv_layers = [
            torch.nn.Conv1d(
                in_channels=4,
                out_channels=self.num_kernels,
                kernel_size=self.kernel_size,
                stride=1,
                padding=0,
                bias=True,
                padding_mode='zeros'
            ),
            torch.nn.MaxPool1d(
                kernel_size=self.pool_size
            ),
            torch.nn.ELU(alpha=1.0),
            torch.nn.Flatten(),
            torch.nn.LazyLinear(
                out_features=self.width_out[l],
                bias=True
            )
        ]
        self.layers.extend(torch.nn.ModuleList(conv_layers))

        # The rest of the layers are KAN layers
        for l in range(0, self.depth):
            self.layers.append(
                KANLayer(
                    in_features=self.width_in[l],
                    out_features=self.width_out[l+1],
                    grid_size=self.grid_size,
                    spline_order=self.spline_order,
                    scale_noise=self.scale_noise,
                    scale_base=self.scale_base,
                    scale_spline=self.scale_spline,
                    base_activation=self.base_fun,
                    grid_eps=self.grid_eps,
                    grid_range=self.grid_range,
                )
            )

    def fit(
        self,
        train_loader,
        valid_loader,
        loss_fn,
        optimizer,
        num_epochs=10,
        update_grid=True,
        device='cpu',
    ):
        losses = super().fit(
            train_loader,
            valid_loader,
            loss_fn,
            optimizer,
            num_epochs,
            update_grid,
            device
        )
        return losses

    def forward(self, x: torch.Tensor, update_grid=False):
        for i, layer in enumerate(self.layers):
            if isinstance(layer, KANLayer) and update_grid:
                layer.update_grid(x)
            if i == 0: # Convolutional layer
                x = x.permute(0, 2, 1)
            x = layer(x)
        return x
