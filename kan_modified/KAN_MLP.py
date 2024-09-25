import torch
# import torch.nn as nn
import torch.nn.functional as F
import numpy as np
# from .KANLayer import KANLayer
from .Symbolic_KANLayer import Symbolic_KANLayer
from .LBFGS import *
import os
import glob
import matplotlib.pyplot as plt
from tqdm import tqdm
import random
import copy
import pandas as pd
from sympy.printing import latex
from sympy import *
import sympy
import yaml
import math
from .spline import curve2coef
from .utils import SYMBOLIC_LIB
from .hypothesis import plot_tree

# from .MultKAN import MultKAN

# https://github.com/Blealtan/efficient-kan/blob/master/src/efficient_kan/kan.py
class KANLayer_efficient(torch.nn.Module):

    def __init__(
        self,
        in_features,
        out_features,
        grid_size=5,
        spline_order=3,
        scale_noise=0.1,
        scale_base=1.0,
        scale_spline=1.0,
        enable_standalone_scale_spline=True,
        base_activation=torch.nn.SiLU,
        grid_eps=0.02,
        grid_range=[-1, 1],
    ):
        super(KANLayer_efficient, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.grid_size = grid_size
        self.spline_order = spline_order

        h = (grid_range[1] - grid_range[0]) / grid_size
        grid = (
            (
                torch.arange(-spline_order, grid_size + spline_order + 1) * h
                + grid_range[0]
            )
            .expand(in_features, -1)
            .contiguous()
        )
        self.register_buffer("grid", grid)

        self.base_weight = torch.nn.Parameter(torch.Tensor(out_features, in_features))
        self.spline_weight = torch.nn.Parameter(
            torch.Tensor(out_features, in_features, grid_size + spline_order)
        )
        if enable_standalone_scale_spline:
            self.spline_scaler = torch.nn.Parameter(
                torch.Tensor(out_features, in_features)
            )

        self.scale_noise = scale_noise
        self.scale_base = scale_base
        self.scale_spline = scale_spline
        self.enable_standalone_scale_spline = enable_standalone_scale_spline
        self.base_activation = base_activation()
        self.grid_eps = grid_eps

        self.reset_parameters()

    def reset_parameters(self):
        torch.nn.init.kaiming_uniform_(self.base_weight, a=math.sqrt(5) * self.scale_base)
        with torch.no_grad():
            noise = (
                (
                    torch.rand(self.grid_size + 1, self.in_features, self.out_features)
                    - 1 / 2
                )
                * self.scale_noise
                / self.grid_size
            )
            self.spline_weight.data.copy_(
                (self.scale_spline if not self.enable_standalone_scale_spline else 1.0)
                * self.curve2coeff(
                    self.grid.T[self.spline_order : -self.spline_order],
                    noise,
                )
            )
            if self.enable_standalone_scale_spline:
                # torch.nn.init.constant_(self.spline_scaler, self.scale_spline)
                torch.nn.init.kaiming_uniform_(self.spline_scaler, a=math.sqrt(5) * self.scale_spline)

    def b_splines(self, x: torch.Tensor):
        """
        Compute the B-spline bases for the given input tensor.

        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, in_features).

        Returns:
            torch.Tensor: B-spline bases tensor of shape (batch_size, in_features, grid_size + spline_order).
        """
        assert x.dim() == 2 and x.size(1) == self.in_features

        grid: torch.Tensor = (
            self.grid
        )  # (in_features, grid_size + 2 * spline_order + 1)
        x = x.unsqueeze(-1)
        bases = ((x >= grid[:, :-1]) & (x < grid[:, 1:])).to(x.dtype)
        for k in range(1, self.spline_order + 1):
            bases = (
                (x - grid[:, : -(k + 1)])
                / (grid[:, k:-1] - grid[:, : -(k + 1)])
                * bases[:, :, :-1]
            ) + (
                (grid[:, k + 1 :] - x)
                / (grid[:, k + 1 :] - grid[:, 1:(-k)])
                * bases[:, :, 1:]
            )

        assert bases.size() == (
            x.size(0),
            self.in_features,
            self.grid_size + self.spline_order,
        )
        return bases.contiguous()

    def curve2coeff(self, x: torch.Tensor, y: torch.Tensor):
        """
        Compute the coefficients of the curve that interpolates the given points.

        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, in_features).
            y (torch.Tensor): Output tensor of shape (batch_size, in_features, out_features).

        Returns:
            torch.Tensor: Coefficients tensor of shape (out_features, in_features, grid_size + spline_order).
        """
        assert x.dim() == 2 and x.size(1) == self.in_features
        assert y.size() == (x.size(0), self.in_features, self.out_features)

        A = self.b_splines(x).transpose(
            0, 1
        )  # (in_features, batch_size, grid_size + spline_order)
        B = y.transpose(0, 1)  # (in_features, batch_size, out_features)
        solution = torch.linalg.lstsq(
            A, B
        ).solution  # (in_features, grid_size + spline_order, out_features)
        result = solution.permute(
            2, 0, 1
        )  # (out_features, in_features, grid_size + spline_order)

        assert result.size() == (
            self.out_features,
            self.in_features,
            self.grid_size + self.spline_order,
        )
        return result.contiguous()

    @property
    def scaled_spline_weight(self):
        return self.spline_weight * (
            self.spline_scaler.unsqueeze(-1)
            if self.enable_standalone_scale_spline
            else 1.0
        )

    def forward(self, x: torch.Tensor):
        assert x.size(-1) == self.in_features
        original_shape = x.shape
        x = x.reshape(-1, self.in_features)

        base_output = F.linear(self.base_activation(x), self.base_weight)
        spline_output = F.linear(
            self.b_splines(x).view(x.size(0), -1),
            self.scaled_spline_weight.view(self.out_features, -1),
        )
        output = base_output + spline_output

        output = output.reshape(*original_shape[:-1], self.out_features)

        return output

    def get_activations(self, x: torch.Tensor, save_act=True):
        '''
        KANLayer forward given input x
        Essentially the same as foward but with the option to save feature-wise activations

        Args:
        -----
            x : 2D torch.float
                inputs, shape (batch_size, in_features)

        Returns:
        --------
            output : 2D torch.float
                outputs, shape (batch_size, out_features)
            preacts : 3D torch.float
                inputs expanded for each output feature, shape (batch_size, out_features, in_features)
            postacts : 3D torch.float
                per-feature contributions to the output, shape (batch_size, out_features, in_features)
            postspline : 3D torch.float
                per-feature spline contributions, shape (batch_size, out_features, in_features)
        '''
        assert x.size(-1) == self.in_features
        original_shape = x.shape
        x = x.reshape(-1, self.in_features)
        batch_size = x.size(0)

        # Compute base activation
        base_activation = self.base_activation(x)  # Shape: (batch_size, in_features)

        # Compute per-feature base contributions
        base_weight_expanded = self.base_weight.unsqueeze(0)  # Shape: (1, out_features, in_features)
        base_activation_expanded = base_activation.unsqueeze(1)  # Shape: (batch_size, 1, in_features)
        base_output_contributions = base_activation_expanded * base_weight_expanded  # Shape: (batch_size, out_features, in_features)
        base_output = base_output_contributions.sum(dim=2)  # Shape: (batch_size, out_features)

        # Compute B-spline activations
        b_spline_vals = self.b_splines(x)  # Shape: (batch_size, in_features, n_spline_coeffs)

        # Compute per-feature spline contributions
        scaled_spline_weight = self.scaled_spline_weight  # Shape: (out_features, in_features, n_spline_coeffs)
        spline_output_contributions = torch.einsum('bik,oik->boi', b_spline_vals, scaled_spline_weight)  # Shape: (batch_size, out_features, in_features)
        spline_output = spline_output_contributions.sum(dim=2)  # Shape: (batch_size, out_features)

        # Total per-feature contributions
        total_output_contributions = base_output_contributions + spline_output_contributions  # Shape: (batch_size, out_features, in_features)

        # Final output
        output = base_output + spline_output  # Shape: (batch_size, out_features)
        output = output.reshape(*original_shape[:-1], self.out_features)

        if save_act:
            preacts = x.unsqueeze(1).expand(batch_size, self.out_features, self.in_features)  # Shape: (batch_size, out_features, in_features)
            postacts = total_output_contributions  # Shape: (batch_size, out_features, in_features)
            postspline = spline_output_contributions  # Shape: (batch_size, out_features, in_features)
            return output, preacts, postacts, postspline

        return output

    @torch.no_grad()
    def update_grid(self, x: torch.Tensor, margin=0.01):
        assert x.dim() == 2 and x.size(1) == self.in_features
        batch = x.size(0)

        splines = self.b_splines(x)  # (batch, in, coeff)
        splines = splines.permute(1, 0, 2)  # (in, batch, coeff)
        orig_coeff = self.scaled_spline_weight  # (out, in, coeff)
        orig_coeff = orig_coeff.permute(1, 2, 0)  # (in, coeff, out)
        unreduced_spline_output = torch.bmm(splines, orig_coeff)  # (in, batch, out)
        unreduced_spline_output = unreduced_spline_output.permute(
            1, 0, 2
        )  # (batch, in, out)

        # sort each channel individually to collect data distribution
        x_sorted = torch.sort(x, dim=0)[0]
        grid_adaptive = x_sorted[
            torch.linspace(
                0, batch - 1, self.grid_size + 1, dtype=torch.int64, device=x.device
            )
        ]

        uniform_step = (x_sorted[-1] - x_sorted[0] + 2 * margin) / self.grid_size
        grid_uniform = (
            torch.arange(
                self.grid_size + 1, dtype=torch.float32, device=x.device
            ).unsqueeze(1)
            * uniform_step
            + x_sorted[0]
            - margin
        )

        grid = self.grid_eps * grid_uniform + (1 - self.grid_eps) * grid_adaptive
        grid = torch.concatenate(
            [
                grid[:1]
                - uniform_step
                * torch.arange(self.spline_order, 0, -1, device=x.device).unsqueeze(1),
                grid,
                grid[-1:]
                + uniform_step
                * torch.arange(1, self.spline_order + 1, device=x.device).unsqueeze(1),
            ],
            dim=0,
        )

        self.grid.copy_(grid.T)
        self.spline_weight.data.copy_(self.curve2coeff(x, unreduced_spline_output))

    def regularization_loss(self, regularize_activation=1.0, regularize_entropy=1.0):
        """
        Compute the regularization loss.

        This is a dumb simulation of the original L1 regularization as stated in the
        paper, since the original one requires computing absolutes and entropy from the
        expanded (batch, in_features, out_features) intermediate tensor, which is hidden
        behind the F.linear function if we want an memory efficient implementation.

        The L1 regularization is now computed as mean absolute value of the spline
        weights. The authors implementation also includes this term in addition to the
        sample-based regularization.
        """
        l1_fake = self.spline_weight.abs().mean(-1)
        regularization_loss_activation = l1_fake.sum()
        p = l1_fake / regularization_loss_activation
        regularization_loss_entropy = -torch.sum(p * p.log())
        return (
            regularize_activation * regularization_loss_activation
            + regularize_entropy * regularization_loss_entropy
        )

    def to(self, device):
        super(KANLayer_efficient, self).to(device)
        self.device = device
        return self


class KAN_Linear(torch.nn.Module):
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
            # symbolic_enabled=True,
            # affine_trainable=False,
            grid_eps=0.02,
            grid_range=[-1, 1],
            # sp_trainable=True,
            # sb_trainable=True,
            seed=1,
            save_act=True,
            # sparse_init=False,
            # auto_save=True,
            # first_init=True,
            # ckpt_path='./model',
            # state_id=0,
            # round=0,
            device='cpu'
    ):
        super(KAN_Linear, self).__init__()

        torch.manual_seed(seed)
        np.random.seed(seed)
        random.seed(seed)

        ### initializeing the numerical front ###
        self.depth = len(width) - 1

        for i in range(len(width)):
            if type(width[i]) == int:
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

        width_in = self.width_in
        width_out = self.width_out

        self.base_fun_name = base_fun
        if base_fun == 'silu':
            base_fun = torch.nn.SiLU()
        elif base_fun == 'identity':
            base_fun = torch.nn.Identity()
        elif base_fun == 'zero':
            base_fun = lambda x: x*0.

        self.grid_eps = grid_eps
        self.grid_range = grid_range

        self.layers = torch.nn.ModuleList()
        # The first layer is a Linear layer
        l = 0
        self.layers.append(
            torch.nn.Linear(
                in_features=width_in[l],
                out_features=width_out[l+1],
                bias=True,
                # device=device,
            )
        )
        # The rest of the layers are KAN layers
        for l in range(1, self.depth):
            self.layers.append(
                KANLayer_efficient(
                    in_features=width_in[l],
                    out_features=width_out[l+1],
                    grid_size=grid_size,
                    spline_order=spline_order,
                    scale_noise=scale_noise,
                    scale_base=scale_base,
                    scale_spline=scale_spline,
                    base_activation=base_fun,
                    grid_eps=grid_eps,
                    grid_range=grid_range,
                )
            )

        self.grid_size = grid_size
        self.spline_order = spline_order
        self.base_fun = base_fun
        # self.layer_types = layer_types

        self.device = device
        self.to(device)

    def to(self, device):
        self.device = device
        super(KAN_Linear, self).to(device)
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
        if self.mult_homo == True:
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

    def forward(self, x: torch.Tensor, update_grid=False):
        for layer in self.layers:
            if isinstance(layer, KANLayer_efficient) and update_grid:
                layer.update_grid(x)
            x = layer(x)
        return x

    def get_activations(self, x):
        '''
        Get the activations of all layers
        '''
        self.preacts = []
        self.postacts = []
        for layer in self.layers:
            if isinstance(layer, KANLayer_efficient):
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
        assert isinstance(self.layers[l], KANLayer_efficient), \
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

    def l1_regularization(self, layer):
        return layer.weight.abs().sum()

    def fit(
        self,
        train_loader,
        valid_loader,
        loss_fn,
        optimizer,
        num_epochs=25,
        update_grid=True,
        device='cpu',
        l1_lambda=0.0,
    ):

        self.to(device)

        losses = {'train': [], 'valid': []}
        for epoch in range(num_epochs):
            # Set model to training mode
            self.train()
            running_loss = 0.0
            total_train = 0

            # Training loop with progress bar
            train_loop = tqdm(
                train_loader,
                desc=f'Epoch [{epoch+1}/{num_epochs}] Training',
                leave=False,
                ncols=100)

            for inputs, labels in train_loop:
                inputs, labels = inputs.to(device), labels.to(device)

                # Zero the gradients
                optimizer.zero_grad()

                # Forward pass
                outputs = self.forward(inputs, update_grid=update_grid)
                loss = loss_fn(outputs, labels)

                # Regularization loss
                l1_reg = l1_lambda * self.l1_regularization(self.layers[0])
                loss += l1_reg

                # Backward pass and optimize
                loss.backward()
                optimizer.step()

                # Accumulate the loss
                batch_size = inputs.size(0)
                running_loss += loss.item() * batch_size
                total_train += batch_size

                # Update progress bar
                train_loss = running_loss / total_train
                # train_loop.set_postfix({'loss': f'{train_loss:.4f}'})
                train_loop.set_description(
                    f'Epoch [{epoch+1}/{num_epochs}] Training '
                    f'| loss: {train_loss:.4f} | reg: {l1_reg:.4f}'
                )

            # Validation phase
            self.eval()  # Set model to evaluation mode
            valid_loss = 0.0
            total_valid = 0

            with torch.no_grad():  # No need to compute gradients during validation
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
                    total_valid += batch_size

                    # Update progress bar
                    valid_loss_avg = valid_loss / total_valid
                    valid_loop.set_postfix({'val_loss': f'{valid_loss_avg:.4f}'})

            # Calculate average training and validation loss
            train_loss = running_loss / total_train
            valid_loss /= total_valid
            losses['train'].append(math.sqrt(train_loss))
            losses['valid'].append(math.sqrt(valid_loss))

            # Print statistics for the epoch
            print(f'Epoch [{epoch+1}/{num_epochs}], '
                  f'Train Loss: {train_loss:.4f}, '
                  f'Valid Loss: {valid_loss:.4f}')

        print("Training complete.")

        return losses
