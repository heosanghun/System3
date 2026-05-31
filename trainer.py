import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
import copy
from router import scale_to_contractive

class EWCManager:
    """
    Manages EWC / Sparse FP-EWC parameter protection.
    Computes and stores consolidated weights and diagonal Fisher Information Matrices (FIM).
    """
    def __init__(self, model, lambda_ewc=10.0):
        self.model = model
        self.lambda_ewc = lambda_ewc
        # Maps parameter names to consolidated values and FIMs
        self.consolidated_params = {}
        self.fisher_matrices = {}
        self.expert_route_counts = {}  # Track N_i for System 3 sparse FIM normalization

    def compute_fisher(self, data_loader, is_system3=False):
        """
        Computes the diagonal Fisher Information Matrix (FIM).
        For System 3, calculates the Sparse FP-EWC FIM exclusively over routed samples
        using the Strong Law of Large Numbers (Equation 4 in paper).
        """
        self.model.eval()
        
        # Initialize Fisher matrices with zeros for all parameters
        local_fisher = {}
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                local_fisher[name] = torch.zeros_like(param.data)

        # Track number of samples routed per expert (for System 3 normalization)
        num_experts = len(self.model.experts) if is_system3 else 1
        expert_counts = torch.zeros(num_experts, device=next(self.model.parameters()).device)
        total_samples = 0

        # We compute FIM strictly over the data using sample-by-sample gradients
        # to ensure exact mathematical alignment with Eq (4)
        for x_batch, y_batch in data_loader:
            x_batch = x_batch.cuda() if torch.cuda.is_available() else x_batch
            y_batch = y_batch.cuda() if torch.cuda.is_available() else y_batch
            
            for j in range(x_batch.shape[0]):
                single_x = x_batch[j:j+1]
                single_y = y_batch[j:j+1]
                
                # Zero out gradients
                self.model.zero_grad()
                
                if is_system3:
                    logits, z_star, _, gates = self.model(single_x, training=False)
                    loss = F.cross_entropy(logits, single_y)
                    loss.backward()
                    
                    # Accumulate squared gradients
                    # For System 3, we normalize each expert's parameters by the number of times 
                    # it was routed (N_i).
                    # Gates shape is [1, num_experts]
                    routed_experts = torch.where(gates[0] > 0.0)[0]
                    expert_counts[routed_experts] += 1.0
                    
                    # Accumulate gradients for all model parameters
                    for name, param in self.model.named_parameters():
                        if param.requires_grad and param.grad is not None:
                            # If parameter belongs to a specific expert, only accumulate if that expert was routed
                            is_expert_param = False
                            for exp_idx in range(num_experts):
                                if f"experts.{exp_idx}." in name:
                                    is_expert_param = True
                                    if exp_idx in routed_experts:
                                        local_fisher[name] += param.grad.data ** 2
                                    break
                            
                            # For non-expert parameters (e.g. head, router, proj_in), accumulate always
                            if not is_expert_param:
                                local_fisher[name] += param.grad.data ** 2
                else:
                    logits, z_star = self.model(single_x)
                    loss = F.cross_entropy(logits, single_y)
                    loss.backward()
                    
                    for name, param in self.model.named_parameters():
                        if param.requires_grad and param.grad is not None:
                            local_fisher[name] += param.grad.data ** 2
                            
                total_samples += 1

        # Normalize Fisher matrices
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                if is_system3:
                    # Normalize expert parameters by N_i, and shared parameters by total_samples
                    is_expert_param = False
                    for exp_idx in range(num_experts):
                        if f"experts.{exp_idx}." in name:
                            is_expert_param = True
                            n_i = max(expert_counts[exp_idx].item(), 1.0)
                            local_fisher[name] /= n_i
                            break
                    if not is_expert_param:
                        local_fisher[name] /= total_samples
                else:
                    local_fisher[name] /= total_samples

        # Consolidate: update running Fisher and save current parameter state
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                # Accumulate Fisher over domains
                if name in self.fisher_matrices:
                    self.fisher_matrices[name] += local_fisher[name]
                else:
                    self.fisher_matrices[name] = local_fisher[name]
                
                # Update consolidated weight snapshot
                self.consolidated_params[name] = param.data.clone()

    def get_ewc_loss(self):
        """
        Computes the quadratic EWC penalty loss:
        L_EWC = 0.5 * lambda * \sum F_\theta * (\theta - \theta^*)^2
        """
        if not self.consolidated_params:
            return torch.tensor(0.0, device=next(self.model.parameters()).device)
            
        ewc_loss = 0.0
        for name, param in self.model.named_parameters():
            if param.requires_grad and name in self.fisher_matrices:
                fisher = self.fisher_matrices[name]
                consolidated = self.consolidated_params[name]
                ewc_loss += torch.sum(fisher * (param - consolidated) ** 2)
                
        return 0.5 * self.lambda_ewc * ewc_loss


def train_single_domain(model, train_x, train_y, test_x, test_y, domain_idx, ewc_manager, 
                        epochs=5, batch_size=32, lr=1e-4, is_system3=False):
    """
    Trains a model on a single domain using sequential continual learning.
    Applies the task loss, load balancing loss, EWC penalty, and contractivity projections.
    """
    model.train()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)
    
    # Standard DataLoader
    dataset = TensorDataset(train_x, train_y)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    
    # Optimizer (AdamW as per paper Section B.2)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    # Cosine learning rate decay per task
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    
    # We will record training statistics
    history = {'loss': [], 'task_loss': [], 'ewc_loss': [], 'balancing_loss': []}
    
    for epoch in range(epochs):
        epoch_losses = []
        epoch_t_losses = []
        epoch_e_losses = []
        epoch_b_losses = []
        
        for bx, by in loader:
            bx, by = bx.to(device), by.to(device)
            optimizer.zero_grad()
            
            # Forward pass
            if is_system3:
                logits, z_star, r_loss, gates = model(bx, training=True)
                task_loss = F.cross_entropy(logits, by)
                ewc_loss = ewc_manager.get_ewc_loss()
                # Total loss
                loss = task_loss + ewc_loss + r_loss
            else:
                logits, z_star = model(bx)
                task_loss = F.cross_entropy(logits, by)
                ewc_loss = ewc_manager.get_ewc_loss()
                r_loss = torch.tensor(0.0, device=device)
                loss = task_loss + ewc_loss
                
            # Backward pass
            loss.backward()
            
            # Gradient clipping to prevent explosion in fixed-points
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            
            # Optimizer step
            optimizer.step()
            
            # Projected Gradient Descent: enforce strict Banach contractivity after update!
            # Scale transition weights using spectral SVD scaling (C-FIRE margin = 0.95)
            if is_system3:
                for expert in model.experts:
                    scale_to_contractive(expert.W_z, margin=model.margin)
            else:
                if hasattr(model, 'transition'):
                    scale_to_contractive(model.transition.W_z, margin=model.margin)
                elif hasattr(model, 'W_z'):
                    scale_to_contractive(model.W_z, margin=model.margin)
            
            # Record losses
            epoch_losses.append(loss.item())
            epoch_t_losses.append(task_loss.item())
            epoch_e_losses.append(ewc_loss.item())
            epoch_b_losses.append(r_loss.item())
            
        scheduler.step()
        
    # Evaluate at the end of training on last 10% of domain data for FIM calculation
    # Paper: "The Sparse FIM is calculated dynamically over the last 10% of tokens of each respective domain."
    fim_split = max(int(0.9 * train_x.shape[0]), 1)
    fim_x, fim_y = train_x[fim_split:], train_y[fim_split:]
    fim_dataset = TensorDataset(fim_x, fim_y)
    fim_loader = DataLoader(fim_dataset, batch_size=16, shuffle=False)
    
    # Compute and consolidate Fisher information matrices
    ewc_manager.compute_fisher(fim_loader, is_system3=is_system3)
    
    return np.mean(epoch_losses)
