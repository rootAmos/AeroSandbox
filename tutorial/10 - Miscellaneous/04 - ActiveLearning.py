import numpy as np
import scipy.stats
import torch
import matplotlib.pyplot as plt
from botorch import acquisition, sampling
from botorch.acquisition.objective import ScalarizedPosteriorTransform, ConstrainedMCObjective
from botorch.fit import fit_gpytorch_mll
from botorch.models import SingleTaskGP
from botorch.models.transforms import Normalize
from botorch.optim import optimize_acqf
from botorch.utils.multi_objective.box_decompositions import FastNondominatedPartitioning
from gpytorch.mlls import ExactMarginalLogLikelihood
from tqdm.notebook import trange



def example_1d_function(
    x: torch.Tensor, a: float = 15.0, b: float = 2.0
) -> torch.Tensor:
    """Ground truth function.""" 
    return torch.sin(a * x) + b * x**2

def example_noisy_1d_function(
    x: torch.Tensor, a: float = 15.0, b: float = 2.0, noise: float = 0.2
) -> torch.Tensor:
    """Noisy ground truth observations."""
    return example_1d_function(x, a, b) + noise * torch.randn_like(x)



if __name__ == "__main__":
    device = "cuda:0"
    torch.set_default_device(device)

    default_a = 15.0 
    default_b = 2.0 
    default_noise = 0.2

    x_values = torch.linspace(0, 1, 1000)
    x_values_np = x_values.detach().cpu().numpy()
    y_clean_np = example_1d_function(x_values, a=default_a, b=default_b).detach().cpu().numpy()
    y_noisy_np = example_noisy_1d_function(x_values, a=default_a, b=default_b, noise=default_noise).detach().cpu().numpy()


    
    # Gradient-based optimization using Adam to find MAXIMUM with constraints
    print("\n=== Bayesian Optimization (Finding MAXIMUM) ===")
    print("Constraints: 0 ≤ x ≤ 1")

    x_init = torch.rand(2, dtype=torch.float32)
    y_init = example_noisy_1d_function(x_init, a=default_a, b=default_b, noise=default_noise)

    print(f"Initial x points: {x_init}")
    print(f"Initial y values: {y_init}")

    gp = SingleTaskGP(x_init[:, None], y_init[:, None])
    mll = ExactMarginalLogLikelihood(gp.likelihood, gp)
    fit_gpytorch_mll(mll)

    gp_posterior = gp.posterior(x_values)
    mean = gp_posterior.mean.detach()[:, 0]
    std = gp_posterior.variance.detach()[:, 0] ** 0.5

    # Convert GP outputs to numpy arrays for plotting
    mean_np = mean.detach().cpu().numpy()
    std_np = std.detach().cpu().numpy()

    # Bayesian optimization loop
    num_iterations = 10
    all_x = [x_init]  # Store all x values
    all_y = [y_init]  # Store all y values
    
    print(f"\nStarting Bayesian Optimization Loop ({num_iterations} iterations)")
    print("=" * 50)
    
    for iteration in range(num_iterations):
        print(f"\n--- Iteration {iteration + 1} ---")
        
        # Get current GP predictions
        gp_posterior = gp.posterior(x_values)
        mean = gp_posterior.mean.detach()[:, 0]
        std = gp_posterior.variance.detach()[:, 0] ** 0.5
        
        # Convert to numpy for plotting
        mean_np = mean.detach().cpu().numpy()
        std_np = std.detach().cpu().numpy()
        
        # Find next sampling point using UCB
        ucb = acquisition.UpperConfidenceBound(gp, beta=4)
        ucb_value = ucb(x_values[:, None, None])
        
        i_max = ucb_value.argmax()
        x_candidate = x_values[i_max]
        value_candidate = ucb_value[i_max]
        
        print(f"UCB suggests sampling at: x = {x_candidate.item():.4f}")
        
        # Sample the function at the suggested point
        y_new = example_noisy_1d_function(x_candidate, a=default_a, b=default_b, noise=default_noise)
        print(f"Function value at x = {x_candidate.item():.4f}: y = {y_new.item():.4f}")
        
        # Add new point to training data
        x_new = x_candidate.unsqueeze(0)
        y_new = y_new.unsqueeze(0)
        
        # Update training data
        all_x.append(x_new)
        all_y.append(y_new)
        
        # Retrain GP with new data
        x_train = torch.cat(all_x, dim=0)
        y_train = torch.cat(all_y, dim=0)
        
        gp = SingleTaskGP(x_train[:, None], y_train[:, None])
        mll = ExactMarginalLogLikelihood(gp.likelihood, gp)
        fit_gpytorch_mll(mll)
        
        print(f"GP retrained with {len(x_train)} points")
        
        # Create updated plot
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10), height_ratios=[2, 1])
        
        # Top subplot: GP vs True Function
        ax1.plot(x_values_np, y_clean_np, 'b-', linewidth=2, label='True Function')
        ax1.plot(x_values_np, mean_np, 'r-', linewidth=1, alpha=0.7, label='GP Mean')
        ax1.fill_between(x_values_np, mean_np - 2 * std_np, mean_np + 2 * std_np, alpha=0.2, label='GP 95% CI')
        
        # Plot all training points with different colors for each iteration
        colors = ['red', 'orange', 'green', 'purple', 'brown', 'pink']
        for i, (x_iter, y_iter) in enumerate(zip(all_x, all_y)):
            x_iter_np = x_iter.detach().cpu().numpy()
            y_iter_np = y_iter.detach().cpu().numpy()
            color = colors[i % len(colors)]
            marker = 'o' if i == 0 else 's' if i == 1 else '^'
            ax1.scatter(x_iter_np, y_iter_np, color=color, s=100, zorder=5, 
                       label=f'Iteration {i}', marker=marker)
        
        # Plot the next suggested sampling point
        x_candidate_np = x_candidate.detach().cpu().numpy()
        ax1.axvline(x=x_candidate_np, color='orange', linestyle='--', linewidth=2, 
                    label=f'Next Sample: x = {x_candidate_np.item():.3f}')
        
        ax1.set_xlabel('x')
        ax1.set_ylabel('y')
        ax1.set_title(f'GP vs True Function - Iteration {iteration + 1}')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        ax1.set_xlim(0, 1)
        
        # Bottom subplot: UCB Acquisition Function
        ucb_np = ucb_value.detach().cpu().numpy().flatten()
        ax2.plot(x_values_np, ucb_np, 'g-', linewidth=2, label='UCB (β=4)')
        
        # Plot UCB values at all training points
        for i, x_iter in enumerate(all_x):
            x_iter_np = x_iter.detach().cpu().numpy()
            ucb_iter = ucb(x_iter[:, None, None]).detach().cpu().numpy().flatten()
            color = colors[i % len(colors)]
            marker = 'o' if i == 0 else 's' if i == 1 else '^'
            ax2.scatter(x_iter_np, ucb_iter, color=color, s=100, zorder=5, 
                       label=f'Iteration {i}', marker=marker)
        
        # Highlight the maximum UCB point
        ax2.scatter(x_candidate_np, value_candidate.detach().cpu().numpy(), 
                    color='orange', s=150, zorder=5, marker='*', label='UCB Maximum')
        
        ax2.set_xlabel('x')
        ax2.set_ylabel('UCB Value')
        ax2.set_title(f'UCB Acquisition Function - Iteration {iteration + 1}')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        ax2.set_xlim(0, 1)
        
        plt.tight_layout()
        plt.show()
        
        # Find current best point
        y_train_np = y_train.detach().cpu().numpy()
        best_idx = y_train_np.argmax()
        best_x = x_train[best_idx].item()
        best_y = y_train_np[best_idx]
        
        print(f"Current best: x = {best_x:.4f}, y = {best_y:.4f}")
        print(f"True function value at best x: {example_1d_function(torch.tensor([best_x]), a=default_a, b=default_b).item():.4f}")
    
    print("\n" + "=" * 50)
    print("Bayesian Optimization Complete!")
    print(f"Final best point: x = {best_x:.4f}, y = {best_y:.4f}")
    print(f"True function value: {example_1d_function(torch.tensor([best_x]), a=default_a, b=default_b).item():.4f}")

    