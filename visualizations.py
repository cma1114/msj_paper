import numpy as np
import matplotlib.pyplot as plt
import scipy.stats as stats

def plot_parity_accuracy(correct_by_shots, suffix, num_parities):
   shots = sorted(correct_by_shots.keys())
   accuracies = [np.mean(correct_by_shots[s])*100 for s in shots]

   plt.figure(figsize=(10, 6))
   plt.plot(shots, accuracies, '-o')
   z = np.polyfit(shots, accuracies, 1)  
   p = np.poly1d(z)
   plt.plot(shots, p(shots), "r--", alpha=0.8, label=f'Fit line: y={z[0]:.3f}x + {z[1]:.3f}')
   plt.text(0.02, 0.98, f'slope={z[0]:.3f}', transform=plt.gca().transAxes, verticalalignment='top')
   plt.xlabel('Number of Shots')
   plt.ylabel('Accuracy')
   plt.title(f'{suffix} Accuracy vs Number of Shots ({num_parities} parities)')
   plt.grid(True)
   plt.savefig(f"parity_accuracy{suffix}.png", dpi=100)
   plt.show()

import scipy.stats as stats
def plot_parity_accuracy_mult(dict_list, suffix, num_parities, labels=None):
    """
    Plots accuracy vs number of shots for multiple data sets (each a dictionary) with 95% confidence intervals
    and linear fit lines using specified colors. Optionally, custom labels can be provided for each data set.

    Parameters:
    - dict_list: list of dictionaries; each dictionary maps shot numbers (keys) to lists of accuracy values.
    - suffix: a string suffix for the plot title and filename.
    - num_parities: number of parities to be included in the title.
    - labels: (optional) list of strings representing the label for each data set.
    """
    plt.figure(figsize=(10, 6))
    colors = ['green', 'black', 'gray', 'orange']

    for idx, correct_by_shots in enumerate(dict_list):
        shots = sorted(correct_by_shots.keys())
        accuracies = []
        ci_margins = []  # Half-width of the 95% CI for each shot

        for s in shots:
            data = correct_by_shots[s]
            mean_acc = np.mean(data) * 100
            accuracies.append(mean_acc)
            
            # Compute 95% confidence interval margin
            if len(data) > 1:
                sem = stats.sem(data) * 100  # Standard error of the mean in percentage
                margin = sem * stats.t.ppf(0.975, len(data) - 1)
            else:
                margin = 0  # If only one data point, CI cannot be computed
            ci_margins.append(margin)
        
        color = colors[idx % len(colors)]
        
        # Use custom label if provided, else default to "Data Set {idx+1}"
        label = labels[idx] if labels and idx < len(labels) else f'Data Set {idx + 1}'
        
        plt.errorbar(shots, accuracies, yerr=ci_margins, fmt='-o', color=color)###, label=label)
        
        z = np.polyfit(shots, accuracies, 1)
        p = np.poly1d(z)
        plt.plot(shots, p(shots), "--", color=color, alpha=0.8, 
                 label=f'{label}: slope={z[0]:.2f}, intercept={z[1]:.1f}')
    
    plt.xlabel('Number of Shots')
    plt.ylabel('Accuracy (%)')
    plt.title(f'{suffix} Accuracy vs Number of Shots ({num_parities} parities)')
    plt.grid(True)
    plt.legend()
    plt.savefig(f"parity_accuracy{suffix}.png", dpi=100)
    plt.show()

def plot_parity_accuracy_mult2(dict_list, suffix, num_parities, labels=None):
    """
    Plots accuracy vs number of shots for multiple data sets (each a dictionary) with 95% confidence
    bands for the regression line (error bounds around the slope) using specified colors.
    
    Parameters:
    - dict_list: list of dictionaries; each maps shot numbers (keys) to lists of accuracy values.
    - suffix: string used for the plot title and filename.
    - num_parities: number of parities (used in the title).
    - labels: (optional) list of custom labels for each dataset.
    """
    plt.figure(figsize=(10, 6))
    colors = ['green', 'black']
    
    for idx, correct_by_shots in enumerate(dict_list):
        # Sort the shots and compute mean accuracies (in percentage)
        shots = sorted(correct_by_shots.keys())
        accuracies = [np.mean(correct_by_shots[s]) * 100 for s in shots]
        
        # Fit a linear regression: y = slope * x + intercept
        z = np.polyfit(shots, accuracies, 1)
        slope, intercept = z
        y_pred = np.polyval(z, shots)
        
        # Compute residuals and the standard error of the regression
        residuals = np.array(accuracies) - y_pred
        n = len(shots)
        if n > 2:
            s_err = np.sqrt(np.sum(residuals**2) / (n - 2))
        else:
            s_err = 0
        
        shots_arr = np.array(shots)
        mean_x = np.mean(shots_arr)
        Sxx = np.sum((shots_arr - mean_x)**2)
        
        # t statistic for 95% CI (n-2 degrees of freedom)
        if n > 2:
            t_val = stats.t.ppf(0.975, n - 2)
        else:
            t_val = 0
        
        # Compute the confidence interval (CI) for the regression line at each shot:
        # CI = t_val * s_err * sqrt(1/n + (x - mean_x)^2/Sxx)
        if Sxx > 0:
            conf_interval = t_val * s_err * np.sqrt(1/n + ((shots_arr - mean_x)**2) / Sxx)
        else:
            conf_interval = np.zeros_like(shots_arr)
        
        # Use custom label if provided, else default to "Data Set X"
        label = labels[idx] if labels and idx < len(labels) else f'Data Set {idx + 1}'
        color = colors[idx % len(colors)]
        
        # Plot the data points (without error bars)
        plt.plot(shots, accuracies, 'o', color=color)###, label=label)
        # Plot the regression (fit) line
        plt.plot(shots, y_pred, '--', color=color, alpha=0.8,
                 label=f'{label}: slope={slope:.3f}x + {intercept:.3f}')
        # Plot the 95% confidence band around the regression line
        plt.fill_between(shots, y_pred - conf_interval, y_pred + conf_interval,
                         color=color, alpha=0.2)
    
    plt.xlabel('Number of Shots')
    plt.ylabel('Accuracy (%)')
    plt.title(f'Parity Task ({num_parities} parities)')
    plt.grid(True)
    plt.legend()
    plt.savefig(f"parity_accuracy{suffix}.png", dpi=100)
    plt.show()


def plot_log_fit_new(title, actual_shot_counts, nlls_mean_by_type, nlls_vars_by_type, fit_threshold=4, ax=None):
    use_plt = ax is None
    if use_plt:
        plt.figure(figsize=(6, 4))
    
    actual_shot_counts = np.array(actual_shot_counts)
    colors = ['gray', 'black', 'cyan', 'blue', 'brown', 'pink']
    
    # Build a list where each entry contains: (mean data, variance data, label, color)
    nlls_data = [
        (np.array(nlls_mean_by_type[label]), np.array(nlls_vars_by_type[label]), label, colors[i % len(colors)])
        for i, label in enumerate(nlls_mean_by_type.keys())
    ]
    
    plot_func = plt if use_plt else ax
    
    for nlls, nlls_vars, label, color in nlls_data:
        valid_plot = ~np.isnan(nlls)
        plot_func.scatter(actual_shot_counts[valid_plot], nlls[valid_plot], color=color, s=30)
        
        # Select points above the fit threshold and where the mean is not NaN
        fit_indices = (actual_shot_counts >= fit_threshold) & ~np.isnan(nlls)
        fit_shot_counts = actual_shot_counts[fit_indices]
        fit_nlls = nlls[fit_indices]
        fit_vars = nlls_vars[fit_indices]  # corresponding variances
        
        # Use the log2 of shot counts for the fitting process
        log_x = np.log2(fit_shot_counts)
        if len(log_x) > 1 and len(fit_nlls) > 1:
            # Weights: 1/sigma, where sigma is the standard deviation = sqrt(variance)
            weights = 1 / np.sqrt(fit_vars)
            # np.polyfit with cov=True returns the coefficients and the covariance matrix.
            fit_coeffs, cov = np.polyfit(log_x, fit_nlls, 1, w=weights, cov=True)
            slope, intercept = fit_coeffs[0], fit_coeffs[1]
            # Standard error of the slope is sqrt(covariance of the slope)
            slope_se = np.sqrt(cov[0, 0])
            margin = 1.96 * slope_se
            lower_bound = slope - margin
            upper_bound = slope + margin
            
            # Compute the best-fit line using the fitted coefficients
            fit_line = np.polyval(fit_coeffs, log_x)
            plot_func.plot(fit_shot_counts, fit_line, color=color, linestyle='-', linewidth=2)
            
            # Optionally, plot the predicted lines if the slope were at its lower/upper bound.
            lower_fit_line = lower_bound * log_x + intercept
            upper_fit_line = upper_bound * log_x + intercept
            plot_func.plot(fit_shot_counts, lower_fit_line, color=color, linestyle='--', linewidth=1)
            plot_func.plot(fit_shot_counts, upper_fit_line, color=color, linestyle='--', linewidth=1)
            
            # Add an entry to the legend including the CI bounds for the slope.
            plot_func.scatter([], [], color=color,
                                label=f'{label}: slope={slope:.3f} ({lower_bound:.3f}, {upper_bound:.3f}), int.={intercept:.2f}')
    
    min_shots = np.min(actual_shot_counts)
    max_shots = np.max(actual_shot_counts)
    min_exp = int(np.floor(np.log2(min_shots))) if min_shots > 0 else 0
    max_exp = int(np.ceil(np.log2(max_shots)))
    xticks = [0]+[2**exp for exp in range(min_exp, max_exp+1)]
    
    if use_plt:
        plt.xscale('symlog', base=2)
        plt.xticks(xticks)
        plt.xlabel('Number of shots', fontsize=8)
        plt.ylabel('Negative Log Likelihood', fontsize=8)
        plt.title(title, fontsize=10)
        plt.grid(True, which='both', linestyle='--', linewidth=0.5)
        plt.legend()
        plt.tight_layout()
        plt.savefig(title.replace(" ", "_") + ".png", dpi=100)
        plt.show()
    else:
        ax.set_xscale('symlog', base=2)
        ax.set_xticks(xticks)
        ax.set_xlabel('Number of shots', fontsize=8)
        ax.set_ylabel('Negative Log Likelihood', fontsize=8)
        ax.set_title(title, fontsize=10)
        ax.grid(True, which='both', linestyle='--', linewidth=0.5)
        #ax.legend()
        ax.legend(loc='upper center', bbox_to_anchor=(0.5, 1.15), ncol=1, fontsize=8, frameon=True)
    
    return ax if ax else None

def plot_log_fit_novars(title, actual_shot_counts, nlls_mean_by_type, fit_threshold=4, ax=None):
    use_plt = ax is None
    if use_plt:
        plt.figure(figsize=(6, 4))
    
    actual_shot_counts = np.array(actual_shot_counts)
    colors = ['black', 'saddlebrown', 'peru']
    colors = ['gray', 'black', 'cyan', 'blue', 'brown', 'pink']
    
    # Build a list where each entry contains: (mean data, variance data, label, color)
    nlls_data = [
        (np.array(nlls_mean_by_type[label]), label, colors[i % len(colors)])
        for i, label in enumerate(nlls_mean_by_type.keys())
    ]
    
    plot_func = plt if use_plt else ax
    
    for nlls, label, color in nlls_data:
        valid_plot = ~np.isnan(nlls)
        plot_func.scatter(actual_shot_counts[valid_plot], nlls[valid_plot], color=color, s=30)
        
        # Select points above the fit threshold and where the mean is not NaN
        fit_indices = (actual_shot_counts >= fit_threshold) & ~np.isnan(nlls)
        fit_shot_counts = actual_shot_counts[fit_indices]
        fit_nlls = nlls[fit_indices]
        
        # Use the log2 of shot counts for the fitting process
        log_x = np.log2(fit_shot_counts)
        if len(log_x) > 1 and len(fit_nlls) > 1:
            fit_coeffs = np.polyfit(log_x, fit_nlls, 1)
            slope, intercept = fit_coeffs[0], fit_coeffs[1]
            
            # Compute the best-fit line using the fitted coefficients
            fit_line = np.polyval(fit_coeffs, log_x)
###            plot_func.plot(fit_shot_counts, fit_line, color=color, linestyle='-', linewidth=2)
            plot_func.plot(fit_shot_counts, fit_nlls, color=color, linestyle='-', linewidth=2)
            
            # Add an entry to the legend including the CI bounds for the slope.
            plot_func.scatter([], [], color=color,
                                label=f'{label}')#: slope={slope:.3f}, int.={intercept:.1f}')
    
    min_shots = np.min(actual_shot_counts)
    max_shots = np.max(actual_shot_counts)
    min_exp = int(np.floor(np.log2(min_shots))) if min_shots > 0 else 0
    max_exp = int(np.ceil(np.log2(max_shots)))
    xticks = [0]+[2**exp for exp in range(min_exp, max_exp+1)]
    
    if use_plt:
        plt.xscale('symlog', base=2)
        plt.xticks(xticks)
        plt.xlabel('Number of shots', fontsize=8)
        plt.ylabel('Negative Log Likelihood', fontsize=8)
        plt.title(title, fontsize=10)
        plt.grid(True, which='both', linestyle='--', linewidth=0.5)
        plt.legend()
        plt.tight_layout()
        plt.savefig(title.replace(" ", "_") + ".png", dpi=100)
        plt.show()
    else:
        ax.set_xscale('symlog', base=2)
        ax.set_xticks(xticks)
        ax.set_xlabel('Number of shots', fontsize=8)
        ax.set_ylabel('Negative Log Likelihood', fontsize=8)
        ax.set_title(title, fontsize=10)
        ax.grid(True, which='both', linestyle='--', linewidth=0.5)
        ax.legend()
    
    return ax if ax else None

def plot_grid_new(plot_data, n_cols=2, save_path=None):
    n_plots = len(plot_data)
    n_rows = (n_plots + n_cols - 1) // n_cols
    
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(4*n_cols, 6*n_rows))
    if n_rows == 1 and n_cols == 1:
        axes = np.array([[axes]])
    elif n_rows == 1 or n_cols == 1:
        axes = axes.reshape(-1, 1) if n_cols == 1 else axes.reshape(1, -1)
    
    for idx, (title, counts, nlls, nllvars, threshold) in enumerate(plot_data):
        row, col = idx // n_cols, idx % n_cols
        if nllvars:
            plot_log_fit_new(title, counts, nlls, nllvars, threshold, axes[row, col])
        else:
            plot_log_fit_novars(title, counts, nlls, threshold, axes[row, col])
            
    # Hide empty subplots
    for idx in range(n_plots, n_rows * n_cols):
        row, col = idx // n_cols, idx % n_cols
        axes[row, col].set_visible(False)
    
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=100)
    plt.show()    