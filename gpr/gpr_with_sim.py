import os
os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'

import pandas as pd
import numpy as np
import torch
import gpytorch
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

class ExactGPModel(gpytorch.models.ExactGP):
    def __init__(self, train_x, train_y, likelihood):
        super().__init__(train_x, train_y, likelihood)
        self.mean_module = gpytorch.means.ConstantMean()
        self.covar_module = gpytorch.kernels.ScaleKernel(
            gpytorch.kernels.MaternKernel(nu=1.5)
        )
    def forward(self, x):
        mean_x = self.mean_module(x)
        covar_x = self.covar_module(x)
        return gpytorch.distributions.MultivariateNormal(mean_x, covar_x)

def train_gpr(train_x, train_y, n_iter=100):
    likelihood = gpytorch.likelihoods.GaussianLikelihood().to(device)
    model = ExactGPModel(train_x, train_y, likelihood).to(device)
    model.train()
    likelihood.train()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.1)
    mll = gpytorch.mlls.ExactMarginalLogLikelihood(likelihood, model)
    for i in range(n_iter):
        optimizer.zero_grad()
        output = model(train_x)
        loss = -mll(output, train_y)
        loss.backward()
        optimizer.step()
        if (i+1) % 20 == 0:
            print(f"    iter {i+1}/{n_iter} | loss={loss.item():.4f}")
    return model, likelihood

dpa_025 = pd.read_csv('dpa_025.csv')
dpa_05  = pd.read_csv('dpa_05.csv')

dpa_025['tooth_num'] = dpa_025['tooth'].str.extract(r'(\d+)').astype(int)
dpa_05['tooth_num']  = dpa_05['tooth'].str.extract(r'(\d+)').astype(int)

targets = ['fx', 'fy', 'fz', 'tx', 'ty', 'tz']
weights = [0.5, 0.8, 1.0, 1.2, 1.5]

force_labels = {
    'fx': 'Fx (N)', 'fy': 'Fy (N)', 'fz': 'Fz (N)',
    'tx': 'Tx (Nmm)', 'ty': 'Ty (Nmm)', 'tz': 'Tz (Nmm)'
}

colors_10  = ['#a8d5a2', '#6dbf67', '#3a9e3a', '#1f7a1f', '#0d4d0d']
colors_125 = ['#f4a8a8', '#e96b6b', '#d93636', '#b01f1f', '#7a0d0d']

for tooth_label, tooth_num in [('U6', 6), ('U7', 7)]:
    print(f"\nProcessing {tooth_label}...")

    raw_025 = dpa_025[dpa_025['tooth_num'] == tooth_num].copy()
    raw_025['thickness'] = 0.25
    raw_05  = dpa_05[dpa_05['tooth_num'] == tooth_num].copy()
    raw_05['thickness'] = 0.5

    mean_025 = raw_025.groupby('time_hours')[targets].mean().reset_index()
    mean_05  = raw_05.groupby('time_hours')[targets].mean().reset_index()

    time_range = np.linspace(raw_025['time_hours'].min(),
                             raw_025['time_hours'].max(), 300)

    fig, axes = plt.subplots(2, 3, figsize=(16, 9))
    fig.suptitle(
        f'GPR (GPU) with Simulated 0.75mm — {tooth_label}\n'
        f'Training: 0.25 + 0.5 + sim_0.75 (full data)  ->  Predict 1.0mm & 1.25mm',
        fontsize=12
    )

    for ax, target in zip(axes.flatten(), targets):

        ax.plot(mean_025['time_hours'], mean_025[target],
                color='steelblue', linewidth=2, label='Actual 0.25mm', zorder=5)
        ax.plot(mean_05['time_hours'], mean_05[target],
                color='darkorange', linewidth=2, label='Actual 0.5mm', zorder=5)

        for i, w in enumerate(weights):
            w_str = str(w).replace('.', '')
            sim = pd.read_csv(f'sim_075_{tooth_label}_w{w_str}.csv')
            sim['thickness'] = 0.75

            train = pd.concat([
                raw_025[['time_hours', 'thickness', target]],
                raw_05[['time_hours', 'thickness', target]],
                sim[['time_hours', 'thickness', target]]
            ], ignore_index=True)

            scaler = StandardScaler()
            X_scaled = scaler.fit_transform(train[['time_hours', 'thickness']].values)
            y_np = train[target].values

            train_x = torch.tensor(X_scaled, dtype=torch.float32).to(device)
            train_y = torch.tensor(y_np,     dtype=torch.float32).to(device)

            X_10  = scaler.transform(np.column_stack([time_range, np.full(len(time_range), 1.0)]))
            X_125 = scaler.transform(np.column_stack([time_range, np.full(len(time_range), 1.25)]))
            test_x_10  = torch.tensor(X_10,  dtype=torch.float32).to(device)
            test_x_125 = torch.tensor(X_125, dtype=torch.float32).to(device)

            print(f"  w={w} | {target} | {len(train)} rows training...")
            model, likelihood = train_gpr(train_x, train_y)

            model.eval()
            likelihood.eval()
            with torch.no_grad(), gpytorch.settings.fast_pred_var():
                pred_10  = likelihood(model(test_x_10))
                pred_125 = likelihood(model(test_x_125))

            y_10  = pred_10.mean.cpu().numpy()
            y_125 = pred_125.mean.cpu().numpy()

            print(f"    1.0mm={y_10.mean():.4f} | 1.25mm={y_125.mean():.4f}")

            # GPU 메모리 해제
            del model, likelihood, train_x, train_y
            torch.cuda.empty_cache()

            ax.plot(time_range, y_10,  color=colors_10[i],  linewidth=1.2,
                    linestyle='--', label=f'1.0mm  w={w}')
            ax.plot(time_range, y_125, color=colors_125[i], linewidth=1.2,
                    linestyle=':',  label=f'1.25mm w={w}')

            pd.DataFrame({
                'time_hours':        time_range,
                f'{target}_gpr_100': y_10,
                f'{target}_gpr_125': y_125
            }).to_csv(f'gpr_sim_{tooth_label}_w{w_str}_{target}.csv', index=False)

        ax.set_title(force_labels[target])
        ax.set_xlabel('Time (hours)')
        ax.set_ylabel(force_labels[target])
        ax.legend(fontsize=6, ncol=2)
        ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(f'gpr_sim_{tooth_label}.png', dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved: gpr_sim_{tooth_label}.png")

print("\nDone!")
