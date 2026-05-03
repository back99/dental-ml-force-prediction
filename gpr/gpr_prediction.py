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
force_labels = {
    'fx': 'Fx (N)', 'fy': 'Fy (N)', 'fz': 'Fz (N)',
    'tx': 'Tx (Nmm)', 'ty': 'Ty (Nmm)', 'tz': 'Tz (Nmm)'
}

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
        f'GPR (GPU) — {tooth_label}\n'
        f'Training: 0.25 + 0.5mm (full data)  ->  Predict 0.75mm & 1.0mm',
        fontsize=12
    )

    csv_data = {'time_hours': time_range}

    for ax, target in zip(axes.flatten(), targets):
        print(f"  {target} | {len(raw_025)+len(raw_05)} rows training...")

        train = pd.concat([
            raw_025[['time_hours', 'thickness', target]],
            raw_05[['time_hours', 'thickness', target]]
        ], ignore_index=True)

        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(train[['time_hours', 'thickness']].values)
        y_np = train[target].values

        train_x = torch.tensor(X_scaled, dtype=torch.float32).to(device)
        train_y = torch.tensor(y_np,     dtype=torch.float32).to(device)

        X_075 = scaler.transform(np.column_stack([time_range, np.full(len(time_range), 0.75)]))
        X_100 = scaler.transform(np.column_stack([time_range, np.full(len(time_range), 1.0)]))
        test_x_075 = torch.tensor(X_075, dtype=torch.float32).to(device)
        test_x_100 = torch.tensor(X_100, dtype=torch.float32).to(device)

        model, likelihood = train_gpr(train_x, train_y)

        model.eval()
        likelihood.eval()
        with torch.no_grad(), gpytorch.settings.fast_pred_var():
            pred_075 = likelihood(model(test_x_075))
            pred_100 = likelihood(model(test_x_100))

        y_075 = pred_075.mean.cpu().numpy()
        y_100 = pred_100.mean.cpu().numpy()
        std_075 = pred_075.stddev.cpu().numpy()
        std_100 = pred_100.stddev.cpu().numpy()

        csv_data[f'{target}_gpr_075'] = y_075
        csv_data[f'{target}_std_075'] = std_075
        csv_data[f'{target}_gpr_100'] = y_100
        csv_data[f'{target}_std_100'] = std_100

        ax.plot(mean_025['time_hours'], mean_025[target], color='steelblue', linewidth=1.5, label='Actual 0.25mm')
        ax.plot(mean_05['time_hours'],  mean_05[target],  color='darkorange', linewidth=1.5, label='Actual 0.5mm')
        ax.plot(time_range, y_075, color='green', linewidth=2, linestyle='--', label='GPR 0.75mm')
        ax.fill_between(time_range, y_075-2*std_075, y_075+2*std_075, color='green', alpha=0.15)
        ax.plot(time_range, y_100, color='red', linewidth=2, linestyle='--', label='GPR 1.0mm')
        ax.fill_between(time_range, y_100-2*std_100, y_100+2*std_100, color='red', alpha=0.15)

        ax.set_title(force_labels[target])
        ax.set_xlabel('Time (hours)')
        ax.set_ylabel(force_labels[target])
        ax.legend(fontsize=7)
        ax.grid(True, alpha=0.3)

    pd.DataFrame(csv_data).to_csv(f'gpr_{tooth_label}.csv', index=False)
    print(f"  Saved: gpr_{tooth_label}.csv")

    plt.tight_layout()
    plt.savefig(f'gpr_{tooth_label}.png', dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved: gpr_{tooth_label}.png")

print("\nDone!")
