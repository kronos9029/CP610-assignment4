import pandas as pd, numpy as np
from sklearn.preprocessing import StandardScaler, PolynomialFeatures
from sklearn.linear_model import SGDRegressor
from sklearn.pipeline import make_pipeline
from sklearn.metrics import mean_absolute_error, root_mean_squared_error
import matplotlib.pyplot as plt
from pathlib import Path


def load_monthly_sales(workbook: Path) -> pd.DataFrame:
    """Read Excel and build the monthly sales table with Month_Index."""
    df = pd.read_excel(workbook, sheet_name='Sales_Cleaned')
    df = df[df['Year'].between(2023, 2025)]
    monthly = (df.groupby(['Year', 'Month'], as_index=False)['Total Spent']
                 .sum()
                 .sort_values(['Year', 'Month']))
    monthly['Month_Index'] = (monthly['Year'] - 2023) * 12 + monthly['Month']
    return monthly


def split_train_test(monthly: pd.DataFrame):
    """Split into train (2023) and test (2024–2025), return X/y."""
    train = monthly[monthly['Year'] == 2023]
    test = monthly[monthly['Year'].isin([2024, 2025])]
    X_train, y_train = train[['Month_Index']], train['Total Spent']
    X_test, y_test = test[['Month_Index']], test['Total Spent']
    return train, test, X_train, y_train, X_test, y_test


def time_val_split(train_df: pd.DataFrame, X_train, y_train):
    """Hold out last 3 months of 2023 for validation."""
    train_mask = train_df['Month'] <= 9
    val_mask = ~train_mask
    X_tr, y_tr = X_train[train_mask], y_train[train_mask]
    X_val, y_val = X_train[val_mask], y_train[val_mask]
    return X_tr, y_tr, X_val, y_val


def build_model(degree: int, lr: float):
    """Make a pipeline with scaling, polynomial features, and SGD regressor."""
    return make_pipeline(
        StandardScaler(),
        PolynomialFeatures(degree=degree, include_bias=False),
        SGDRegressor(
            loss='squared_error',
            penalty='l2',
            alpha=1e-6,
            max_iter=2000,
            tol=1e-6,
            learning_rate='constant',
            eta0=lr,
            random_state=42,
        ),
    )


def search_hyperparams(X_tr, y_tr, X_val, y_val, degrees, lrs):
    """Try many degrees and learning rates, pick the best by val MAE."""
    results = []
    for d in degrees:
        for lr in lrs:
            pipe = build_model(d, lr)
            pipe.fit(X_tr, y_tr)
            pred_val = pipe.predict(X_val)
            mae = mean_absolute_error(y_val, pred_val)
            rmse = root_mean_squared_error(y_val, pred_val)
            results.append((d, lr, mae, rmse))
    best_d, best_lr, _, _ = sorted(results, key=lambda r: r[2])[0]
    return best_d, best_lr, results


def fit_and_predict(best_d, best_lr, X_train, y_train, X_test, y_test, train_df, test_df):
    """Fit the best model and get train/test predictions and metrics."""
    model = build_model(best_d, best_lr)
    model.fit(X_train, y_train)
    pred_train = model.predict(X_train)
    pred_test = model.predict(X_test)
    mae_train = mean_absolute_error(y_train, pred_train)
    rmse_train = root_mean_squared_error(y_train, pred_train)
    mae_test = mean_absolute_error(y_test, pred_test)
    rmse_test = root_mean_squared_error(y_test, pred_test)
    pred_df = pd.concat(
        [
            train_df.assign(Split='Train', Predicted=pred_train),
            test_df.assign(Split='Test', Predicted=pred_test),
        ],
        axis=0,
    )
    pred_df['Residual'] = pred_df['Predicted'] - pred_df['Total Spent']
    pred_df.rename(columns={'Total Spent': 'Actual'}, inplace=True)
    pred_df = pred_df[
        ['Year', 'Month', 'Month_Index', 'Split', 'Actual', 'Predicted', 'Residual']
    ]
    return model, pred_df, (mae_train, rmse_train, mae_test, rmse_test)


def save_artifacts(pred_df: pd.DataFrame, best_d: int, best_lr: float, out_dir: Path):
    """Save csv and plots so we can show results."""
    out_dir.mkdir(parents=True, exist_ok=True)
    csv_path = out_dir / 'monthly_predictions.csv'
    try:
        pred_df.to_csv(csv_path, index=False)
    except PermissionError:
        alt_path = out_dir / 'monthly_predictions_out.csv'
        print(f"Could not write {csv_path} (maybe open). Saved to {alt_path}.")
        pred_df.to_csv(alt_path, index=False)

    plt.figure(figsize=(10, 5))
    plt.plot(pred_df['Month_Index'], pred_df['Actual'], label='Actual')
    plt.plot(
        pred_df['Month_Index'],
        pred_df['Predicted'],
        label=f'Predicted (Deg {best_d}, lr={best_lr})',
    )
    plt.xlabel('Month_Index (Jan-2023 = 1)')
    plt.ylabel('Total Spent')
    plt.title('Actual vs Predicted')
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_dir / 'actual_vs_pred.png', dpi=200)

    plt.figure(figsize=(10, 3))
    plt.axhline(0, color='k', lw=1)
    plt.scatter(pred_df['Month_Index'], pred_df['Residual'])
    plt.xlabel('Month_Index')
    plt.ylabel('Residual')
    plt.title('Residuals by Month')
    plt.tight_layout()
    plt.savefig(out_dir / 'residuals.png', dpi=200)


def main():
    """Run the full regression flow end to end."""
    base_dir = Path(__file__).resolve().parent.parent
    workbook = base_dir / 'D4_work.xlsx'
    out_dir = base_dir / 'python'

    monthly = load_monthly_sales(workbook)
    train_df, test_df, X_train, y_train, X_test, y_test = split_train_test(monthly)
    X_tr, y_tr, X_val, y_val = time_val_split(train_df, X_train, y_train)

    degrees = [1, 2, 3, 4]
    lrs = [1e-4, 1e-3, 1e-2]
    best_d, best_lr, results = search_hyperparams(X_tr, y_tr, X_val, y_val, degrees, lrs)
    model, pred_df, metrics = fit_and_predict(
        best_d, best_lr, X_train, y_train, X_test, y_test, train_df, test_df
    )
    mae_train, rmse_train, mae_test, rmse_test = metrics

    save_artifacts(pred_df, best_d, best_lr, out_dir)

    print(f"Best Model: Degree={best_d}, Learning Rate={best_lr}")
    print(f"Train MAE: {mae_train:.2f}, Train RMSE: {rmse_train:.2f}")
    print(f"Test MAE: {mae_test:.2f}, Test RMSE: {rmse_test:.2f}")


if __name__ == '__main__':
    main()
