import pandas as pd, numpy as np
from sklearn.preprocessing import StandardScaler, PolynomialFeatures
from sklearn.linear_model import SGDRegressor
from sklearn.pipeline import make_pipeline
from sklearn.metrics import mean_absolute_error, root_mean_squared_error
import matplotlib.pyplot as plt
from pathlib import Path

## Read Excel and build the monthly sales table with Month_Index.
def load_monthly_sales(workbook: Path) -> pd.DataFrame:
    df = pd.read_excel(workbook, sheet_name='Sales_Cleaned') # Read cleaned sales data in sheet Sales_Cleaned
    df = df[df['Year'].between(2023, 2025)] # Filter for years 2023-2025
    monthly = (df.groupby(['Year', 'Month'], as_index=False)['Total Spent'] # Aggregate total spent by year and month
                 .sum()
                 .sort_values(['Year', 'Month']))
    monthly['Month_Index'] = (monthly['Year'] - 2023) * 12 + monthly['Month'] # Create Month_Index
    return monthly # Return the monthly DataFrame

## Split into train (2023) and test (2024–2025), return X/y.
def split_train_test(monthly: pd.DataFrame):
    train = monthly[monthly['Year'] == 2023] # Train set on 2023 data
    test = monthly[monthly['Year'].isin([2024, 2025])] # Test set on 2024-2025 data
    X_train, y_train = train[['Month_Index']], train['Total Spent'] # Features and target for train
    X_test, y_test = test[['Month_Index']], test['Total Spent'] # Features and target for test
    return train, test, X_train, y_train, X_test, y_test # Return train/test splits and X/y

## Further split train into train/val based on Month (<=9 for train, >9 for val).
def time_val_split(train_df: pd.DataFrame, X_train, y_train):
    train_mask = train_df['Month'] <= 9 # Train months are Jan-Sep
    val_mask = ~train_mask # Validation months are Oct-Dec
    X_tr, y_tr = X_train[train_mask], y_train[train_mask] # Train split
    X_val, y_val = X_train[val_mask], y_train[val_mask] # Validation split
    return X_tr, y_tr, X_val, y_val # Return train/val splits


### Build a pipeline model with scaling, polynomial features, and SGD regressor.
def build_model(degree: int, lr: float):
    return make_pipeline( # init a pipeline
        StandardScaler(), # Scale features 
        PolynomialFeatures(degree=degree, include_bias=False), # Add polynomial features
        SGDRegressor( # Stochastic Gradient Descent regressor
            loss='squared_error', # Use squared error loss
            penalty='l2', # L2 regularization
            alpha=1e-6, # Regularization strength
            max_iter=2000, # Max iterations
            tol=1e-6, # Tolerance for stopping
            learning_rate='constant', # Constant learning rate
            eta0=lr, # Initial learning rate
            random_state=42, # seed for for reproducibility
        ),
    )


### Search over degrees and learning rates to find best hyperparameters.
def search_hyperparams(X_tr, y_tr, X_val, y_val, degrees, lrs):
    results = [] # initialize results list
    for d in degrees: # loop over degrees
        for lr in lrs: # loop over learning rates
            pipe = build_model(d, lr) # build model with current hyperparams
            pipe.fit(X_tr, y_tr) # fit model on training set
            pred_val = pipe.predict(X_val) # predict on validation set
            mae = mean_absolute_error(y_val, pred_val) # calculate MAE
            rmse = root_mean_squared_error(y_val, pred_val) # calculate RMSE
            results.append((d, lr, mae, rmse)) # store results
    best_d, best_lr, _, _ = sorted(results, key=lambda r: r[2])[0] # get best hyperparams by MAE 
    return best_d, best_lr, results # return best hyperparams and all results


### Fit the best model and get predictions and metrics on train/test sets.
def fit_and_predict(best_d, best_lr, X_train, y_train, X_test, y_test, train_df, test_df):
    model = build_model(best_d, best_lr) # build model with best hyperparams from search function
    model.fit(X_train, y_train) # fit model on full training set
    pred_train = model.predict(X_train) # predict on training set
    pred_test = model.predict(X_test) # predict on test set
    mae_train = mean_absolute_error(y_train, pred_train) # calculate train MAE
    rmse_train = root_mean_squared_error(y_train, pred_train) # calculate train RMSE
    mae_test = mean_absolute_error(y_test, pred_test) # calculate test MAE
    rmse_test = root_mean_squared_error(y_test, pred_test) # calculate test RMSE
    pred_df = pd.concat( # concatenate train and test predictions
        [
            train_df.assign(Split='Train', Predicted=pred_train), # combine train actuals with predictions
            test_df.assign(Split='Test', Predicted=pred_test), # combine test actuals with predictions
        ],
        axis=0, # axis 0 means stack rows
    )
    pred_df['Residual'] = pred_df['Predicted'] - pred_df['Total Spent'] # calculate residuals for all data
    pred_df.rename(columns={'Total Spent': 'Actual'}, inplace=True) # rename column for clarity
    pred_df = pred_df[ 
        ['Year', 'Month', 'Month_Index', 'Split', 'Actual', 'Predicted', 'Residual'] # reorder columns for output
    ]
    return model, pred_df, (mae_train, rmse_train, mae_test, rmse_test) # return model, predictions, and metrics


### Save predictions to csv and plots to files.
def save_artifacts(pred_df: pd.DataFrame, best_d: int, best_lr: float, out_dir: Path):
    out_dir.mkdir(parents=True, exist_ok=True) # create output directory if it doesn't exist
    csv_path = out_dir / 'monthly_predictions.csv' # define path for csv output
    try:
        pred_df.to_csv(csv_path, index=False) # try to save predictions to csv
        print(f"Saved predictions to {csv_path}.")
    except PermissionError: # handle case where file is open or cannot be written
        alt_path = out_dir / 'monthly_predictions_out.csv' # alternative path if original fails
        print(f"Could not write {csv_path} (maybe open). Saved to {alt_path}.") 
        pred_df.to_csv(alt_path, index=False) # save to alternative path

    plt.figure(figsize=(10, 5)) # create figure for actual vs predicted plot
    plt.plot(pred_df['Month_Index'], pred_df['Actual'], label='Actual') # plot actual values
    plt.plot( # init plot predicted values
        pred_df['Month_Index'], # x values
        pred_df['Predicted'], #  y values
        label=f'Predicted (Deg {best_d}, lr={best_lr})', # label with hyperparams
    )
    plt.xlabel('Month_Index (Jan-2023 = 1)') # label x-axis
    plt.ylabel('Total Spent') # label y-axis
    plt.title('Actual vs Predicted') # title of the plot
    plt.legend() # show legend
    plt.tight_layout() # adjust layout
    plt.savefig(out_dir / 'actual_vs_pred.png', dpi=200) # save plot to file

    plt.figure(figsize=(10, 3)) # create figure for residuals plot
    plt.axhline(0, color='k', lw=1) # add horizontal line at y=0
    plt.scatter(pred_df['Month_Index'], pred_df['Residual']) # scatter plot of residuals
    plt.xlabel('Month_Index') # label x-axis
    plt.ylabel('Residual') # label y-axis
    plt.title('Residuals by Month') # title of the plot
    plt.tight_layout() # adjust layout 
    plt.savefig(out_dir / 'residuals.png', dpi=200) # save plot to file


### Main function to run the regression flow.
def main():
    base_dir = Path(__file__).resolve().parent.parent # base directory
    workbook = base_dir / 'D4_work.xlsx' # path to the Excel workbook
    out_dir = base_dir / 'python' / 'output' # output directory for artifacts

    monthly = load_monthly_sales(workbook) # load and preprocess monthly sales data
    train_df, test_df, X_train, y_train, X_test, y_test = split_train_test(monthly) # split into train/test sets
    X_tr, y_tr, X_val, y_val = time_val_split(train_df, X_train, y_train) # split train into train/val sets

    degrees = [1, 2, 3, 4] # degrees to search
    lrs = [1e-4, 1e-3, 1e-2] # learning rates to search
    best_d, best_lr, results = search_hyperparams(X_tr, y_tr, X_val, y_val, degrees, lrs) # hyperparameter search
    model, pred_df, metrics = fit_and_predict( # fit best model and get predictions/metrics
        best_d, best_lr, X_train, y_train, X_test, y_test, train_df, test_df
    )
    mae_train, rmse_train, mae_test, rmse_test = metrics # unpack metrics

    save_artifacts(pred_df, best_d, best_lr, out_dir) # save predictions and plots

    print(f"Best Model: Degree={best_d}, Learning Rate={best_lr}") # print best model hyperparameters
    print(f"Train MAE: {mae_train:.2f}, Train RMSE: {rmse_train:.2f}") # print training metrics 
    print(f"Test MAE: {mae_test:.2f}, Test RMSE: {rmse_test:.2f}") # print test metrics


if __name__ == '__main__':
    main()
