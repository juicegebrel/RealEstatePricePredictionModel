# Real Estate Price Prediction – Custom Multi-Layer Perceptron (C#)

Regression of Philippine house prices with the hand-written 3-layer MLP (`INeuron` / `HNeuron` / `ONeuron` / `NeuralNet`
in namespace `Backprop`) from the previous activity. No external ML libraries are used.

## How to run
1. Open `RealEstatePricePredictionModel.slnx` in Visual Studio (Windows Forms, .NET Framework project).
2. Build and run (F5). `PH_Housing.csv` is copied next to the exe automatically (or use *Browse...* for another CSV with the same columns).
3. In the window:
   * **Tune hyperparameters** – grid search (48 training-run cells x 3 seeds; takes a few minutes). The best settings are copied into the controls.
   * **Train model** – final run with the chosen settings. Prints the loss every 50 epochs, draws the loss curve and the actual-vs-predicted plot, prints all metrics.
   * **Predict price** – estimate one listing with the trained model.
   * **Load weights...** – restore a saved `Results/model_weights.txt` instead of training again. The hidden-layer size is derived from the file; it prints the train/validation/test metrics, redraws the plot and enables **Predict price**. Select the **same dataset** the weights were trained on (the scalers are rebuilt from it with the fixed split seed).
4. Output files go to the `Results/` folder in the project root (commit them to GitHub):

| File | Content |
|---|---|
| `training_log.csv` | Train/validation loss for **every** epoch |
| `training_summary.txt` | Final metrics table + automatic interpretation |
| `test_predictions.csv` | Actual vs predicted price for every test listing |
| `experiment_results.csv`, `tuning_summary.txt` | Full hyperparameter grid |
| `model_weights.txt` | Trained weights (`NeuralNet.saveWeights`) |

## Data description
`PH_Housing.csv` – 1,500 Philippine property listings (house & lot, townhouses, ...). Columns: `HouseID, Description, Location, Price (PHP), Bedrooms, Bathrooms, Floor Area (sqm), Land Area (sqm), Latitude, Longitude`.
The target is **Price**. `HouseID`, `Description`, `Location`, `Latitude` and `Longitude` are **not** used as inputs - the model predicts from the property's size and room counts only.

Data-quality issues found and how they are handled:

| Issue | Handling |
|---|---|
| 49 rows without a Price | skipped (no target) -> 1,451 usable rows |
| Missing Bedrooms (63), Bathrooms (80), Floor Area (37), Land Area (23) | median imputation (median of the **train** split) **plus** a 0/1 "was missing" flag per column (a missing floor area usually means a lot-only listing) |
| Price from PHP 300k to PHP 2.5 billion, areas from 22 to 25,000 sqm (heavily right-skewed) | the network predicts **ln(price)**; areas are log-transformed |
| Implausible counts (e.g. 43 bathrooms) | room counts above 10 are capped |

## Pipeline
* **Parsing** (`CsvHouseLoader`): columns found by header name, quoted fields supported (descriptions contain commas), bad rows skipped and counted.
* **Features** (`FeatureEncoder`) -> **8 inputs**: ln(Floor Area), ln(Land Area), Bedrooms, Bathrooms and 4 "missing" flags.
* **Preprocessing** (`MedianImputer`, `MinMaxScaler`, `TargetScaler`): impute -> MinMax scale to [0,1]; target ln(price) scaled to [0.1, 0.9] (the output neuron is a sigmoid). All statistics are fitted on the **training split only** (no leakage).
* **Split** (`DataPreparer`): shuffled with a fixed seed into 70 % train / 15 % validation / 15 % test. Tuning uses validation only; the test set is evaluated once, in the final run.

## Network and training
Input(8) -> Hidden(n, sigmoid) -> Output(1, sigmoid), online (per-sample) gradient descent with back-propagation inside `NeuralNet.learn()`; the samples are reshuffled every epoch. Delta rule: `d_out = o(1-o)(t-o)`, `d_hidden = h(1-h) * sum(d_out * w)`, `w += lr * d * input`. Loss = MSE on the scaled ln(price).

Changes to the original classes (minimal): the learning rates were `const` and are now adjustable (`setLearningRates`); `seedRandom` makes weight initialisation reproducible.

## Hyperparameter tuning
Grid: hidden nodes {4, 8, 16, 32} x learning rate {0.05, 0.1, 0.3, 0.5} x epochs {100, 300, 500}, every cell averaged over 3 random seeds and ranked by **validation RMSLE**. (Epoch checkpoints come from one 500-epoch run per seed - identical to separate shorter runs with the same seed.)

## Evaluation
Because a few listings cost hundreds of millions of pesos, squared errors in pesos are dominated by outliers. The report therefore shows both kinds of metrics, on train / validation / test, next to a **predict-the-typical-price baseline**:
* log scale: **RMSLE**, **R-squared of ln(price)** (used for model selection);
* percentage: **median absolute % error**, **share within +-25 %**, MAPE;
* raw pesos: MSE, RMSE, MAE, R-squared.

The train-vs-test gap shows over-fitting.

## Results (fill in from your run)
Paste the table from `Results/training_summary.txt` and the best row of `Results/tuning_summary.txt` here, plus a short analysis (which hyperparameters mattered, over/under-fitting, where the model fails - e.g. very expensive or lot-only listings).

| Best config (hidden / lr / epochs) | Test RMSLE | Test R2 (ln) | Median error | Baseline RMSLE |
|---|---|---|---|---|
| | | | | |

## Project structure
`Form1.cs` UI · `Trainer.cs` training loop · `ExperimentRunner.cs` grid search · `CsvHouseLoader.cs`, `FeatureEncoder.cs`, `MedianImputer.cs`, `MinMaxScaler.cs`, `TargetScaler.cs`, `DataPreparer.cs` data pipeline · `RegressionMetrics.cs` metrics · `ReportBuilder.cs`, `ResultsWriter.cs` output · `*ChartControl.cs` GDI+ charts · `BackNeural 1.cs`, `InputN 1.cs`, `HiddenN 1.cs`, `OuputN 1.cs` the neural-network classes.
