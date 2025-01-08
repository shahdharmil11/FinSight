import yfinance as yf
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
import logging
import mlflow
import numpy as np
import optuna
import time
import json
from functools import partial
from google.cloud import storage
import os
from tensorflow.keras.models import load_model
from functools import partial
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.losses import MeanSquaredError
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras import metrics
from FinSight.model import *
from sklearn.metrics import mean_squared_error
import joblib


# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Configuring Mlflow settings
os.environ["GIT_PYTHON_REFRESH"] = "quiet"
experiment_name = "LSTM Stock Prediction"
MLFLOW_TRACKING_URI = "http://mlflow-server:5000"
mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)

# Check if the experiment exists, and create it if it doesn't
experiment = mlflow.get_experiment_by_name(experiment_name)
if experiment is None:
    experiment_id = mlflow.create_experiment(
        experiment_name,
        artifact_location=os.path.abspath(os.path.join(os.getcwd(), "mlruns","artifacts")),
        tags={"version": "v2", "priority": "P1"},
    )
else:
    experiment_id = experiment.experiment_id

# Set the experiment
mlflow.set_experiment(experiment_name)
mlflow.autolog()
mlflow.enable_system_metrics_logging()

# Buckets Config
storage_client = storage.Client.from_service_account_json("./dags/FinSight/.google-auth.json")
bucket_name = "mlops_deploy_storage"
bucket = storage_client.bucket(bucket_name)

def download_and_uploadToDVCBucket(ticker_symbol, start_date, end_date, ti):
    """
    Download stock data from Yahoo Finance and upload it to a Google Cloud Storage bucket.
    """
    mlflow.start_run(run_name="Download Data")
    time.sleep(15)
    try:
        logging.info(f"Downloading data for {ticker_symbol} from {start_date} to {end_date}.")
        mlflow.log_param("ticker_symbol", ticker_symbol)
        mlflow.log_param("start_date", start_date)
        mlflow.log_param("end_date", end_date)
        
        stock_data = yf.download(ticker_symbol, start=start_date, end=end_date)
        filename = "./data/train/" + f"{ticker_symbol}_stock_data_{start_date}_{end_date}.csv"
        
        # Save stock data to CSV
        stock_data.to_csv(filename)
        logging.info(f"Data downloaded and saved as {filename}")

        mlflow.log_artifact(filename)
            
        # Upload the plot to GCS
        destination_blob_name = 'data/train/' + f"{ticker_symbol}_stock_data_{start_date}_{end_date}.csv"
        blob = bucket.blob(destination_blob_name)
        blob.upload_from_filename(filename)

        return stock_data

    except Exception as e:
        logging.error(f"Failed to download or upload data: {e}")
        mlflow.log_param("error", str(e))
        raise
    finally:
        mlflow.end_run()
        return stock_data
    

def visualize_raw_data(stock_data,file_path):
    """
    Read stock data from a CSV file and visualize it, saving the plot to a specified GCS location.
    """
    mlflow.start_run(run_name="Visualize Data")
    time.sleep(15)
    try:
        df = pd.DataFrame(stock_data)

        logging.info("Converting 'Date' column to datetime format and setting it as index.")
        df['Date'] = df.index
        df.set_index('Date', inplace=True)

        logging.info("Plotting data.")
        plt.figure(figsize=(14, 7))
        plt.suptitle('Stock Data Visualizations', fontsize=16)

        plt.subplot(3, 1, 1)
        plt.plot(df.index, df['Open'], label='Open Price', color='blue')
        plt.title('Open Prices Over Time')
        plt.xlabel('Date')
        plt.ylabel('Price')
        plt.legend()

        plt.subplot(3, 1, 2)
        plt.plot(df.index, df['Close'], label='Close Price', color='green')
        plt.title('Close Prices Over Time')
        plt.xlabel('Date')
        plt.ylabel('Price')
        plt.legend()

        plt.subplot(3, 1, 3)
        plt.plot(df.index, df['Volume'], label='Volume', color='red')
        plt.title('Trading Volume Over Time')
        plt.xlabel('Date')
        plt.ylabel('Volume')
        plt.legend()

        plt.tight_layout(rect=[0, 0, 1, 0.96])
        
        plt.savefig(file_path)
        
        file_name = os.path.basename(file_path)
        
        # Upload the plot to GCS
        destination_blob_name = 'visualization/' + file_name
        blob = bucket.blob(destination_blob_name)
        blob.upload_from_filename(file_path)

    except Exception as e:
        logging.error(f"Failed to visualize Raw data: {e}")
        raise
    finally:
        mlflow.end_run()
        return df

def divide_train_eval_test_splits(df,ti):
    """
    - file_path (str): Path to the CSV file containing stock data.
    - test_size (float): Proportion of the dataset to include in the test split.
    - eval_size (float): Proportion of the train dataset to include in the eval split.
    - random_state (int): Random seed for reproducibility.
    
    Returns:
    - train_df (pd.DataFrame): DataFrame containing the training data.
    - eval_df (pd.DataFrame): DataFrame containing the evaluation data.
    - test_df (pd.DataFrame): DataFrame containing the testing data.
    """
    mlflow.start_run(run_name="Divide data set")
    time.sleep(15)
    # mlflow.log_params(test_size,eval_size,random_state)
    try:
        # logging.info(f"Reading data from {file_path}")
        # df = ti.xcom_pull(task_ids='download_upload_data', key='stock_data') #pd.read_csv(file_path)

        df = df['Open'].values

        # Reshape the data
        df = df.reshape(-1, 1) 
        
        train_df = np.array(df[:int(df.shape[0]*0.7)])
        eval_df = np.array(df[int(df.shape[0]*0.7):int(df.shape[0]*0.8)])
        test_df = np.array(df[int(df.shape[0]*0.8):])

        logging.info("Splitting data into train+eval and test sets.")
        # train_eval_df, test_df = train_test_split(df, test_size=test_size, random_state=random_state)
        
        logging.info("Further splitting train+eval set into train and eval sets.")
        # train_df, eval_df = train_test_split(train_eval_df, test_size=eval_size, random_state=random_state)

        train_dataset = mlflow.data.from_numpy(train_df)
        eval_dataset = mlflow.data.from_numpy(eval_df)
        test_dataset = mlflow.data.from_numpy(test_df)

        mlflow.log_input(train_dataset, context="training")
        mlflow.log_input(eval_dataset, context="Eval") 
        mlflow.log_input(test_dataset, context="Test") 

        train_df = pd.DataFrame(train_df)
        eval_df = pd.DataFrame(eval_df)
        test_df = pd.DataFrame(test_df)


        logging.info("Pushing data splits to XCom.")
        ti.xcom_push(key='train', value=train_df)
        ti.xcom_push(key='eval', value=eval_df)
        ti.xcom_push(key='test', value=test_df)
        return pd.DataFrame(train_df), pd.DataFrame(eval_df), pd.DataFrame(test_df)

    except Exception as e:
        logging.error(f"Failed to split data: {e}")
        raise
    finally:
        mlflow.end_run()

def handle_missing_values(df):
    """
    Handles null values in the DataFrame:
    - Forward fills null values in all columns.

    Parameters:
    df: Input stock data.

    Returns:
    pd.DataFrame: DataFrame with null values handled.
    """
    mlflow.start_run(run_name="Handle Missing Values - PreProcessing Step 1")
    try:
        logging.info("Handling missing values.")
        logging.info("Dataset before handling missing values:\n{}".format(df))

        # df = handle_null_open(df)
        df.fillna(method='ffill', inplace=True)

        return df
    except Exception as e:
        logging.error(f"Failed to handle missing values: {e}")
        raise
    finally:
        mlflow.end_run()

def handle_outliers(df):
    """
    Removes outliers from the specified columns in the DataFrame using the IQR method.

    Parameters:
    df (pd.DataFrame): Input DataFrame.
    columns (list): List of column names to check for outliers.

    Returns:
    pd.DataFrame: DataFrame with outliers removed.
    """
    mlflow.start_run(run_name="Handle Outlier Values - PreProcessing Step 2")
    try:
        logging.info("Handling outliers.")
        columns = df.columns.tolist()
        for column in columns:
            Q1 = df[column].quantile(0.25)
            Q3 = df[column].quantile(0.75)
            IQR = Q3 - Q1
            outliers = (df[column] < (Q1 - 1.5 * IQR)) | (df[column] > (Q3 + 1.5 * IQR))
            df = df[~outliers]
            logging.info(f"Removed outliers from column: {column}")

        return df
    except Exception as e:
        logging.error(f"Failed to handle outliers: {e}")
        raise
    finally:
        mlflow.end_run()


def visualize_df(df, file_path):
    """
    Visualize the preprocessed DataFrame, saving the plot to a specified location.
    """
    mlflow.start_run(run_name="Visualize Preprocessed Data")
    try:
        logging.info("Visualizing DataFrame.")
        
        # Create a new DataFrame from the input
        df = pd.DataFrame(df)

        plt.figure(figsize=(14, 10))
        plt.suptitle('Preprocessed Data Visualizations', fontsize=16)

        # Line plot
        plt.subplot(3, 1, 1)
        plt.plot(df.index, df.iloc[:, 0], label='Preprocessed Data', color='blue')
        plt.title('Preprocessed Data Over Time')
        plt.xlabel('Index')
        plt.ylabel('Value')
        plt.legend()

        # Histogram
        plt.subplot(3, 1, 2)
        plt.hist(df.iloc[:, 0], bins=30, color='green', edgecolor='black')
        plt.title('Distribution of Preprocessed Data')
        plt.xlabel('Value')
        plt.ylabel('Frequency')

        # Box plot
        plt.subplot(3, 1, 3)
        plt.boxplot(df.iloc[:, 0], vert=False, patch_artist=True, boxprops=dict(facecolor='red'))
        plt.title('Box Plot of Preprocessed Data')
        plt.xlabel('Value')

        plt.tight_layout(rect=[0, 0, 1, 0.96])
        
        plt.savefig(file_path)
 
        file_name = os.path.basename(file_path)
        
        # Upload the plot to GCS
        destination_blob_name = 'visualization/' + file_name
        blob = bucket.blob(destination_blob_name)
        blob.upload_from_filename(file_path)


    except Exception as e:
        logging.error(f"Failed to visualize DataFrame: {e}")
        raise
    finally:
        mlflow.end_run()
        return df


def apply_transformation(df,ti):
    """
    Normalizes the columns using MinMaxScaler, checks the data, and saves the preprocessed data.

    Parameters:
    df (pd.DataFrame): Input DataFrame with stock data.

    Returns:
    pd.DataFrame: DataFrame with scaled columns.
    """
    mlflow.start_run(run_name="Apply Transformations on Train Data Sets")    
    try:
        logging.info("Applying transformations to DataFrame.")
        scaler = MinMaxScaler(feature_range=(0,1))
        df= scaler.fit_transform(df)
        df = pd.DataFrame(df)
        ti.xcom_push(key='scalar', value=scaler)
    except Exception as e:
        logging.error(f"Failed to apply transformations: {e}")
        raise
    finally:
        mlflow.end_run()
        return df

def apply_transformation_eval_test(df,ti):
    """
    Normalizes the columns using MinMaxScaler, checks the data, and saves the preprocessed data.

    Parameters:
    df (pd.DataFrame): Input DataFrame with stock data.

    Returns:
    pd.DataFrame: DataFrame with scaled columns.
    """
    mlflow.start_run(run_name="Apply Transformations on Eval and Test Data Sets")    
    try:
        logging.info("Applying transformations to DataFrame.")
        scaler = ti.xcom_pull(task_ids='apply_transformation_training', key="scalar")
        df = scaler.transform(df)        
        df = pd.DataFrame(df)
        ti.xcom_push(key='scalar', value=scaler)

    except Exception as e:
        logging.error(f"Failed to apply transformations: {e}")
        raise
    finally:
        mlflow.end_run()
        return df

def generate_schema(df):
    """
    Generate schema from the DataFrame.

    Parameters:
    df (pd.DataFrame): Input data frame.

    Returns:
    dict: Schema definition with column types.
    """
    mlflow.start_run(run_name="Generate Schema", nested=True)   
    try: 
        schema = {}
        for column in df.columns:
            schema[column] = df[column].dtype
        mlflow.log_param("Scheme", schema)
    except Exception as e:
        logging.error(f"Failed to apply transformations: {e}")
        raise    
    finally:
        mlflow.end_run()
        return schema

def generate_statistics(df):
    """
    Generate descriptive statistics from the DataFrame.

    Parameters:
    df (pd.DataFrame): Input data frame.

    Returns:
    dict: Dictionary with descriptive statistics.
    """
    mlflow.start_run(run_name="Generate Schema", nested=True)   
    try: 
        # Generate descriptive statistics
        statistics = df.describe(include='all').transpose()

        # Convert the DataFrame to a dictionary
        statistics_dict = statistics.to_dict()
        mlflow.log_param("Stats", statistics_dict)

    except Exception as e:
        logging.error(f"Failed to apply transformations: {e}")
        raise  
    finally:
        mlflow.end_run()
        return statistics_dict

def generate_scheme_and_stats(df,ti):
    """
    Placeholder function for generating and validating scheme.
    """
    mlflow.start_run(run_name="Generate Schema & Statistics", nested=True)   
    try:
        logging.info("Generating scheme and stats.")
        
        # Scheme
        schema = generate_schema(df)
        logging.info(f"Schema: {schema}")

        # Stats
        data_stats = generate_statistics(df)
        logging.info(f"Statistics: \n{data_stats}")
        
        logging.info("Pushing data splits to XCom.")
        ti.xcom_push(key='schema', value=schema)
        ti.xcom_push(key='stats', value=data_stats)
        
        temp = {'RunID':ti.dag_run.run_id,'schema':str(schema),'stats':data_stats}

        # Upload the plot to GCS
        temp_str = json.dumps(temp)

        # Upload the JSON string to GCS
        destination_blob_name = 'data/schema_and_stats/schema_and_stats.json'
        blob = bucket.blob(destination_blob_name)
        blob.upload_from_string(temp_str, content_type='application/json')

    except Exception as e:
        logging.error(f"Failed to generate and validate scheme: {e}")
        raise
    finally:
        mlflow.end_run()
        return df

def detect_anomalies(eval_df, training_schema, training_stats):
    """
    Detect anomalies in the evaluation DataFrame by comparing it against the training schema and statistics.

    Parameters:
    eval_df (pd.DataFrame): Evaluation data frame.
    training_schema (dict): Schema of the training data.
    training_stats (dict): Statistics of the training data.

    Returns:
    dict: Detected anomalies including missing values and outliers.
    """
    mlflow.start_run(run_name="Detecting Anomalies", nested=True)   
    try:
        anomalies = {'missing_values': {}, 'outliers': {}, 'schema_mismatches': {}, 'statistical_anomalies': {}}

        # Detect missing values in the evaluation data
        missing_values = eval_df.isnull().sum()
        anomalies['missing_values'] = {col: count for col, count in missing_values.items() if count > 0}

        # Detect outliers in the evaluation data
        numeric_cols = eval_df.select_dtypes(include=['number']).columns
        for col in numeric_cols:
            Q1 = eval_df[col].quantile(0.25)
            Q3 = eval_df[col].quantile(0.75)
            IQR = Q3 - Q1
            lower_bound = Q1 - 1.5 * IQR
            upper_bound = Q3 + 1.5 * IQR
            outliers = eval_df[(eval_df[col] < lower_bound) | (eval_df[col] > upper_bound)][col]
            if not outliers.empty:
                anomalies['outliers'][col] = outliers.tolist()

        # Compare schema and detect schema mismatches
        for col in training_schema:
            if col not in eval_df.columns:
                anomalies['schema_mismatches'][col] = 'Column missing in evaluation data'
            elif eval_df[col].dtype != training_schema[col]:
                anomalies['schema_mismatches'][col] = f'Type mismatch: expected {training_schema[col]}, got {eval_df[col].dtype}'

        # Compare statistical properties
        for col in training_stats:
            if col in eval_df.columns:
                eval_mean = eval_df[col].mean()
                eval_std = eval_df[col].std()
                train_mean = training_stats[col]['mean']
                train_std = training_stats[col]['std']
                if abs(eval_mean - train_mean) > 3 * train_std:
                    anomalies['statistical_anomalies'][col] = {'eval_mean': eval_mean, 'train_mean': train_mean}
                if abs(eval_std - train_std) > 3 * train_std:
                    anomalies['statistical_anomalies'][col].update({'eval_std': eval_std, 'train_std': train_std}) 
    except Exception as e:
        logging.error(f"Failed to generate and validate scheme: {e}")
        raise
    finally:
        mlflow.end_run()
        return anomalies

def calculate_and_display_anomalies(eval_df, training_schema, training_stats, ti):
    """
    Calculate and display anomalies in the evaluation DataFrame by comparing it against the training schema and statistics.

    Parameters:
    eval_df (pd.DataFrame): Evaluation data frame.
    ti (TaskInstance): Airflow TaskInstance for XCom operations.
    training_schema (dict): Schema of the training data.
    training_stats (dict): Statistics of the training data.

    Returns:
    pd.DataFrame: The original evaluation DataFrame after anomaly detection.
    """
    mlflow.start_run(run_name="Calculating anomalies from Eval Df and Training (Schema, Stats)", nested=True)   
    try:
        logging.info("Calculating and Displaying Anomalies")

        # Log the values of training schema and stats for debugging purposes
        logging.info(f"Training Schema: {training_schema}")
        logging.info(f"Training Statistics: {training_stats}")

        # Detect anomalies
        anomalies = detect_anomalies(eval_df, training_schema, training_stats)
        logging.info(f"Anomalies: {anomalies}")

        temp = {'RunID':ti.dag_run.run_id,'Anomalies':anomalies}

        # Upload the plot to GCS
        temp_str = json.dumps(temp)

        # Upload the JSON string to GCS
        destination_blob_name = 'data/anomalies/anomalies.json'
        blob = bucket.blob(destination_blob_name)
        blob.upload_from_string(temp_str, content_type='application/json')
        
    except Exception as e:
        logging.error(f"Failed to calculate and display anomalies: {e}")
        raise
    finally:
        mlflow.end_run()
        return anomalies
    
# Create the function that will help us to create the datasets
def divide_features_and_labels(train_df, eval_df, test_df, ti):
    """
    Divide the data into features and labels.

    Parameters:
    - train_df (pd.DataFrame): DataFrame containing the training data.
    - eval_df (pd.DataFrame): DataFrame containing the evaluation data.
    - test_df (pd.DataFrame): DataFrame containing the testing data.
    - ti (TaskInstance): Airflow TaskInstance for XCom operations.
    """
    mlflow.start_run(run_name="Divide Data set into features and labels")   
    try:
        dfs = [train_df, eval_df, test_df]
        x_train = []
        x_eval = []
        x_test = []
        y_train = []
        y_eval = []
        y_test = []
        x = [x_train, x_eval, x_test]
        y = [y_train, y_eval, y_test]
        for ind, df in enumerate(dfs):
            for i in range(50, df.shape[0]):
                x[ind].append(df.iloc[i-50:i, 0].values) 
                y[ind].append(df.iloc[i, 0]) 
        
        ti.xcom_push(key='x', value=x)
        ti.xcom_push(key='y', value=y)

        mlflow.log_params({"x": x,"y": y})

    except Exception as e:
        logging.error(f"Error in Dividing Features and Labels: {e}")
        raise
    finally:
        mlflow.end_run()

def objective(trial, x , y):
    """
    Objective Function for Hyperparameter Tuning

    Parameters:
    - trail:
    - x: Features to train on 
    - y: Labels to evaluate against

    Returns:
    Loss Val: The loss value of the trail.
    """
    mlflow.start_run(run_name="Objective Function to run experiments on, used by optuna", nested=True)   
    try:
        # Define hyperparameters to be optimized
        units = trial.suggest_int('units', 32, 128)
        num_layers = trial.suggest_int('num_layers', 1, 3)
        dropout_rate = trial.suggest_float('dropout_rate', 0.1, 0.5)
        learning_rate = trial.suggest_float('learning_rate', 1e-4, 1e-2)
        batch_size = trial.suggest_int('batch_size', 32, 128)

        x_train, y_train = np.array(x[0]), np.array(y[0])
        x_eval, y_eval = np.array(x[1]), np.array(y[1])

        x_train = np.reshape(x_train, (x_train.shape[0], x_train.shape[1], 1))
        # y_train = np.reshape(y_train, (y_train.shape[0], y_train.shape[1], 1))
        x_eval = np.reshape(x_eval, (x_eval.shape[0], x_eval.shape[1], 1))
        # y_eval = np.reshape(x_train, (x_train.shape[0], x_train.shape[1], 1))

        # Create the model
        model = Model()
        model.hyperparameter_layers(units=units,num_layers=num_layers,dropout_rate=dropout_rate,input_shape=(x_train.shape[1], 1))

        # Compile the model
        model.get_model("hyperparameter").compile(optimizer=Adam(learning_rate=learning_rate), loss=MeanSquaredError())

        # Train the model
        early_stopping = EarlyStopping(monitor='val_loss', patience=3, restore_best_weights=True)
        model.get_model("hyperparameter").fit(x_train, y_train, validation_data=(x_eval, y_eval), epochs=1, batch_size=batch_size, callbacks=[early_stopping], verbose=0)

        # Evaluate the model
        val_loss = model.get_model("hyperparameter").evaluate(x_eval, y_eval, verbose=0)

    except Exception as e:
        logging.error(f"Error in Objective Function: {e}")
        raise
    finally:
        mlflow.end_run()
        return val_loss
    

def hyper_parameter_tuning(x,y):
    """
    Objective Function for Hyperparameter Tuning

    Parameters:
    - x: Features to train on 
    - y: Labels to evaluate against

    Returns:
    Loss Val: The loss value of the trail.
    """
    mlflow.start_run(run_name="Hyper-parameter Tuning", nested=True)   
    try:
        # Partial function to pass x and y to the objective
        objective_fn = partial(objective, x=x, y=y)

        # Define the study
        study = optuna.create_study(direction='minimize')
        study.optimize(objective_fn, n_trials=1)

        # Get the best trial
        best_trial = study.best_trial
        best_params = best_trial.params

    except Exception as e:
        logging.error(f"Error in Hyper Parameter Tuning: {e}")
        raise
    finally:
        mlflow.end_run()
        return best_params   
    # return {'units': 106, 'num_layers': 1, 'dropout_rate': 0.13736332505446322, 'learning_rate': 0.0008486320428172737, 'batch_size': 75}
    # return {'units': 96, 'num_layers': 1, 'dropout_rate': 0.2, 'batch_size': 64}


def training(best_params, x, y):
    """
   Train the model with the best hyperparameters

    Parameters:
    - best_params: Best parameters from Hyperparameter Tuning
    - x: Features to train on
    - y: Labels to evaluate against

    Returns:
    output_path: Saved model output path.
    """
    mlflow.start_run(run_name="training")  
    try:
        x_train, y_train = np.array(x[0]), np.array(y[0])
        x_train = np.reshape(x_train, (x_train.shape[0], x_train.shape[1], 1))
            
        # Create the model
        model = Model()
        model.create_training_layers(best_params=best_params,input_shape=(x_train.shape[1], 1))

        model.get_model("training").summary()

        # Compile the model
        model.get_model("training").compile(optimizer=Adam(learning_rate=best_params["learning_rate"]),loss=MeanSquaredError(), metrics=[metrics.MeanSquaredError(), metrics.AUC()])

        # Log parameters with MLflow
        mlflow.log_params(best_params)

        # Train the model
        # early_stopping = EarlyStopping(monitor='val_loss', patience=3, restore_best_weights=True)
        model.get_model("training").fit(x_train, y_train, epochs=1, batch_size=best_params["batch_size"], verbose=1)

        # Save the model with MLflow
        # mlflow.keras.log_model(model, "model")

        output_path = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), "model", 'trained_stock_prediction.h5')
        model.get_model("training").save(output_path)

    except Exception as e:
        logging.error(f"Error in Training: {e}")
        raise
    finally:
        mlflow.end_run()
        return output_path


def load_and_predict(x, file_path,ti):
    """
   Load and Predict from trained model

    Parameters:
    - x: Features to train on
    - file_path: Trained Model File

    Returns:
    predictions: Predictions on the test feature set.
    """
    mlflow.start_run(run_name="Load and Prediction")  
    try:

        # Load and predict
        model = load_model(file_path)

        x_test = np.array(x[2])
        x_test = np.reshape(x_test, (x_test.shape[0], x_test.shape[1], 1)) 
        scaler = ti.xcom_pull(task_ids='apply_transformation_training', key='scalar')

        # Upload Scaler file
        joblib.dump(scaler, './model/scaler.joblib')
        file_name = os.path.basename("./model/scaler.joblib")
        
        # Upload the plot to GCS
        destination_blob_name = 'model/archived/' + file_name
        blob = bucket.blob(destination_blob_name)
        blob.upload_from_filename('./model/scaler.joblib')

        predictions =  model.predict(x_test)
        predictions = scaler.inverse_transform(predictions)

    except Exception as e:
        logging.error(f"Error in Load and Prediction: {e}")
        raise
    finally:
        mlflow.end_run()
        return predictions

def evaluate_and_visualize(newly_trained_model_predictions, x, y, ti, save_path='/opt/airflow/visualizations/act.png'):
    """
   Evaluate and visualize the predictions 

    Parameters:
    - predictions: Predictions on the test feature set.
    - y: Label set

    Returns:
    predictions: Predictions on the test feature set.
    """
    mlflow.start_run(run_name="Evaluate and Visualize")  
    try:
        
        x_test = np.array(x[2])
        x_test = np.reshape(x_test, (x_test.shape[0], x_test.shape[1], 1)) 
        scaler = ti.xcom_pull(task_ids='apply_transformation_training', key='scalar')
        y_test_actual = np.array(y[2])       
        y_test_scaled = scaler.inverse_transform(y_test_actual.reshape(-1, 1))

        # Define your parameters
        source_blob_name = 'model/production/stock_prediction.h5'
        destination_file_name = './model/stock_prediction.h5'

        # Download the model from GCS
        blob = bucket.blob(source_blob_name)
        blob.download_to_filename(destination_file_name)
        
        # Load the models using MLflow
        gcs_model = load_model(destination_file_name)

        gcs_model_predictions = gcs_model.predict(x_test)
        gcs_model_predictions = scaler.inverse_transform(gcs_model_predictions)

        gcs_model_mse = mean_squared_error(y_test_scaled, gcs_model_predictions)
        newly_trained_model_mse = mean_squared_error(y_test_scaled, newly_trained_model_predictions)
        
        if gcs_model_mse > newly_trained_model_mse:
            logging.info("The newly trained model is better than the model in GCS")
            logging.info("Pushing model to Staging")


            file_name = os.path.basename("./model/trained_stock_prediction.h5")
        
            # Upload the plot to GCS
            destination_blob_name = 'model/staging/' + file_name
            blob = bucket.blob(destination_blob_name)
            blob.upload_from_filename("./model/trained_stock_prediction.h5")


            # Visualize the difference between actual vs predicted y-values
            plt.figure(figsize=(10, 6))
            plt.plot(y_test_scaled, label='Actual Values')
            plt.plot(newly_trained_model_predictions, label='Predicted Values')
            plt.title('Actual vs Predicted Values')
            plt.ylabel('Stock Price')
            plt.legend()
            plt.savefig(save_path)
            plt.show()

    except Exception as e:
        logging.error(f"Error in Evaluate and Visualize: {e}")
        raise
    finally:
        mlflow.end_run()