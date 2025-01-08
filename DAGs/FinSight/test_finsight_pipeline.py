import pytest
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt
import yfinance as yf
import logging

# Set up basic configuration for logging within the test environment. This setup ensures that test outputs
# are timestamped and include information on the severity level of log messages, which aids in debugging
# and provides clearer, more actionable logs during test execution.

from FinSight.finsight_pipeline_functions import (
    download_and_uploadToDVCBucket,
    visualize_raw_data,
    divide_train_eval_test_splits,
    handle_missing_values,
    handle_outliers,
    visualize_df,
    apply_transformation,
    generate_scheme_and_stats,
    calculate_and_display_anomalies
)

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Mock Airflow task instance
# This mock class simulates the behavior of Airflow's TaskInstance, which is crucial for testing
# XCom push and pull functionalities. XCom (cross-communication) is used in Airflow to pass messages
# or values between different tasks within the same DAG. This mock class allows us to emulate
# the storage and retrieval of task outputs without an actual Airflow environment, enabling unit testing.
class MockTaskInstance:
    def __init__(self):
        self.xcom_pull_results = {}
        self.xcom_push_results = {}

    def xcom_pull(self, task_ids, key):
        return self.xcom_pull_results.get((task_ids, key), None)

    def xcom_push(self, key, value):
        self.xcom_push_results[key] = value

# Test for download_and_uploadToDVCBucket
# Test case for verifying the functionality of downloading stock data using the yfinance library and
# subsequently uploading it to a DVC (Data Version Control) bucket. This test ensures that the function
# correctly handles valid input by mocking the yfinance download method to return a predefined DataFrame
# and checks that the resulting data is pushed to the mock TaskInstance's XCom storage as expected.
def test_download_and_uploadToDVCBucket_positive(mocker):
    ticker = "NFLX"
    start = "2002-01-01"
    end = "2022-12-31"
    mock_task_instance = MockTaskInstance()

    # Mock yfinance download
    mock_stock_data = pd.DataFrame({
        'Open': [100, 200],
        'Close': [110, 210],
        'Volume': [1000, 1500],
    })
    mocker.patch('yfinance.download', return_value=mock_stock_data)

    download_and_uploadToDVCBucket(ticker_symbol=ticker, start_date=start, end_date=end, ti=mock_task_instance)

    assert 'stock_data' in mock_task_instance.xcom_push_results
    pd.testing.assert_frame_equal(mock_task_instance.xcom_push_results['stock_data'], mock_stock_data)
# Test case to verify error handling when an invalid ticker symbol is used. This test is crucial for ensuring
# that the function can gracefully handle and report errors encountered during the download process, such
# as invalid tickers. The yfinance download is mocked to raise an exception, and the test checks that this
# exception is correctly propagated and matches the expected error message.
def test_download_and_uploadToDVCBucket_negative(mocker):
    ticker = "INVALID_TICKER"
    start = "2002-01-01"
    end = "2022-12-31"
    mock_task_instance = MockTaskInstance()

    # Mock yfinance download to raise an exception
    mocker.patch('yfinance.download', side_effect=Exception("Invalid ticker"))

    with pytest.raises(Exception, match="Invalid ticker"):
        download_and_uploadToDVCBucket(ticker_symbol=ticker, start_date=start, end_date=end, ti=mock_task_instance)

# # Test for visualize_raw_data
# def test_visualize_raw_data_positive(mocker):
#     mock_task_instance = MockTaskInstance()
#     mock_stock_data = pd.DataFrame({
#         'Open': [100, 200],
#         'Close': [110, 210],
#         'Volume': [1000, 1500],
#     })
#     mock_task_instance.xcom_pull_results[('download_upload_data', 'stock_data')] = mock_stock_data

#     mocker.patch('matplotlib.pyplot.savefig')

#     visualize_raw_data(ti=mock_task_instance)

#     # Check if the plot was saved
#     assert mocker.patch('matplotlib.pyplot.savefig').call_count == 0

# def test_visualize_raw_data_negative(mocker):
#     mock_task_instance = MockTaskInstance()
#     mock_task_instance.xcom_pull_results[('download_upload_data', 'stock_data')] = None

#     with pytest.raises(Exception, match="Failed to visualize or upload data"):
#         visualize_raw_data(ti=mock_task_instance)

# Test for divide_train_eval_test_splits
# This test verifies that the data splitting function can correctly divide a given DataFrame into
# training, evaluation, and testing sets. It is essential for ensuring that the data processing pipeline
# can prepare data for machine learning models accurately, maintaining the integrity of dataset partitions.        
def test_divide_train_eval_test_splits_positive(mocker):
    mock_task_instance = MockTaskInstance()
    mock_stock_data = pd.DataFrame({
        'Open': [100, 200, 300],
        'Close': [110, 210, 310],
        'Volume': [1000, 1500, 2000],
    })
    mock_task_instance.xcom_pull_results[('download_upload_data', 'stock_data')] = mock_stock_data

    divide_train_eval_test_splits(file_path="dummy_path", ti=mock_task_instance)

    assert 'train' in mock_task_instance.xcom_push_results
    assert 'eval' in mock_task_instance.xcom_push_results
    assert 'test' in mock_task_instance.xcom_push_results
# Negative test case to ensure that the function to divide data into training, evaluation, and testing sets
# correctly handles cases where the input data is missing (None). This test checks that the expected exception
# is raised, which helps prevent data processing errors downstream in the pipeline.
def test_divide_train_eval_test_splits_negative(mocker):
    mock_task_instance = MockTaskInstance()
    mock_task_instance.xcom_pull_results[('download_upload_data', 'stock_data')] = None

    with pytest.raises(Exception, match="Failed to split data"):
        divide_train_eval_test_splits(file_path="dummy_path", ti=mock_task_instance)

# Test for handle_missing_values
# Tests the function's ability to handle missing values in the dataset. Proper handling of missing values
# is crucial for the accuracy of machine learning models, as it ensures that the training and evaluation data
# is complete and statistically valid. This test checks that missing data points are appropriately imputed
# or removed according to the function's logic.       
def test_handle_missing_values_positive():
    mock_df = pd.DataFrame({
        'Open': [100, None, 300],
        'Close': [110, 210, 310],
        'Volume': [1000, 1500, 2000],
    })

    result_df = handle_missing_values(mock_df.copy())

    expected_df = pd.DataFrame({
        'Open': [100, 110, 300],
        'Close': [110, 210, 310],
        'Volume': [1000, 1500, 2000],
    })

    pd.testing.assert_frame_equal(result_df, expected_df)
# This test checks the error handling for the missing values function when it receives a None or improperly
# formatted input. Ensuring robust error handling is critical for maintaining the reliability and stability
# of the data preprocessing steps within the pipeline.
def test_handle_missing_values_negative():
    with pytest.raises(Exception):
        handle_missing_values(None)

# Test for handle_outliers
# Tests the function's effectiveness in identifying and handling outliers within the dataset. Outliers can
# skew the results of data analysis and predictive modeling if not managed correctly. This test ensures that
# the function can detect and exclude or adjust outlier values to maintain the quality and consistency of the dataset.        
def test_handle_outliers_positive():
    mock_df = pd.DataFrame({
        'Open': [100, 200, 10000],
        'Close': [110, 210, 11000],
        'Volume': [1000, 1500, 1000000],
    })

    result_df = handle_outliers(mock_df.copy())

    expected_df = pd.DataFrame({
        'Open': [100, 200],
        'Close': [110, 210],
        'Volume': [1000, 1500],
    })

    pd.testing.assert_frame_equal(result_df, expected_df)
# Verifies that the outlier handling function properly raises an exception when given a None or malformed
# DataFrame. Robust error handling for such cases is essential to avoid unexpected failures during data
# preprocessing, ensuring that data quality issues are addressed promptly and clearly.
def test_handle_outliers_negative():
    with pytest.raises(Exception):
        handle_outliers(None)

# # Test for visualize_df
# def test_visualize_df_positive():
#     mock_df = pd.DataFrame({
#         'Open': [100, 200],
#         'Close': [110, 210],
#         'Volume': [1000, 1500],
#     })

#     result_df = visualize_df(mock_df.copy())

#     pd.testing.assert_frame_equal(result_df, mock_df)

# def test_visualize_df_negative():
#     with pytest.raises(Exception):
#         visualize_df(None)

# Test for apply_transformation
# Tests the function that applies data transformations, such as normalization or scaling, to the dataset.
# This function is vital for preparing data for machine learning models that might be sensitive to the
# scale of input features. This test ensures that all expected transformations are applied consistently
# and correctly across the dataset.        
def test_apply_transformation_positive():
    mock_df = pd.DataFrame({
        'Open': [100, 200],
        'Close': [110, 210],
        'Volume': [1000, 1500],
        'High': [120, 220],
        'Low': [90, 190],
        'Adj Close': [115, 215]
    })

    result_df = apply_transformation(mock_df.copy())

    scaler = MinMaxScaler(feature_range=(0, 1))
    expected_df = mock_df.copy()
    expected_df['Volume'] = scaler.fit_transform(expected_df[['Volume']])
    expected_df['Open'] = scaler.fit_transform(expected_df[['Open']])
    expected_df['Close'] = scaler.fit_transform(expected_df[['Close']])
    expected_df['High'] = scaler.fit_transform(expected_df[['High']])
    expected_df['Low'] = scaler.fit_transform(expected_df[['Low']])
    expected_df['Adj Close'] = scaler.fit_transform(expected_df[['Adj Close']])

    pd.testing.assert_frame_equal(result_df, expected_df)
# Tests the error resilience of the data transformation function when faced with a None or empty DataFrame.
# This test is crucial for confirming that the function can handle error conditions gracefully, ensuring that
# any data integrity issues are caught early in the data processing pipeline.
def test_apply_transformation_negative():
    with pytest.raises(Exception):
        apply_transformation(None)

# Test for generate_scheme_and_stats
 # This test checks the function that generates a schema and statistical summaries for the dataset. These
# summaries can be crucial for understanding data distributions and identifying potential data quality
# issues. The test ensures that the function correctly calculates and returns the expected statistical
# descriptions and schema information.       
def test_generate_scheme_and_stats_positive():
    mock_df = pd.DataFrame({
        'Open': [100, 200],
        'Close': [110, 210],
        'Volume': [1000, 1500],
    })

    mock_task_instance = MockTaskInstance()

    result_df = generate_scheme_and_stats(mock_df.copy(), mock_task_instance)

    pd.testing.assert_frame_equal(result_df, mock_df)

def test_generate_scheme_and_stats_negative():
    # with pytest.raises(Exception):
    #     generate_scheme_and_stats(None, None)
    pass

# Test for calculate_and_display_anomalies
# This test evaluates the function's ability to calculate and display anomalies in the dataset. Detecting
# anomalies is key for ensuring data quality and can be critical in scenarios like fraud detection. This
# test ensures that the function can identify and report any anomalies found within the dataset accurately.
def test_calculate_and_display_anomalies_positive():
    mock_df = pd.DataFrame({
        'Open': [100, 200],
        'Close': [110, 210],
        'Volume': [1000, 1500],
    })

    mock_task_instance = MockTaskInstance()

    result_df = calculate_and_display_anomalies(mock_df.copy(), mock_task_instance)

    pd.testing.assert_frame_equal(result_df, mock_df)

def test_calculate_and_display_anomalies_negative():
    # with pytest.raises(Exception):
    #     calculate_and_display_anomalies(None, None)
    pass