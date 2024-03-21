import pandas as pd
from skorecard.bucketers import EqualWidthBucketer

# Get data
df = pd.read_parquet('LGD_PS_Q_CF_SCORECARD_DATA_BINARY_TARGET_20240227.parquet')
time_series_field_name = 'SNAPSHOT_DT'
target_variable_field_name = 'LGD_TARGET'
field_to_bin_name = 'B1_PRMY_INDUSTRY_GROUP1_EN_NAME'
# Select only the columns that are needed
df = df[[time_series_field_name, target_variable_field_name, field_to_bin_name]]
# Update data types
df[time_series_field_name] = pd.to_datetime(df[time_series_field_name])
df[target_variable_field_name] = pd.to_numeric(df[target_variable_field_name], errors='coerce')
# Filter data by time
# df = df[(df[time_series_field_name] >= pd.to_datetime(start_date)) & (df[time_series_field_name] <= pd.to_datetime(end_date))]

ewb = EqualWidthBucketer(n_bins=5)
ewb.fit_transform(df)

ewb.bucket_table('column')
#>    bucket                       label  Count  Count (%)
#> 0      -1                     Missing    0.0        0.0
#> 1       0                (-inf, 19.8]   20.0       20.0
#> 2       1                (19.8, 39.6]   20.0       20.0
#> 3       2  (39.6, 59.400000000000006]   20.0       20.0
#> 4       3  (59.400000000000006, 79.2]   20.0       20.0
#> 5       4                 (79.2, inf]   20.0       20.0

# I get this output:
# (CVXPY) Mar 15 08:34:39 AM: Encountered unexpected exception importing solver GLOP:
# RuntimeError('Unrecognized new version of ortools (9.9.3963). Expected < 9.8.0. Please open a feature request on cvxpy to enable support for this version.')
# (CVXPY) Mar 15 08:34:39 AM: Encountered unexpected exception importing solver PDLP:
# RuntimeError('Unrecognized new version of ortools (9.9.3963). Expected < 9.8.0. Please open a feature request on cvxpy to enable support for this version.')
# Traceback (most recent call last):
#   File "dale_test.py", line 18, in <module>
#     ewb.fit_transform(df)
#   File "C:\Users\beblowd\.conda\envs\input_fin_los_mpm\lib\site-packages\sklearn\utils\_set_output.py", line 157, in wrapped
#     data_to_wrap = f(self, X, *args, **kwargs)
#   File "C:\Users\beblowd\.conda\envs\input_fin_los_mpm\lib\site-packages\sklearn\base.py", line 916, in fit_transform
#     return self.fit(X, **fit_params).transform(X)
#   File "C:\GitHub\skorecard\skorecard\bucketers\base_bucketer.py", line 270, in fit
#     splits, right = self._get_feature_splits(feature, X=X_flt, y=y_flt, X_unfiltered=X)
#   File "C:\GitHub\skorecard\skorecard\bucketers\bucketers.py", line 289, in _get_feature_splits
#     _, boundaries = np.histogram(X.values, bins=self.n_bins)
#   File "<__array_function__ internals>", line 180, in histogram
#   File "C:\Users\beblowd\.conda\envs\input_fin_los_mpm\lib\site-packages\numpy\lib\histograms.py", line 793, in histogram
#     bin_edges, uniform_bins = _get_bin_edges(a, bins, range, weights)
#   File "C:\Users\beblowd\.conda\envs\input_fin_los_mpm\lib\site-packages\numpy\lib\histograms.py", line 446, in _get_bin_edges
#     bin_edges = np.linspace(
#   File "<__array_function__ internals>", line 180, in linspace
#   File "C:\Users\beblowd\.conda\envs\input_fin_los_mpm\lib\site-packages\numpy\core\function_base.py", line 127, in linspace
#     start = asanyarray(start) * 1.0
# numpy.core._exceptions.UFuncTypeError: ufunc 'multiply' cannot use operands with types dtype('<M8[ns]') and dtype('float64')

# from skorecard import datasets
# from skorecard.bucketers import OptimalBucketer
# import numpy as np
# import pandas as pd

# # Get data
# df = pd.read_parquet('LGD_PS_Q_CF_SCORECARD_DATA_BINARY_TARGET_20240227.parquet')
# # print(df.columns)
# time_series_field_name = 'SNAPSHOT_DT'
# target_variable_field_name = 'LGD_TARGET'
# var1_name = 'CF_CRED_FAC_TO_SECURITY_RATE' # B1_PRMY_INDUSTRY_GROUP1_EN_NAME, CF_CRED_FAC_TO_SECURITY_RATE
# # var2_name = 'CF_CRED_FAC_TO_SECURITY_RATE'
# # field_to_bin_name = ['B1_PRMY_INDUSTRY_GROUP1_EN_NAME', 'CF_CRED_FAC_TO_SECURITY_RATE']
# # Select only the columns that are needed
# df = df[[time_series_field_name, target_variable_field_name, var1_name]]
# print(df.columns)
# # Update data types
# df[time_series_field_name] = pd.to_datetime(df[time_series_field_name])
# df[target_variable_field_name] = pd.to_numeric(df[target_variable_field_name], errors='coerce')
# # Filter data by time
# # df = df[(df[time_series_field_name] >= pd.to_datetime(start_date)) & (df[time_series_field_name] <= pd.to_datetime(end_date))]

# X = df[[var1_name]]
# print(type(X))
# y = np.array(df[target_variable_field_name].values, dtype='int8')
# bucketer = OptimalBucketer(variables = [var1_name])
# bucketer.fit_transform(X, y)

# Output from running:
# (CVXPY) Mar 15 08:33:25 AM: Encountered unexpected exception importing solver GLOP:
# RuntimeError('Unrecognized new version of ortools (9.9.3963). Expected < 9.8.0. Please open a feature request on cvxpy to enable support for this version.')
# (CVXPY) Mar 15 08:33:25 AM: Encountered unexpected exception importing solver PDLP:
# RuntimeError('Unrecognized new version of ortools (9.9.3963). Expected < 9.8.0. Please open a feature request on cvxpy to enable support for this version.')
# Index(['SNAPSHOT_DT', 'LGD_TARGET', 'CF_CRED_FAC_TO_SECURITY_RATE'], dtype='object')
# <class 'pandas.core.frame.DataFrame'>
# Traceback (most recent call last):
#   File "dale_test.py", line 55, in <module>
#     bucketer.fit_transform(X, y)
#   File "C:\Users\beblowd\.conda\envs\input_fin_los_mpm\lib\site-packages\sklearn\utils\_set_output.py", line 157, in wrapped
#     data_to_wrap = f(self, X, *args, **kwargs)
#   File "C:\Users\beblowd\.conda\envs\input_fin_los_mpm\lib\site-packages\sklearn\base.py", line 919, in fit_transform
#     return self.fit(X, y, **fit_params).transform(X)
#   File "C:\GitHub\skorecard\skorecard\bucketers\base_bucketer.py", line 270, in fit
#     splits, right = self._get_feature_splits(feature, X=X_flt, y=y_flt, X_unfiltered=X)
#   File "C:\GitHub\skorecard\skorecard\bucketers\bucketers.py", line 155, in _get_feature_splits
#     raise NotPreBucketedError(
# skorecard.utils.exceptions.NotPreBucketedError:
#                     OptimalBucketer requires numerical feature 'CF_CRED_FAC_TO_SECURITY_RATE' to be pre-bucketed
#                     to max 100 unique values (for performance reasons).
#                     Currently there are 7916 unique values present.

#                     Apply pre-binning, f.e. with skorecard.bucketers.DecisionTreeBucketer.