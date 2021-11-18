import pandas as pd
import numpy as np
from typing import Optional, Dict
import warnings

from sklearn.utils.validation import check_is_fitted
from sklearn.exceptions import NotFittedError
from sklearn.pipeline import Pipeline

from skorecard.bucket_mapping import BucketMapping
from skorecard.metrics.metrics import _IV_score


def build_bucket_table(
    X: pd.DataFrame,
    y: np.ndarray,
    column: str,
    bucketer=None,
    bucket_mapping: Optional[BucketMapping] = None,
    epsilon=0.00001,
    display_missing=True,
    verbose=False,
) -> pd.DataFrame:
    """
    Calculates summary statistics for a bucket generated by a skorecard bucketing object.

    This report currently works for just 1 column at a time.

    ``python
    from skorecard import datasets
    from skorecard.bucketers import DecisionTreeBucketer
    from skorecard.reporting import create_report
    X, y = datasets.load_uci_credit_card(return_X_y=True)

    # make sure that those cases
    specials = {
        "LIMIT_BAL":{
            "=50000":[50000],
            "in [20001,30000]":[20000,30000],
            }
    }

    dt_bucketer = DecisionTreeBucketer(variables=['LIMIT_BAL'], specials = specials)
    dt_bucketer.fit(X, y)
    dt_bucketer.transform(X)

    df_report = create_report(X,y,column="LIMIT_BAL", bucketer= dt_bucketer)
    df_report
    ```

    Args:
         X (pd.DataFrame): features
         y (np.array): target
         column (str): column for which you want the report
         bucketer: (optional) Skorecard.bucketing bucketer object. Ignored if bucket_mapping is specified.
         bucket_mapping: (optional) Skorecard.bucket_mapping BucketMapping object
         epsilon(float): small value to prevent zero division error for WoE
         display_missing (boolean): Add a row for missing even when not present in data
         verbose(boolean): be verbose

    Returns:
        df (pandas DataFrame): reporting df
    """
    assert column in X.columns

    X = X.copy()

    if bucket_mapping and bucketer:
        warnings.warn("Both bucket_mapping and bucketer specified. Ignoring bucketer.")

    if not bucket_mapping and not bucketer:
        raise Exception("Specify either bucket_mapping or bucketer")
        # TODO: In case no bucket_mapping and no bucketer specified,use values as-is

    if bucket_mapping and not bucketer:
        col_bucket_mapping = bucket_mapping

    if not bucket_mapping and bucketer:

        bucket_dict = bucketer.features_bucket_mapping_

        col_bucket_mapping = bucket_dict.get(column)

    X_transform = pd.DataFrame(data={"bucket_id": col_bucket_mapping.transform(X[column])}, index=X.index)
    
    if y is not None:
        X_transform["Event"] = y
    else:
        X_transform["Event"] = np.nan
    
    # If missing_treatment == passthrough, we reformat the bucket_id and un-do this later
    X_transform["bucket_id"].fillna(31415926535, inplace=True)

    stats = X_transform.groupby("bucket_id", as_index=False).agg(
        def_rate=pd.NamedAgg(column="Event", aggfunc="mean"),
        Event=pd.NamedAgg(column="Event", aggfunc="sum"),
        Count=pd.NamedAgg(column="bucket_id", aggfunc="count"),
    )

    stats["label"] = stats["bucket_id"].map(col_bucket_mapping.labels)
    # Make sure missing is present even when not present
    if display_missing:
        ref = pd.DataFrame.from_dict(col_bucket_mapping.labels, orient="index", columns=["label"])
        ref["bucket_id"] = ref.index
        stats = (
            stats.merge(ref, how="outer", on=["bucket_id", "label"])
            .fillna(0)
            .sort_values("bucket_id")
            .reset_index(drop=True)
        )

    stats["Count (%)"] = np.round(100 * stats["Count"] / stats["Count"].sum(), 2)

    # If unsupervised bucketer, we don't always have y info.
    if y is None:
        columns = ["bucket_id", "label", "Count", "Count (%)"]
        return stats.sort_values(by="bucket_id")[columns]

    stats["Non-event"] = stats["Count"] - stats["Event"]
    # Default rates
    stats["Event Rate"] = stats["Event"] / stats["Count"]  # TODO: can we divide by 0 accidentally?

    stats["% Event"] = stats["Event"] / stats["Event"].sum()
    stats["% Non-event"] = stats["Non-event"] / stats["Non-event"].sum()

    stats["WoE"] = ((stats["% Non-event"] + epsilon) / (stats["% Event"] + epsilon)).apply(lambda x: np.log(x))
    stats["IV"] = (stats["% Non-event"] - stats["% Event"]) * stats["WoE"]

    stats["WoE"] = np.round(stats["WoE"], 3)
    stats["IV"] = np.round(stats["IV"], 3)

    if verbose:
        iv_total = stats["IV"].sum()
        print(f"IV for {column} = {np.round(iv_total, 4)}")

    columns = [
        "bucket_id",
        "label",
        "Count",
        "Count (%)",
        "Non-event",
        "Event",
        "Event Rate",
        "WoE",
        "IV",
    ]

    # A little reformatting for if missing_treatment is passthrough
    if 31415926535 in stats["bucket_id"].values:
        stats["label"] = np.where(stats["bucket_id"] == 31415926535, "Missing", stats["label"])
        stats = stats.drop_duplicates(subset=["label"], keep="last").reset_index(drop=True).replace([31415926535], np.nan)
    return stats.sort_values(by="bucket_id")[columns]


class BucketTableMethod:
    """
    Add method for bucketing tables to another class.

    To be used with skorecard.pipeline.BucketingProcess and skorecard.bucketers.BaseBucketer
    """

    def bucket_table(self, column):
        """
        Generates the statistics for the buckets of a particular column.

        The pre-buckets are matched to the post-buckets, so that the user has a much clearer understanding of how
        the BucketingProcess ends up with the final buckets.
        An example:

        bucket     | label              | Count | Count (%) | Non-event | Event | Event Rate | WoE  |  IV
        -----------|--------------------|-------|-----------|-----------|-------|------------|------|-----
        0          | (-inf, 25000.0)    | 479.0 | 7.98      | 300.0     | 179.0 | 37.37      | 0.73 | 0.05
        1          | [25000.0, 45000.0) | 370.0 | 6.17      | 233.0     | 137.0 | 37.03      | 0.71 | 0.04

        Args:
            column: The column we wish to analyse

        Returns:
            df (pd.DataFrame): A pandas dataframe of the format above
        """  # noqa
        if isinstance(self, Pipeline):
            check_is_fitted(self.steps[0][1])
        else:
            check_is_fitted(self)

        if not hasattr(self, "bucket_tables_"):
            raise NotFittedError("You need to fit the bucketer on data in order to calculate the bucket table.")

        if column not in self.bucket_tables_.keys():
            raise ValueError(f"'{column}' is not one of the variables transformed by this bucketer.")

        table = self.bucket_tables_.get(column)
        table = table.rename(columns={"bucket_id": "bucket"})

        return table


class SummaryMethod:
    """
    Adds a `.summary()` method to a bucketing class.

    To be used with skorecard.pipeline.BucketingProcess and skorecard.bucketers.BaseBucketer
    """

    def _generate_summary(self, X, y):
        """
        Calculate the summary table.
        """
        if isinstance(self, Pipeline):
            check_is_fitted(self.steps[-1][1])
        else:
            check_is_fitted(self)

        self.summary_dict_ = {}

        # Calculate information value
        if y is not None:
            iv_scores = iv(self.transform(X), y)
        else:
            iv_scores = {}

        for col in X.columns:
            # In case the column was never (pre)-bucketed
            try:
                prebucket_number = str(len(self.prebucket_tables_[col]["bucket_id"].unique()))
            except KeyError:
                # This specific column was not prebucketed
                prebucket_number = "not_prebucketed"
            except AttributeError:
                # This bucketer did not do any prebucketing
                prebucket_number = "not_prebucketed"

            try:
                bucket_number = str(len(self.bucket_tables_[col]["bucket_id"].unique()))
            except KeyError:
                bucket_number = "not_bucketed"

            self.summary_dict_[col] = [
                prebucket_number,
                bucket_number,
                iv_scores.get(col, "not available"),
                X[col].dtype,
            ]

    def summary(self):
        """
        Display a summary table for columns passed to `.fit()`.

        The format is the following:

        column    | num_prebuckets | num_buckets | dtype
        ----------|----------------|-------------|-------
        LIMIT_BAL |      15        |     10      | float64
        BILL_AMT1 |      15        |     6       | float64
        """  # noqa
        if isinstance(self, Pipeline):
            check_is_fitted(self.steps[-1][1])
        else:
            check_is_fitted(self)

        if not hasattr(self, "summary_dict_"):
            raise NotFittedError("You need to fit Bucketer on data in order to calculate the summary table.")

        return (
            pd.DataFrame.from_dict(
                self.summary_dict_, orient="index", columns=["num_prebuckets", "num_buckets", "IV_score", "dtype"]
            )
            .rename_axis("column")
            .reset_index()
        )


def psi(X1: pd.DataFrame, X2: pd.DataFrame, epsilon=0.0001, digits=None) -> Dict:
    """
    Calculate the PSI between the features in two dataframes, `X1` and `X2`.

    `X1` and `X2` should be bucketed (outputs of fitted bucketers).

    $$
    PSI = \sum((\%{ Good } - \%{ Bad }) \times \ln \frac{\%{ Good }}{\%{ Bad }})
    $$

    Args:
        X1 (pd.DataFrame): bucketed features, expected
        X2 (pd.DataFrame): bucketed features, actual data
        epsilon (float): Amount to be added to relative counts in order to avoid division by zero in the WOE
            calculation.
        digits: (int): number of significant decimal digits in the IV calculation

    Returns: dictionary of psi values. keys are feature names, values are the psi values

    Examples:

    ```python
    from skorecard import datasets
    from sklearn.model_selection import train_test_split
    from skorecard.bucketers import DecisionTreeBucketer
    from skorecard.reporting import psi

    X, y = datasets.load_uci_credit_card(return_X_y=True)
    X_train, X_test, y_train, y_test = train_test_split(
        X,y,
        test_size=0.25,
        random_state=42
    )

    dbt = DecisionTreeBucketer()
    X_train_bins = dbt.fit_transform(X_train,y_train)
    X_test_bins = dbt.transform(X_test)

    psi_dict = psi(X_train_bins, X_test_bins)
    ```
    """  # noqa
    assert (X1.columns == X2.columns).all(), "X1 and X2 must have same columns"

    y1 = pd.Series(0, index=X1.index)
    y2 = pd.Series(1, index=X2.index)

    X = pd.concat([X1, X2], axis=0)
    y = pd.concat([y1, y2], axis=0).reset_index(drop=True)

    psis = {col: _IV_score(y, X[col], epsilon=epsilon, digits=digits) for col in X1.columns}

    return psis


def iv(X: pd.DataFrame, y: pd.Series, epsilon: float = 0.0001, digits: Optional[int] = None) -> Dict:
    """
    Calculate the Information Value (IV) of the features in `X`.

    `X` must be the output of fitted bucketers.

    $$
    IV = \sum { (\% goods - \% bads) } * { WOE }
    $$

    $$
    WOE=\ln (\% { goods } /  \% { bads })
    $$

    Example:

    ```python
    from skorecard import datasets
    from sklearn.model_selection import train_test_split
    from skorecard.bucketers import DecisionTreeBucketer
    from skorecard.reporting import iv

    X, y = datasets.load_uci_credit_card(return_X_y=True)

    dbt = DecisionTreeBucketer()
    X_bins = dbt.fit_transform(X,y)

    iv_dict = iv(X_bins, y)
    ```

    Args:
        X: pd.DataFrame (bucketed) features
        y: pd.Series: target values
        epsilon (float): Amount to be added to relative counts in order to avoid division by zero in the WOE
            calculation.
        digits (int): number of significant decimal digits in the IV calculation

    Returns:
        IVs (dict): Keys are feature names, values are the IV values
    """  # noqa
    return {col: _IV_score(y, X[col], epsilon=epsilon, digits=digits) for col in X.columns}
