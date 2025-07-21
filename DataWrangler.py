import math
from collections import Counter
from typing import Any, Callable, Optional

import numpy as np
import pandas as pd
from pandas import DataFrame
import seaborn as sns
import matplotlib.pyplot as plt
import plotly.express as px
import scipy.stats as stats

class Wrangler(pd.DataFrame):
    """ A custom DataFrame class with additional attributes. """

    # List of metadata attributes to be preserved when DataFrame methods create new objects.
    _metadata: list[str] = ['my_attr']

    @property
    def _constructor(self) -> Callable[..., "Wrangler"]:
        """
        Ensures that operations on Wrangler objects (like df[mask], df.copy()) 
        return new Wrangler instances (not just plain pd.DataFrames), 
        preserving any custom behavior and attributes.
        """
        def _c(*args: Any, **kwargs: Any) -> "Wrangler":
            # Creates a new Wrangler instance and copies metadata from self.
            return Wrangler(*args, **kwargs).__finalize__(self)
        return _c

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        """
        Initializes a new Wrangler object. 
        - Extracts the custom 'my_attr' from kwargs if present, and sets it as an instance attribute.
        - Initializes the underlying DataFrame with the remaining args and kwargs.
        """
        # Pop the 'my_attr' keyword argument if provided.
        self.my_attr: Optional[Any] = kwargs.pop('my_attr', None)
        super().__init__(*args, **kwargs)

    # Dataframe Not Empty Validation
    def _check_not_empty(self: pd.DataFrame) -> None:
        if self.empty:
            raise ValueError("Dateframe is empty and cannot be analyzed. Please supply data before calling this method.")
    
    # Check if a column is present
    def _check_valid_column(self: pd.DataFrame, column: str) -> None:
        if column not in self.columns:
            raise KeyError(f"Column '{column}' not found in the DataFrame.")
    
    # Check if column has a categorical datatype (i.e., object, category, boolean)    
    def _check_cat_column(self: pd.DataFrame, column: str) -> None:
        if self[column].dtype not in ['object', 'category', 'bool']:
            raise ValueError(f"Column '{column}' is not a categorical datatype.")

    # Check if column has a numerical datatype (i.e., integer, floating-point)    
    def _check_num_column(self: pd.DataFrame, column: str) -> None:
        if self[column].dtype not in ['int64', 'float64']:
            raise ValueError(f"Column '{column}' is not a numerical data type (int64, float64).")

    # Check for nulls in column    
    def _check_nulls(self: pd.DataFrame, column: str):
        if self[column].isnull().any():
            raise ValueError(f"Column '{column}' contains null values. Please impute or drop null values.")

    # Strip leading and trailing white spaces from all column headers and values
    def strip_dataframe(self: pd.DataFrame) -> None:
        self._check_not_empty()
        self.columns = self.columns.str.strip()

        for col in self.select_dtypes(include=['object', 'category']).columns:
            self[col] = self[col].str.strip()

    # Summary statistics 
    def dataframe_analysis(self: pd.DataFrame) -> None:
        self._check_not_empty()

        print("Descriptive Statistics:")
        print(self.describe(include=['object', 'float', 'int', 'category', 'bool']).T)
        print("-" * 60)

        print("\nNumber of nulls in each column:")
        print(self.isnull().sum())
        print("-" * 60)

        print("\nDuplicate Rows:")
        print(self.duplicated().sum())

    # Identify numerical and categorical columns
    def identify_columns(self: pd.DataFrame) -> None:
        self._check_not_empty()

        num_cols: list[str] = [col for col in self.columns if self[col].dtypes in ['int64', 'float64']]
        cat_cols: list[str] = [col for col in self.columns if self[col].dtypes not in ['int64', 'float64']]

        print(f"Numerical columns: {num_cols}.")
        print(f"Categorical columns: {cat_cols}.")

    # This method identifies numerical and categorical columns
    def column_analysis(self: pd.DataFrame) -> None:
        self._check_not_empty()

        category_columns: list[str] = [col for col in self.columns if self[col].dtype in ['object', 'category', 'bool']]

        numerical_but_categorical: list[str] = [
            col for col in self.columns if self[col].nunique() < 10 and self[col].dtype in ['int64', 'float64']
        ]

        category_with_hi_cardinality: list[str] = [
            col for col in self.columns if self[col].nunique() > 50 and self[col].dtype in ['category', 'object']
        ]

        # Filter out numerical_but_categorical from categorical_columns
        categorical_columns: list[str] = [col for col in category_columns if col not in numerical_but_categorical]

        # Identify purely numerical columns
        numerical_columns: list[str] = [col for col in self.columns if self[col].dtype in ['int64', 'float64']]
        numerical_columns: list[str] = [col for col in numerical_columns if col not in category_columns]

        # Print analysis
        print(f'Observations : {self.shape[0]}')
        print(f'Variables : {self.shape[1]}')
        print(f'Categorical Columns : {len(category_columns)}')
        print(f'Numerical Columns : {len(numerical_columns)}')
        print(f'Categorical Columns with High Cardinality : {len(category_with_hi_cardinality)}')
        print(f'Numerical Columns that are Categorical: {len(numerical_but_categorical)}')

        return category_columns, numerical_columns, category_with_hi_cardinality

    # Summary of categorical column
    def categorical_column_summary(self: pd.DataFrame, column: str, plot: bool = False, warn_on_nulls: bool = True) -> None:
        self._check_not_empty()
        self._check_valid_column(column)
        self._check_cat_column(column)
        
        if warn_on_nulls:
            n_null_values = self[column].isnull().sum()
            if n_null_values > 0:
                print(f"Warning: '{column}' has {n_null_values} null(s). Nulls are excluded in calculation.\n")

        # Calculate value counts and ratios
        value_counts = self[column].value_counts()
        ratios = round(100 * value_counts / len(self), 2)

        summary_df = pd.DataFrame({column: value_counts, 'Ratio (%)': ratios})

        print(summary_df)
        print('-' * 40)

        if plot:
            if self[column].dtype == 'bool':
                sns.countplot(x=self[column].astype(int), data=self)
            else:
                sns.countplot(x=self[column], data=self)

            plt.show(block=True)

    # Summary of numerical column
    def numerical_column_summary(self: pd.DataFrame, column: str, plot: bool = False, warn_on_nulls: bool = True) -> None:
        self._check_not_empty()
        self._check_valid_column(column)
        self._check_num_column(column)
        
        if warn_on_nulls:
            n_null_values = self[column].isnull().sum()
            if n_null_values > 0:
                print(f"Warning: '{column}' has {n_null_values} null(s). Nulls are excluded in calculation.\n")
        

        quantiles = [0.05, 0.10, 0.20, 0.30, 0.40, 0.50, 0.60, 0.70, 0.80, 0.90, 0.95, 0.99]
        summary_df = self[column].describe(quantiles).to_frame()
        print(summary_df)

        if plot:
            self[column].hist()
            plt.xlabel(column)
            plt.title(column)
            plt.show()
        print('-' * 40)

    # Cross examines the relationship between categorical columns and target column that is numerical
    def target_cross_analysis_cat(self: pd.DataFrame, target: str, cat_col: str, warn_on_nulls: bool = True) -> None:
        self._check_not_empty()
        self._check_valid_column(target)
        self._check_valid_column(cat_col)
        self._check_num_column(target)
        self._check_cat_column(cat_col)
        
        if warn_on_nulls:
            n_null_target = self[target].isnull().sum()
            n_null_cat = self[cat_col].isnull().sum()
            if n_null_target > 0 or n_null_cat > 0:
                print(f"Warning: '{target}' has {n_null_target} nulls, '{cat_col}' has {n_null_cat} nulls. Nulls in '{target}' are ignored in the mean. Nulls in '{cat_col}' will be omitted.")
        
        result = self.groupby(cat_col, observed=False)[target].mean().sort_values(ascending=False)
        print(pd.DataFrame({'TARGET MEAN': result}))

    # Cross examines the relationship between numerical columns and target column regardless of target datatype
    def target_cross_analysis_num(self: pd.DataFrame, target: str, num_col: str, warn_on_nulls: bool = True) -> None:
        self._check_not_empty()
        self._check_valid_column(target)
        self._check_valid_column(num_col)
        self._check_num_column(num_col)
        
        if warn_on_nulls:
            n_null_target = self[target].isnull().sum()
            n_null_num = self[num_col].isnull().sum()
            if n_null_target > 0 or n_null_num > 0:
                print(
                    f"Warning: '{target}' has {n_null_target} nulls, '{num_col}' has {n_null_num} nulls. "
                    "Nulls in '{num_col}' are ignored in the mean. Nulls in '{target}' will form their own group."
                )
        
        result = self.groupby(target, observed=False)[num_col].mean().sort_index(ascending=False)
        print(result)

    # Normalization of numerical column
    def normalize(self: pd.DataFrame, column: str, warn_on_nulls: bool = True) -> None:
        self._check_not_empty()
        self._check_valid_column(column)
        self._check_num_column(column)
        
        if warn_on_nulls:
            n_null_col = self[column].isnull().sum()
            if n_null_col > 0:
                print(f"Warning: {column} column has {n_null_col} null value(s). Normalization complete, nulls remain in column")

        self[column] = (self[column] - self[column].mean()) / self[column].std()

    # Imputate missing all missing data.
    def complete_imputation(self: pd.DataFrame) -> None:
        self._check_not_empty()
        for column in self.columns:
            if self[column].dtype in ['int64', 'float64'] and self[column].isna().any():
                if self[column].isna().all():
                    raise ValueError(f"Column '{column}' contains only null values. Imputation with mean is not possible.")
                self[column] = self[column].fillna(self[column].mean())
            elif self[column].dtype in ['object', 'category', 'bool'] and self[column].isna().any():
                # Handle columns where all values are null, so mode() returns empty
                if self[column].mode().empty:
                    raise ValueError(f"Column '{column}' contains only null values. Imputation with mode is not possible.")
                self[column] = self[column].fillna(self[column].mode().iloc[0])

    # Identify and remove outilier in column using the interquartile range method (IQR)
    def remove_outlier_iqr(self: pd.DataFrame, column: str) -> None:
        self._check_not_empty()
        self._check_valid_column(column)
        self._check_num_column(column)
        
        column_data = self[column].dropna()

        if column_data.empty:
            raise ValueError(f"Column '{column}' contains only null values. Cannot compute IQR outliers.")

        Q1 = np.percentile(column_data, 25, method='midpoint')
        Q3 = np.percentile(column_data, 75, method='midpoint')
        IQR = Q3 - Q1

        upper_bound = Q3 + 1.5 * IQR
        lower_bound = Q1 - 1.5 * IQR

        # Identify the outliers
        outliers = self[(self[column] < lower_bound) | (self[column] > upper_bound)]

        # Remove the outliers
        self.drop(outliers.index, inplace=True)

    # Produce the upper and lower bound rows of a DataFrame using the IQR method for a given column
    def outlier_limits_iqr(self: pd.DataFrame, column: str) -> pd.DataFrame:
        self._check_not_empty()
        self._check_valid_column(column)
        self._check_num_column(column)

        column_data = self[column].dropna()
        if column_data.empty:
            raise ValueError(f"Column '{column}' contains only null values. Cannot compute IQR outliers.")

        Q1, Q3 = column_data.quantile([0.25, 0.75])
        IQR = Q3 - Q1
        upper_limit = Q3 + 1.5 * IQR
        lower_limit = Q1 - 1.5 * IQR

        return self[(self[column] < lower_limit) | (self[column] > upper_limit)]

    # Displays all outliers in the DataFrame using the IQR method
    def show_outlier_rows(self: pd.DataFrame) -> None:
        self._check_not_empty()
        num_cols: list[str] = [col for col in self.columns if self[col].dtype in ['int64', 'float64']]

        if not num_cols:
            print("No numerical columns in the DataFrame.")
            return

        for col in num_cols:
            print('-' * 40, col, '-' * 40)
            outliers = self.outlier_limits_iqr(col)
            if outliers.empty:
                print("No outliers detected.")
            else:
                print(outliers)

    # Type casts an object datatype into a category datatype
    def category_datatype(self: pd.DataFrame) -> None:
        self._check_not_empty()
        obj_cols = self.select_dtypes(include=['object']).columns
        if not obj_cols.any():
            # No object columns to convert
            return
        self[obj_cols] = self[obj_cols].astype('category')

    # Replace values in a DataFrame that represent an unknown value but are not recorded as null (e.g. -, ?, *)
    def turn_null(self: pd.DataFrame, val: int | float | str) -> None:
        self._check_not_empty()
        self[self.columns] = self.apply(lambda col: col.replace({val: np.nan}))

    # Output the percentage of null values in each column of the DataFrame
    def null_percent(self: pd.DataFrame) -> None:
        self._check_not_empty()
        print(self.isnull().mean().round(4).mul(100).sort_values(ascending=False))

    # Drop columns in a DataFrame with a certain percentage of null values
    def drop_null_by_percent(self: pd.DataFrame, percent: int | float) -> None:
        self._check_not_empty()
        if not (0 <= percent <= 100):
            raise ValueError("Parameter 'percent' must be between 0 and 100.")
        min_count = int(((100 - percent) / 100) * self.shape[0] + 1)
        self.dropna(axis=1, thresh=min_count, inplace=True)

    # Type casts an object datatype into a boolean datatype
    def bool_datatype(self: pd.DataFrame, column: str, true_value: int | float | str, false_value: int | float | str) -> None:
        """
        This method type casts an object datatype into a boolean datatype.

        Parameters:
        - column (str): The column name in the DataFrame.
        - true_value: The value in the column to be considered as True.
        - false_value: The value in the column to be considered as False.
        """
        self._check_not_empty()
        self._check_valid_column(column)
        
        unique_values = set(self[column].unique())
        expected_values = {true_value, false_value}
        if not unique_values.issubset(expected_values):
            raise ValueError(f"Column '{column}' contains values other than the specified true/false values ({expected_values}).")
        encoded_values = {true_value: True, false_value: False}
        self[column] = self[column].map(encoded_values).astype(bool)

    # Print a dictionary with the unique values of a column and the number of occurrences
    def counter(self: pd.DataFrame, column: str) -> None:
        self._check_not_empty()
        self._check_valid_column(column)
        
        counts = dict(Counter(self[column]))
        print(counts)

""" New class for graphing values of dataset """
class Graphs:
    def __init__(self, df: DataFrame, style: str = 'ggplot') -> None:
        self.df = df
        self.style = style

    def _validate_not_empty(self) -> None:
        if self.df.empty:
            raise ValueError("Dateframe is empty and cannot be graphed. Please supply data before calling this method.")
        
    def _check_column_present(self, column: str) -> None:
        if column not in self.df.columns:
            raise KeyError(f'Column "{column}" not found in DataFrame')
        
    def _check_no_nulls(self, column: str) -> None:
        if self.df[column].isnull().any():
            raise ValueError(f'Column "{column}" contains null values')
        
    def _check_categorical_column(self, column: str) -> None:
        if self.df[column].dtype not in ['object', 'category', 'bool']:
            raise ValueError(f"Column '{column}' is not a categorical datatype.")
        
    def _check_numerical_column(self, column: str) -> None:
        if self.df[column].dtype not in ['int64', 'float64']:
            raise ValueError(f"Column '{column}' not a numerical data type.")

    """ Single Visualization Graphs """
    def histogram(self, column: str) -> None:
        # Error Handling
        self._validate_not_empty()
        self._check_column_present(column)
        self._check_no_nulls(column)
        
        # plotting
        with plt.style.context(self.style):
            fig, ax = plt.subplots(figsize=(17, 8))
            sns.histplot(data=self.df, x=self.df[column], kde=True, ax=ax)
            plt.grid(False)
            ax.set_title(f"{column.title().replace('_', ' ')} Histogram")

        plt.show()

    def categorical_boxplot(self, categorical_column: str, numerical_column: str) -> None:
        # Error Handling
        self._validate_not_empty()
        self._check_column_present(categorical_column)
        self._check_column_present(numerical_column)
        self._check_no_nulls(categorical_column)
        self._check_no_nulls(numerical_column)
        self._check_categorical_column(categorical_column)
        self._check_numerical_column(numerical_column)

        # Plotting
        with plt.style.context(self.style):
            fig, ax = plt.subplots(figsize=(17, 8))
            sns.boxplot(data=self.df, x=categorical_column, y=numerical_column, ax=ax)
            ax.set_title(f"{numerical_column.title().replace('_', ' ')} by {categorical_column.title().replace('_', ' ')} Boxplot")

            medians = self.df.groupby(categorical_column, observed=False)[numerical_column].median()
            obs = self.df[categorical_column].value_counts().reindex(medians.index).values

            pos = range(len(obs))
            for tick, label in zip(pos, ax.get_xticklabels()):
                ax.text(pos[tick], medians.iloc[tick] + 0.03, f'n: {obs[tick]}',
                        ha='center', size='large', color='black', weight='semibold')

            ax.grid(False)

        plt.show()

    def cate_boxplot(
        self,
        categorical_column: str,
        numerical_column: str,
        hue_column: str = None
    ) -> None:
        self._validate_not_empty()
        self._check_column_present(categorical_column)
        self._check_column_present(numerical_column)
        self._check_no_nulls(categorical_column)
        self._check_no_nulls(numerical_column)
        self._check_categorical_column(categorical_column)
        self._check_numerical_column(numerical_column)
        if hue_column:
            self._check_column_present(hue_column)
            self._check_no_nulls(hue_column)
            self._check_categorical_column(hue_column)

        with plt.style.context(self.style):
            fig, ax = plt.subplots(figsize=(17, 8))
            sns.boxplot(
                data=self.df,
                x=categorical_column,
                y=numerical_column,
                hue=hue_column,
                showmeans=False,
                ax=ax
            )

            # Plot title
            title = f"{numerical_column.title().replace('_', ' ')} by {categorical_column.title().replace('_', ' ')} Boxplot"
            if hue_column:
                title += f" with {hue_column.title().replace('_', ' ')} Grouping"
            ax.set_title(title)
            ax.grid(False)

            # --- Add sample sizes above each box ---
            if not hue_column:
                counts = self.df[categorical_column].value_counts().sort_index()
                medians = self.df.groupby(categorical_column, observed=False)[numerical_column].median()
                for i, label in enumerate(ax.get_xticklabels()):
                    n = counts.iloc[i]
                    y = medians.iloc[i]
                    ax.text(
                        i, y + 0.03 * (self.df[numerical_column].max() - self.df[numerical_column].min()),
                        f'n: {n}',
                        ha='center', va='bottom', fontsize=12, color='black', fontweight='semibold'
                    )
            else:
                # Get the order as used by seaborn for ticks/hues
                cat_order = [tick.get_text() for tick in ax.get_xticklabels()]
                hue_order = [t.get_text() for t in ax.legend_.texts]

                # Group counts
                counts = (
                    self.df.groupby([categorical_column, hue_column], observed=False)
                    .size()
                    .unstack(fill_value=0)
                    .reindex(index=cat_order, columns=hue_order)
                )
                medians = (
                    self.df.groupby([categorical_column, hue_column], observed=False)[numerical_column]
                    .median()
                    .unstack()
                    .reindex(index=cat_order, columns=hue_order)
                )

                # Position annotation above each box:
                n_hue = len(hue_order)
                width = 0.8  # default seaborn width
                for i, cat in enumerate(cat_order):
                    for j, hue in enumerate(hue_order):
                        n = counts.loc[cat, hue]
                        y = medians.loc[cat, hue]
                        # Compute the x position for this box:
                        x = i - width / 2 + (j + 0.5) * width / n_hue
                        ax.text(
                            x, y + 0.03 * (self.df[numerical_column].max() - self.df[numerical_column].min()),
                            f'n: {n}',
                            ha='center', va='bottom', fontsize=11, color='black', fontweight='semibold'
                        )

            # Move legend out of plot for clarity if hue present
            if hue_column:
                ax.legend(loc='upper left', bbox_to_anchor=(1, 1))

            plt.tight_layout()
            plt.show()

    def categorical_boxplot_with_hue(
            self,
            categorical_column: str,
            numerical_column: str, 
            hue_column: str
    ) -> None:
        # Error Handling
        self._validate_not_empty()
        self._check_column_present(categorical_column)
        self._check_column_present(numerical_column)
        self._check_column_present(hue_column)
        self._check_no_nulls(categorical_column)
        self._check_no_nulls(numerical_column)
        self._check_no_nulls(hue_column)
        self._check_categorical_column(categorical_column)
        self._check_numerical_column(numerical_column)
        self._check_categorical_column(hue_column)

        # Plotting
        with plt.style.context(self.style):
            fig, ax = plt.subplots(figsize=(17, 8))
            sns.boxplot(data=self.df, x=categorical_column, y=numerical_column, hue=hue_column, showmeans=True, ax=ax)
            ax.set_title(f"{numerical_column.title().replace('_', ' ')} by {categorical_column.title().replace('_', ' ')} with {hue_column.title().replace('_', ' ')} Grouping Boxplot")
            plt.grid(False)

        plt.show()

    def categorical_barplot(self, cat_column: str, num_column: str, hue_col: Optional[str] = None, limit: Optional[int] = None) -> None:
        # Error Handling
        self._validate_not_empty()
        self._check_column_present(cat_column)
        self._check_column_present(num_column)
        self._check_no_nulls(cat_column)
        self._check_no_nulls(num_column)
        self._check_categorical_column(cat_column)
        self._check_numerical_column(num_column)
        if hue_col:
            self._check_column_present(hue_col)
            self._check_categorical_column(hue_col)
            self._check_no_nulls(hue_col)

        if limit is not None and not isinstance(limit, int):
            raise ValueError(f'limit should be an integer, got {type(limit)}')

        # Plotting
        with plt.style.context(self.style):
            fig, ax = plt.subplots(figsize=(17, 8))
            if limit is not None:
                order = self.df.groupby(cat_column).mean(numeric_only=True).sort_values(num_column, ascending=False).iloc[:limit].index
            else:
                order = self.df.groupby(cat_column, observed=False).mean(numeric_only=True).sort_values(num_column, ascending=False).index
            sns.barplot(data=self.df, x=cat_column, y=num_column, hue=hue_col, order=order, err_kws={'linewidth': 0}, ax=ax)

            ax.set(ylabel=None)
            ax.grid(False)
            if limit is not None:
                title = f"Average {num_column.replace('_', ' ').title()} by {cat_column.replace('_', ' ').title()} Barplot [Top {limit}]"
            else:
                title = f"Average {num_column.replace('_', ' ').title()} by {cat_column.replace('_', ' ').title()} Barplot"
            if hue_col:
                title += f" with {hue_col.replace('_', ' ').title()} Grouping"
            ax.set_title(title)

            for p in ax.patches:
                count = int(p.get_height())
                if count == 0:
                    continue # skip annotating zero-height (empty) bars
                ax.annotate(format(p.get_height(), '.1f'),
                            (p.get_x() + p.get_width() / 2, p.get_height()),
                            ha='center', va='center',
                            size=15, xytext=(0, 8),
                            textcoords='offset points')

        plt.yticks([])
        plt.show()

    def scatterplot(self, num_col1: str, num_col2: str, hue_col: Optional[str] = None) -> None:
        # Error Handling
        self._validate_not_empty()
        self._check_column_present(num_col1)
        self._check_column_present(num_col2)
        self._check_numerical_column(num_col1)
        self._check_numerical_column(num_col2)
        self._check_no_nulls(num_col1)
        self._check_no_nulls(num_col2)
        if hue_col:
            self._check_column_present(hue_col)
            self._check_categorical_column(hue_col)
            self._check_no_nulls(hue_col)
        
        # Plotting
        with plt.style.context(self.style):
            fig, ax = plt.subplots(figsize=(17, 8))
            sns.scatterplot(data=self.df, x=num_col1, y=num_col2, hue=hue_col, ax=ax)
            title = f"{num_col1.replace('_', ' ').title()} vs {num_col2.replace('_', ' ').title()} Scatterplot"
            if hue_col:
                title += f" with {hue_col.replace('_', ' ').title()} Grouping"
            ax.set_title(title)
            plt.grid(False)

        plt.show()

    # This method returns seaborn jointplot with regression line
    def jointplot(self, num_col1: str, num_col2: str) -> None:
        # Error Handling
        self._validate_not_empty()
        self._check_column_present(num_col1)
        self._check_column_present(num_col2)
        self._check_no_nulls(num_col1)
        self._check_no_nulls(num_col2)
        self._check_numerical_column(num_col1)
        self._check_numerical_column(num_col2)

        # Plotting
        with plt.style.context(self.style):
            g = sns.jointplot(data=self.df, x=num_col1, y=num_col2, kind='reg')

            # Calculate and display Pearson correlation coefficient
            if len(set(self.df[num_col1])) > 1 and len(set(self.df[num_col2])) > 1:
                r, p = stats.pearsonr(self.df[num_col1].values, self.df[num_col2].values)
                annotation_text = f'$\\rho = {r:.3f}$'
                g.ax_joint.annotate(annotation_text, xy=(0.1, 0.9), xycoords='axes fraction',
                                    ha='left', va='center', bbox={'boxstyle': 'round'}, fontsize=20)
            else:
                annotation_text = 'No variability in data'
                g.ax_joint.annotate(annotation_text, xy=(0.1, 0.9), xycoords='axes fraction',
                                    ha='left', va='center', bbox={'boxstyle': 'round'}, fontsize=20)

            # Scatter plot on the joint axis
            g.ax_joint.scatter(self.df[num_col1], self.df[num_col2])
            g.figure.set_figwidth(17)
            g.figure.set_figheight(8)

            # Set axis labels and adjust layout
            g.set_axis_labels(xlabel=num_col1, ylabel=num_col2, size=15)
            plt.tight_layout()

            # Remove grid
            plt.grid(False)

        # Display the plot
        plt.show()

    # This method returns seaborn heatmap
    def list_heatmap(self, columns: list) -> None:
        # Error Handling
        self._validate_not_empty()

        if not isinstance(columns, list):
            raise ValueError('Parameter must be a list')
        
        for col in columns:
            self._check_column_present(col)
            self._check_numerical_column(col)
            self._check_no_nulls(col)

        with plt.style.context(self.style):
            fig, ax = plt.subplots(figsize=(17, 8))

            selected_columns = self.df[columns]
            correlation_matrix = selected_columns.corr()

            sns.heatmap(correlation_matrix, annot=True, cmap='winter', ax=ax)
            ax.set_title(f"Heatmap of {columns}")

        # Display the plot
        plt.show()

    # This method returns seaborn univariate barplot with grouping if desired
    def countplot(self, column: str, hue_col: Optional[str] = None, limit: Optional[int] = None) -> None:
        # Error Handling
        self._validate_not_empty()
        self._check_column_present(column)
        self._check_no_nulls(column)
        if hue_col:
            self._check_column_present(hue_col)
            self._check_no_nulls(hue_col)
            self._check_categorical_column(hue_col)

        with plt.style.context(self.style):
            fig, ax = plt.subplots(figsize=(17, 8))
            if limit is not None and isinstance(limit, int):
                order = self.df[column].value_counts(normalize=True).iloc[:limit].index
            else:
                order = self.df[column].value_counts(normalize=True).index
            sns.countplot(data=self.df, x=column, hue=hue_col, order=order, ax=ax)

            ax.grid(False)
            if limit is not None:
                title = f"{column.title().replace('_', ' ')} Countplot [Top {limit}]"
            else:
                title = f"{column.title().replace('_', ' ')} Countplot"
            if hue_col:
                title += f' with {hue_col.title().replace("_", " ")} Categories'
            ax.set_title(title)
            ax.set(xlabel=None)
            ax.set(ylabel=None)

            # total = len(self.df[column])
            for p in ax.patches:
                count = int(p.get_height())
                if count == 0:
                    continue #skip annotating zero-height (empty) bars
                x = p.get_x() + p.get_width() / 2
                y = p.get_height()
                ax.annotate(f"{count}", (x, y), ha="center", va="bottom", fontsize=13, fontweight="semibold")
        # Display the plot
        plt.yticks(ticks=[])
        plt.show()

    # This method returns seaborn line graph with color encoding from a certian column if desired
    def lineplot(self, x_column: str, y_column: str, hue_column: Optional[str] = None, errors: Optional[str] = None) -> None:
        # Error Handling
        self._validate_not_empty()
        self._check_column_present(x_column)
        self._check_column_present(y_column)
        self._check_no_nulls(x_column)
        self._check_no_nulls(y_column)
        self._check_categorical_column(x_column)
        self._check_numerical_column(y_column)
        if hue_column:
            self._check_column_present(hue_column)
            self._check_no_nulls(hue_column)
            self._check_categorical_column(hue_column)

        # Plotting
        with plt.style.context(self.style):
                fig, ax = plt.subplots(figsize=(17, 8))

                sns.lineplot(data = self.df, 
                             x = x_column, 
                             y = y_column, 
                             hue = hue_column, 
                             marker='o', 
                             errorbar = errors, 
                             ax=ax
                )
                ax.set_ylabel(None)
                ax.grid(False)

                # Get lines and annotate each point on each line
                for line in ax.lines:
                    x_data = line.get_xdata()
                    y_data = line.get_ydata()
                    for (x, y) in zip(x_data, y_data):
                        ax.annotate(
                            f"{y:.2f}",        # Format as you like
                            (x, y),
                            textcoords="offset points",
                            xytext=(0, 15),
                            ha="center",
                            fontsize=10,
                            fontweight="bold"
                )

                title = f"{x_column.title().replace('_', ' ')} vs {y_column.title().replace('_', ' ')} lineplot"
                if hue_column:
                    title += f' with {hue_column.title().replace("_", " ")} Categories'
                ax.set_title(title)

        # Display the plot
        plt.yticks([])
        plt.show()

    # This method returns a pie chart
    def pie_chart(self, column: str) -> None:
        # Error Handling
        self._validate_not_empty()
        self._check_column_present(column)
        self._check_categorical_column(column)
        self._check_no_nulls(column)

        # Create the pie chart
        with plt.style.context(self.style):
            sorted_counts = self.df[column].value_counts()
            fig, ax = plt.subplots(figsize=(8, 8))

            # Customize the pie chart appearance
            wedges_props = {'linewidth': 3, 'edgecolor': 'white'}
            ax.pie(sorted_counts, labels=sorted_counts.index, startangle=90, counterclock=False,
                   autopct='%1.1f%%', textprops={'fontsize': 15}, wedgeprops=wedges_props)

            # Set equal aspect ratio and title
            ax.axis('equal')
            ax.set_title(f"{column.title().replace('_', ' ')} Pie Chart")

            # Adjust layout and display the chart
            plt.tight_layout()

        # Display the plot
        plt.show()

    # This method returns a donut pie chart
    def donut_pie_chart(self, column: str) -> None:
        # Error Handling
        self._validate_not_empty()
        self._check_column_present(column)
        self._check_categorical_column(column)
        self._check_no_nulls(column)

        # Plotting
        with plt.style.context(self.style):
            # Create a figure and axis with a specified size
            fig, ax = plt.subplots(figsize=(12, 12))

            # Create a circle at the center of the plot
            my_circle = plt.Circle((0, 0), 0.7, color='black')

            # Get sorted value counts of the categorical column
            sorted_counts = self.df[column].value_counts()

            # Set properties for the wedges in the pie chart
            wedge_props = {'width': 0.5, 'linewidth': 7, 'edgecolor': 'black'}

            # Plot the donut chart
            ax.pie(sorted_counts, labels=sorted_counts.index, startangle=90,
                   counterclock=False, wedgeprops=wedge_props,
                   autopct='%1.1f%%', pctdistance=0.85, textprops={'fontsize': 14})

            # Add the circle to create the donut effect
            p = plt.gcf()
            p.gca().add_artist(my_circle)

            # Set the title of the chart
            ax.set_title(f"{column.title().replace('_', ' ')} Donut Chart")

            # Display the plot
            plt.show()

    def violinplot(self, cat_col: str, num_col: str) -> None:
        # Error Handling
        self._validate_not_empty()
        self._check_column_present(cat_col)
        self._check_column_present(num_col)
        self._check_categorical_column(cat_col)
        self._check_numerical_column(num_col)
        self._check_no_nulls(cat_col)
        self._check_no_nulls(num_col)

        # Set the plotting style
        with plt.style.context(self.style):
            # Create a figure and axis for the violin plot
            fig, ax = plt.subplots(figsize=(17, 8))

            # Create the violin plot
            sns.violinplot(data=self.df, x=cat_col, y=num_col, ax=ax)

            # Calculate medians and observations for each group
            medians = self.df.groupby([cat_col], observed=False)[num_col].median()
            obs = self.df[cat_col].value_counts().reindex(medians.index).values

            # Add labels with number of observations to the plot
            obs_labels = [f'n: {count}' for count in obs]
            for i, (tick, label) in enumerate(zip(ax.get_xticks(), ax.get_xticklabels())):
                ax.text(tick, medians.iloc[i] + 0.03, obs_labels[i], ha='center', size='large', color='black',
                        weight='semibold')

            # Turn off the grid
            ax.grid(False)

            # Set the title of the chart
            ax.set_title(f"{num_col.title().replace('_', ' ')} by {cat_col.title().replace('_', ' ')} Violin Chart")

        # Display the plot
        plt.show()

    # This method shows seaborn violinplot with grouping
    def violinplot_with_hue(self, cat_col: str, num_col: str, hue_col: str) -> None:
        # Error Handling
        self._validate_not_empty()
        self._check_column_present(cat_col)
        self._check_column_present(num_col)
        self._check_column_present(hue_col)
        self._check_categorical_column(cat_col)
        self._check_numerical_column(num_col)
        self._check_categorical_column(hue_col)
        self._check_no_nulls(cat_col)
        self._check_no_nulls(num_col)
        self._check_no_nulls(hue_col)

        # Set the plotting style
        with plt.style.context(self.style):
            # Create a figure and axis for the violin plot
            fig, ax = plt.subplots(figsize=(17, 8))

            # Create the violin plot
            sns.violinplot(data = self.df, x=cat_col, y=num_col, hue=hue_col, ax=ax)

            # Turn off the grid
            ax.grid(False)

            # Adjust legend in plot
            ax.legend(loc='upper center', title=hue_col.title().replace('_', ' '))

            # Set the title of the chart
            ax.set_title(f"{num_col.title().replace('_', ' ')} by {cat_col.title().replace('_', ' ')} with {hue_col.title().replace('_', ' ')} Grouping Violin Chart")

        # Display the plot
        plt.show()

    # The method creates circular bar plot
    def circular_barplot(self, cat_col: str, num_col: str, bar_color: str) -> None:
        # Error Handling
        self._validate_not_empty()
        self._check_column_present(cat_col)
        self._check_column_present(num_col)
        self._check_categorical_column(cat_col)
        self._check_numerical_column(num_col)
        self._check_no_nulls(cat_col)
        self._check_no_nulls(num_col)

        # Set plotting style
        with plt.style.context(self.style):
            # Reorder dataframe
            df_grouped = self.df.groupby([cat_col], observed=False)[num_col].mean(numeric_only=True).round(0).reset_index()
            df_sorted = df_grouped.sort_values(by=[num_col])

            pal = list(sns.color_palette(palette=bar_color, n_colors=len(df_sorted)).as_hex())

            # initialize figure
            plt.figure(figsize=(20, 10))
            ax = plt.subplot(111, polar=True)
            plt.axis('off')

            # constraints = parameters controling plot layout
            # upper_limit = 100
            lower_limit = 30
            label_padding = 4

            # compute max and min in the dataset
            max_vl = df_sorted[num_col].max()

            # Let's compute heights: they are a conversion of each item value in those new coordinates
            # In our example, 0 in the dataset will be converted to the lowerLimit (10)
            # The maximum will be converted to the upperLimit (100)
            slope = (max_vl - lower_limit) / max_vl
            heights = slope * df_sorted[num_col] + lower_limit

            # compute the width of each bar. In total we have 2*pi = 360 degrees
            width = 2 * np.pi / len(df_sorted.index)

            # compute the angle each bar is centered on:
            indexes = list(range(1, len(df_sorted.index) + 1))
            angles = [item * width for item in indexes]

            # draw bars
            bars = ax.bar(x=angles, height=heights, width=width, bottom=lower_limit, linewidth=2,
                          edgecolor='white', color=pal)

            # Add labels
            for bar, angle, height, label in zip(bars, angles, heights, df_sorted[cat_col]):
                # labels are rotated. rotation must be specified in degrees
                rotation = np.rad2deg(angle)

                # flip some labels upside down
                alignment = ""
                if angle >= np.pi / 2 and angle < 3 * np.pi / 2:
                    alignment = 'right'
                    rotation = rotation + 180
                else:
                    alignment = "left"

                # add the labels
                ax.text(x=angle, y=lower_limit + bar.get_height() + label_padding,
                        s=label, ha=alignment, va="center", rotation=rotation, rotation_mode="anchor")

                ax.set_thetagrids([], labels=[])
        plt.show()

    # This method creates race track bar plot
    def race_track_plot(self, cat_col: str, num_col: str, bar_color: str) -> None:
        # Error Handling
        self._validate_not_empty()
        self._check_column_present(cat_col)
        self._check_column_present(num_col)
        self._check_categorical_column(cat_col)
        self._check_numerical_column(num_col)
        self._check_no_nulls(cat_col)
        self._check_no_nulls(num_col)

        # Set plotting theme
        with plt.style.context(self.style):
            # Reorder dataframe
            df_grouped = self.df.groupby([cat_col], observed=False)[num_col].mean().round(0).reset_index()
            df_sorted = df_grouped.sort_values(by=[num_col])

            pal = list(sns.color_palette(palette=bar_color, n_colors=len(df_sorted)).as_hex())

            # initialize figure
            plt.gcf().set_size_inches(12,12)
            sns.set_style('darkgrid')

            # set max value
            max_v1 = max(df_sorted[num_col]) * 1.01
            ax = plt.subplot(projection= 'polar')

            for i in range(len(df_sorted)):
                ax.barh(i, list(df_sorted[num_col])[i]*2*np.pi/max_v1,
                        label=list(df_sorted[cat_col])[i], color=pal[i])

            # Set subplot
            ax.set_theta_zero_location('N')
            ax.set_theta_direction(1)
            ax.set_rlabel_position(0)
            ax.set_thetagrids([], labels=[])
            ax.set_rgrids(range(len(df_sorted)), labels=df_sorted[cat_col])

            #set the pojection
            plt.legend(bbox_to_anchor=(1, 1), loc=2)

            # Display the plot
            plt.show()

    """ Interactive Visualization Charts using Plotly library """
    # This method produces an interactive Treemap
    def treemap(self, cat_col: str, num_col: str, color_scale: str) -> None:
        # Error Handling
        self._validate_not_empty()
        self._check_column_present(cat_col)
        self._check_column_present(num_col)
        self._check_categorical_column(cat_col)
        self._check_numerical_column(num_col)
        self._check_no_nulls(cat_col)
        self._check_no_nulls(num_col)

        # Group and remove any remaining NaNs (just in case)
        df_grouped = self.df.groupby([cat_col], observed=False)[num_col].mean(numeric_only=True).round().reset_index()
        df_grouped = df_grouped.dropna(subset=[cat_col, num_col])

        # Check for empty DataFrame after grouping/dropping
        if df_grouped.empty:
            raise ValueError("No data available after grouping. Check your input columns.")

        fig = px.treemap(
            df_grouped,
            path=[px.Constant(f'{cat_col.title()} Categories'), cat_col],
            values=num_col,
            color=num_col,
            color_continuous_scale=color_scale,
            color_continuous_midpoint=np.average(df_grouped[num_col]),
            title=f"Treemap of {cat_col.title()} Categories and the Average {num_col.title().replace('_', ' ')} for Each Category."
        )
        fig.update_layout(margin=dict(t=50, l=25, r=25, b=25), title_x=0.5)
        fig.show()

    # This method produces interactive pie chart with percentages
    def percentage_pie_chart(self, cat_col: str, num_col: str, bar_col: str) -> None:
        # Error Handling
        self._validate_not_empty()
        self._check_column_present(cat_col)
        self._check_column_present(num_col)
        self._check_categorical_column(cat_col)
        self._check_numerical_column(num_col)
        self._check_no_nulls(cat_col)
        self._check_no_nulls(num_col)

        # Group by and calculate mean
        df_grouped = self.df.groupby([cat_col], observed=False)[num_col].mean(numeric_only=True).round(2).reset_index()

        # Reorder dataframe
        df_sorted = df_grouped.sort_values(by=[num_col])

        # Color palette
        pal = list(sns.color_palette(palette=bar_col, n_colors=len(df_sorted)).as_hex())

        # initialize Plotly Express pie chart
        fig = px.pie(df_sorted, values=num_col, names=cat_col, color=cat_col, color_discrete_sequence=pal,
                     title=f"Percentage of {num_col.replace('_', ' ').title()} by {cat_col.replace('_', ' ').title()}")

        # Update traces and layout
        fig.update_traces(textposition='inside', textinfo='percent+label', sort=False)
        fig.update_layout(width=950, height=500)

        # Display the plot
        fig.show()

    # This method produces interactive bar chart
    def interactive_bar_chart(self, cat_col: str, num_col: str, bar_col: str) -> None:
        # Error Handling
        self._validate_not_empty()
        self._check_column_present(cat_col)
        self._check_column_present(num_col)
        self._check_categorical_column(cat_col)
        self._check_numerical_column(num_col)
        self._check_no_nulls(cat_col)
        self._check_no_nulls(num_col)

        # Group by and calculate mean
        df_grouped = self.df.groupby([cat_col], observed=False)[num_col].mean(numeric_only=True).round(2).reset_index()

        # Reorder dataframe
        df_sorted = df_grouped.sort_values(by=[num_col])

        # Color palette
        pal = list(sns.color_palette(palette=bar_col, n_colors=len(df_sorted)).as_hex())

        # initialize Plotly Express bar chart
        fig = px.bar(df_sorted, x=cat_col, y=num_col, text=num_col,
                        color=cat_col, color_discrete_sequence=pal,
                        title= f"Average {num_col.replace('_', ' ').title()} by {cat_col.replace('_', ' ').title()}")

        # Update traces and layout
        fig.update_traces(texttemplate='%{text:.3s}', textposition='outside')
        fig.update_layout({'plot_bgcolor':'white', 'paper_bgcolor':'white'}, title_x=0.5)
        fig.update_layout(width=1100, height=500, margin=dict(t=50, l=15, r=15, b=15))

        # Display the plot
        fig.show()

    # This method creates interactive polar chart
    def polar_line_chart(self, cat_col: str, num_col: str, bar_col: str) -> None:
        # Error Handling
        self._validate_not_empty()
        self._check_column_present(cat_col)
        self._check_column_present(num_col)
        self._check_categorical_column(cat_col)
        self._check_numerical_column(num_col)
        self._check_no_nulls(cat_col)
        self._check_no_nulls(num_col)

        # Group by and calculate mean
        df_grouped = self.df.groupby([cat_col], observed=False)[num_col].mean(numeric_only=True).round(2).reset_index()

        # Reorder dataframe
        df_sorted = df_grouped.sort_values(by=[num_col])

        # Color palette
        pal = list(sns.color_palette(palette=bar_col, n_colors=len(df_sorted)).as_hex())

        # initialize Plotly Express bar chart
        fig = px.line_polar(df_sorted, r=num_col, theta=cat_col, line_close=True)

        # Update traces and layout
        fig.update_traces(fill='toself', line = dict(color=pal[0]))
        fig.update_traces(mode="markers+lines")
        fig.update_layout({'plot_bgcolor': 'rgba(0,0,0,0)', 'paper_bgcolor': 'rgba(0,0,0,0)'})

        # Display the plot
        fig.show()

    # This method creates an interactive circular bubble chart
    def circular_bubble_chart(self, cat_col: str, num_col: str, bar_col: str) -> None:
        # Error Handling
        self._validate_not_empty()
        self._check_column_present(cat_col)
        self._check_column_present(num_col)
        self._check_categorical_column(cat_col)
        self._check_numerical_column(num_col)
        self._check_no_nulls(cat_col)
        self._check_no_nulls(num_col)

        # Group by and calculate mean
        df_grouped = self.df.groupby([cat_col], observed=False)[num_col].mean(numeric_only=True).round(2).reset_index()

        # Reorder dataframe
        df_sorted = df_grouped.sort_values(by=[num_col])

        # Labels for each bubble
        df_sorted['labels'] = ['<b>'+i+'<b>'+format(j,',') for i,j in zip(df_sorted[cat_col], df_sorted[num_col])]

        # Change the layout of the bubble so they are in a circluar pattern
        e = 360 / len(df_sorted)
        deg = [i * e for i in list(range(len(df_sorted)))]
        df_sorted['x'] = [math.cos(i * math.pi / 180) for i in deg]
        df_sorted['y'] = [math.sin(i * math.pi / 180) for i in deg]

        # Color palette
        pal = list(sns.color_palette(palette=bar_col, n_colors=len(df_sorted)).as_hex())

        # initialize Plotly Express scatter chart
        fig = px.scatter(df_sorted, x='x', y='y', color=cat_col, color_discrete_sequence=pal,
                             size=num_col, text='labels', size_max=40)

        # Update traces and layout
        fig.update_layout(width=800, height=800, margin= dict(t=0, l=0, r=0, b=0),
                              showlegend=False)
        fig.update_traces(textposition='bottom center')
        fig.update_xaxes(showgrid=False, zeroline=False, visible=False)
        fig.update_yaxes(showgrid=False, zeroline=False, visible=False)
        fig.update_layout({'plot_bgcolor':'white', 'paper_bgcolor':'white'})

        # Display the plot
        fig.show()

    """ Multi-graph subplots """

    # This method makes subplots of regresssion plots
    def regression_subplots(self, cat_col: str, num_col1: str, num_col2: str, sub_1: int, sub_2: int) -> None:
        # Error Handling
        self._validate_not_empty()
        self._check_column_present(cat_col)
        self._check_column_present(num_col1)
        self._check_column_present(num_col2)
        self._check_categorical_column(cat_col)
        self._check_numerical_column(num_col1)
        self._check_numerical_column(num_col2)
        self._check_no_nulls(cat_col)
        self._check_no_nulls(num_col1)
        self._check_no_nulls(num_col2)

        # Set plotting theme
        with plt.style.context(self.style):
            cat_values = self.df[cat_col].unique().tolist()

            num_subplots = min(len(cat_values), sub_1 * sub_2)

            # Initialize figure
            fig, axes = plt.subplots(sub_1, sub_2, figsize=(20, 20))
            axes = axes.flatten()

            # Iterate columns and plot corresponding data
            for i, cat_value in enumerate(cat_values):
                ax = axes[i]
                data = self.df[self.df[cat_col] == cat_value]

                # Scatter plot with regression line
                sns.regplot(data=data, x=num_col1, y=num_col2, ci=False, ax=ax)

                # Annotations
                ax.annotate(cat_value, xy=(0.05, 0.9), xycoords='axes fraction',
                            bbox=dict(boxstyle="round", fc='tab:red', alpha=0.6))
                correlation_coefficient = stats.pearsonr(data[num_col1], data[num_col2])[0]
                ax.annotate(f'$\\rho = {correlation_coefficient:.2f}$', xy=(0.05, 0.8), xycoords='axes fraction')

                # Remove grid
                ax.grid(False)

        # Remove extra empty subplots
        for i in range(num_subplots, sub_1 * sub_2):
            fig.delaxes(axes[i])

        # Adjust layout and display
        plt.tight_layout()

        # Display the plot
        plt.show()

    # This method returns subplots of histograms for numerical columns
    def histogram_subplots(self, sub_1: int, sub_2: int) -> None:
        # Error Handling
        self._validate_not_empty()
        
        # Identify all the columns that are numerical
        num_cols: list[str] = [col for col in self.df.columns if self.df[col].dtype in ['int64', 'float64']]

        # Check for null values
        for col in num_cols:
            self._check_no_nulls(col)

        # Set plotting style
        with plt.style.context(self.style):
            # Initialize figure
            fig, axes = plt.subplots(sub_1, sub_2, figsize=(20, 20))
            axes = axes.flatten()

            # Iterate columns and plot corresponding data
            for i, col in enumerate(num_cols):

                # Seaborn histogram
                sns.histplot(x=self.df[col], kde=True, ax=axes[i])
                axes[i].set_xlabel(col)
                axes[i].grid(False)


            # Remove extra subplots if there are more axes than needed
            for j in range(len(num_cols), len(axes)):
                fig.delaxes(axes[j])

            # Display the plot
            plt.show()

    # This method return subplots of countplots of categorical columns
    def cat_count_subplots(self, sub_1: int, sub_2: int, limit: Optional[int] = None) -> None:
        # Error Handling
        self._validate_not_empty()
        
        # Identify all the columns that are categorical
        cat_cols: list[str] = [col for col in self.df.columns if self.df[col].dtype not in ['int64', 'float64']]

        # Check for null values
        for col in cat_cols:
            self._check_no_nulls(col)

        # Set plotting style
        with plt.style.context('ggplot'):
            # Initialize figure
            fig, axes = plt.subplots(sub_1, sub_2, figsize=(20, 20))
            axes = axes.flatten()

        # Iterate columns and plot corresponding data
        for i, col in enumerate(cat_cols):

            # Seaborn count plot
            if limit is not None and isinstance(limit, int):
                order = self.df[col].value_counts(normalize=True).iloc[:limit].index
            else:
                order = self.df[col].value_counts(normalize=True).index
            sns.countplot(data=self.df, x=col, order=order, ax=axes[i])
            axes[i].set_xlabel(col)
            axes[i].grid(False)

            # Annotate each bar with its count
            for p in axes[i].patches:
                count = int(p.get_height())
                if count == 0:
                    continue # skip annotating zero-height (empty) bars
                x = p.get_x() + p.get_width() / 2
                y = p.get_height()
                axes[i].annotate(f"{count}", (x, y), ha="center", va="bottom", fontsize=10, fontweight="semibold")

        # Remove extra subplots if there are more axes than needed
        for j in range(i + 1, len(axes)):
            fig.delaxes(axes[j])

        # Display the plot
        plt.show()

    # This method return subplots of scatter plots
    def scatter_subplots(self, num_col: str, sub_1: int, sub_2: int, hue_col: Optional[str] = None) -> None:
        # Error Handling
        self._validate_not_empty()
        self._check_column_present(num_col)
        self._check_numerical_column(num_col)
        self._check_no_nulls(num_col)
        if hue_col:
            self._check_column_present(hue_col)
            self._check_no_nulls(hue_col)
            self._check_categorical_column(hue_col)

        # Identify all the columns that are numerical
        num_cols: list[str] = [col for col, dtype in zip(self.df.columns, self.df.dtypes)
                    if dtype in ['int64', 'float64'] and col != num_col]

        # Set plotting style
        with plt.style.context(self.style):
            num_plots = min(len(num_cols), sub_1 * sub_2)

            # Initialize figure
            fig, axes = plt.subplots(sub_1, sub_2, figsize=(20, 20))
            axes = axes.flatten()

            # Iterate columns and plot corresponding data
            for col, ax in zip(num_cols, axes):
                # Seaborn Scatterplot
                sns.scatterplot(x=col, y=num_col, hue=hue_col, data=self.df, ax=ax)
                ax.set_xlabel(col)
                ax.grid(False)

            # Remove extra subplots if there are more axes than needed
            for i in range(num_plots, sub_1 * sub_2):
                fig.delaxes(axes[i])

            plt.tight_layout()

        # Display the plot
        plt.show()

    # This method returns subplots of boxplots for numerical columns
    def box_subplots(self, sub_1: int, sub_2: int) -> None:
        # Error Handling
        self._validate_not_empty()
        
        # Identify all the columns that are numerical
        num_cols: list[str] = [col for col in self.df.columns if self.df[col].dtype in ['int64', 'float64']]

        for col in num_cols:
            self._check_no_nulls(col)

        # Set plotting style
        with plt.style.context(self.style):    

        # Initialize figure
            fig, axes = plt.subplots(sub_1, sub_2, figsize=(20, 20))
            axes = axes.flatten()

        # Iterate columns and plot corresponding data
        for i, col in enumerate(num_cols):

            sns.boxplot(x=self.df[col], ax=axes[i])
            axes[i].set_xlabel(col)

        # Remove extra subplots if there are more axes than needed
        for j in range(len(num_cols), len(axes)):
            fig.delaxes(axes[j])

        # Display the plot
        plt.show()

    # This method returns subplots of barplots
    def bar_subplots(self, cat_col: str, sub_1: int, sub_2: int, limit: Optional[int] = None) -> None:
        # Error Handling
        self._validate_not_empty()
        self._check_column_present(cat_col)
        self._check_no_nulls(cat_col)

        # Identify all the columns that are numerical
        num_cols = self.df.select_dtypes(include=['int64', 'float64']).columns

        for col in num_cols:
            self._check_no_nulls(col)

        # Set plotting style
        with plt.style.context(self.style):

            # Initialize figure
            fig, axes = plt.subplots(sub_1, sub_2, figsize=(20, 20))
            axes = axes.flatten()

            # Iterate columns and plot corresponding data
            for j, ax in zip(num_cols, axes):

                # Use seaborn barplot
                if limit is not None and isinstance(limit, int):
                    order = self.df.groupby(cat_col, observed=False).mean(numeric_only=True).sort_values(j, ascending=False).iloc[:limit].index
                else:
                    order = self.df.groupby(cat_col, observed=False).mean(numeric_only=True).sort_values(j, ascending=False).index
                sns.barplot(x=cat_col, y=j, hue=cat_col, order=order, palette='magma', legend=False, err_kws={'linewidth': 0}, data=self.df, ax=ax)

                ax.set_title(f"{cat_col.title().replace('_', ' ')} vs. Average {j.title().replace('_', ' ')} Bar Chart")
                ax.set_xlabel(j)
                ax.grid(False)

                # Annotate the bars with their heights
                for p in ax.patches:
                    ax.annotate(format(p.get_height(), '.1f'),
                                (p.get_x() + p.get_width() / 2, p.get_height()),
                                ha='center', va='center', size=15, xytext=(0, 8), textcoords='offset points')

            # Remove extra subplots if there are more axes than needed
            for i in range(len(num_cols), len(axes)):
                fig.delaxes(axes[i])

                plt.tight_layout()

            # Display the plot
            plt.show()

