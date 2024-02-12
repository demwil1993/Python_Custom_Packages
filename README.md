# Wrangler Class

The `Wrangler` class is a custom extension of the `pd.DataFrame` class with additional functionalities for data preprocessing and analysis.

## Class Attributes

- `_metadata`: List of metadata attributes preserved in the class.

## Methods

### Initialization

- **`__init__(self, *args, **kwargs)`**: Initializes the `Wrangler` object, with an optional `my_attr` attribute.

### Data Preprocessing

- **`strip_dataframe(self)`**: Strips leading and trailing white spaces from column names and values.
- **`normalize(self, col)`**: Normalizes numerical column values.
- **`complete_imputation(self)`**: Performs imputation for missing values in the DataFrame.
- **`remove_outlier_iqr(self, column)`**: Removes outliers using the interquartile range method.
- **`outlier_limits_iqr(self, column)`**: Calculates the upper and lower bounds for outliers using the interquartile range method.
- **`show_outlier_rows(self)`**: Displays rows containing outliers in the DataFrame.

### Data Analysis

- **`dataframe_analysis(self)`**: Provides descriptive statistics, checks for null values, and duplicated rows.
- **`identify_columns(self)`**: Identifies numerical and categorical columns.
- **`column_analysis(self)`**: Provides an overview of the DataFrame, including the number of observations, variables, and types of columns.
- **`categorical_column_summary(self, column_name, plot=False)`**: Generates a summary of a categorical column and optionally plots a count plot.
- **`numerical_column_summary(self, column, plot=False)`**: Generates a summary of a numerical column and optionally plots a histogram.
- **`target_cross_analysis_cat(self, target, cat_col)`**: Cross-examines the relationship between a categorical column and a numerical target.
- **`target_cross_analysis_num(self, target, num_col)`**: Cross-examines the relationship between a numerical column and a numerical target.

### Data Transformation

- **`category_datatype(self)`**: Converts object data types to category data types.
- **`turn_null(self, val)`**: Replaces specified values with null values.
- **`bool_datatype(self, column, true_value, false_value)`**: Converts object data types to boolean data types.

### Miscellaneous

- **`counter(self, column)`**: Prints a dictionary with the unique values of a column and their occurrences.

## Inheritance

The `Wrangler` class inherits from the `pd.DataFrame` class and extends its functionalities with custom methods and attributes.

