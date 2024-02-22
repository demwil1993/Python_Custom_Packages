# Wrangler Class

The `Wrangler` class is a custom extension of the `pd.DataFrame` class with additional functionalities for data preprocessing and analysis.

## Inheritance

The `Wrangler` class inherits from the `pd.DataFrame` class and extends its functionalities with custom methods and attributes.

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

# Graphs Class Documentation

The `Graphs` class provides a variety of methods for visualizing data using seaborn and matplotlib libraries. This class is designed to facilitate the creation of different types of plots for exploratory data analysis.

## Class Initialization

### Constructor

- **`__init__(self, df, style='ggplot')`**:

- Description: Initializes the `Graphs` object,
- Parameters:
  - `df`: Pandas DataFrame: The dataset to be visualized.
  - `style`: String (optional): The style of the plots. Default is 'ggplot'.

## Visualization Methods

### Single Visualization Graphs

- **`histogram(self, column)`**:

  - Description: Generates a histogram plot for the specified column.
  - Parameters:
    - `column`: String: The name of the column for which the histogram is to be plotted.

- **`categorical_boxplot(self, categorical_column, numerical_column)`**:

  - Description: Generates a boxplot for categorical vs numerical data.
  - Parameters:
    - `categorical_column`: String: The name of the categorical column.
    - `numerical_column`: String: The name of the numerical column.

- **`categorical_boxplot_with_hue(self, categorical_column, numerical_column, hue_column)`**:

  - Description: Generates a boxplot with hue encoding for categorical vs numerical data.
  - Parameters:
    - `categorical_column`: String: The name of the categorical column.
    - `numerical_column`: String: The name of the numerical column.
    - `hue_column`: String: The name of the column to encode with hues.

- **`categorical_barplot(self, cat_column, num_column, hue_col=None)`**

  - Description: Generates a barplot for categorical vs numerical data.
  - Parameters:
    - `cat_column`: String: The name of the categorical column.
    - `num_column`: String: The name of the numerical column.
    - `hue_col`: String (optional): The name of the column for hue encoding.

- **`scatterplot(self, num_col1, num_col2, hue_col=None)`**:

  - Description: Generates a scatterplot for two numerical columns.
  - Parameters:
    - `num_col1`: String: The name of the first numerical column.
    - `num_col2`: String: The name of the second numerical column.
    - `hue_col`: String (optional): The name of the column for hue encoding.

- **`jointplot(self, num_col1, num_col2)`**:

  - Description: Generates a joint plot with regression line for two numerical columns.
  - Parameters:
    - `num_col1`: String: The name of the first numerical column.
    - `num_col2`: String: The name of the second numerical column.

- **`list_heatmap(self, columns)`**:

  - Description: Generates a heatmap for a list of numerical columns.
  - Parameters:
    - `columns`: List of Strings: The names of the numerical columns.

- **`countplot(self, column, hue_col=None)`**:

  - Description: Generates a countplot for a single column.
  - Parameters:
    - `column`: String: The name of the column.
    - `hue_col`: String (optional): The name of the column for hue encoding.

- **`lineplot(self, x_column, y_column, hue_column=None, errors=None)`**:

  - Description: Generates a lineplot for two columns.
  - Parameters:
    - `x_column`: String: The name of the x-axis column.
    - `y_column`: String: The name of the y-axis column.
    - `hue_column`: String (optional): The name of the column for hue encoding.
    - `errors`: bool (optional): Whether to plot error bars.

- **`pie_chart(self, column)`**:

  - Description: Generates a pie chart for a categorical column.
  - Parameters:
    - `column`: String: The name of the categorical column.

- **`donut_pie_chart(self, column)`**:

  - Description: Generates a donut pie chart for a categorical column.
  - Parameters:
    - `column`: String: The name of the categorical column.

- **`violinplot(self, cat_col, num_col)`**:

  - Description: Generates a violin plot for categorical vs numerical data.
  - Parameters:
    - `cat_col`: String: The name of the categorical column.
    - `num_col`: String: The name of the numerical column.

- **`violinplot_with_hue(self, cat_col, num_col, hue_col)`**:

  - Description: Generates a violin plot with hue encoding for categorical vs numerical data.
  - Parameters:
    - `cat_col`: String: The name of the categorical column.
    - `num_col`: String: The name of the numerical column.
    - `hue_col`: String: The name of the column to encode with hues.

- **`circular_barplot(self, cat_col, num_col, bar_color)`**

  - Description: Generates a circular bar plot for categorical vs numerical data.
  - Parameters:
    - `cat_col`: String: The name of the categorical column.
    - `num_col`: String: The name of the numerical column.
    - `bar_color`: String: The color palette for the bars.

- **`race_track_plot(self, cat_col, num_col, bar_color)`**

  - Description: Generates a race track bar plot for categorical vs numerical data.
  - Parameters:
    - `cat_col`: String: The name of the categorical column.
    - `num_col`: String: The name of the numerical column.
    - `bar_color`: String: The color palette for the bars.

