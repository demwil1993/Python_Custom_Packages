# Wrangler Class Documentation

The `Wrangler` class is a custom extension of the `pd.DataFrame` class with additional functionalities for data preprocessing and analysis.

## Inheritance

The `Wrangler` class inherits from the `pd.DataFrame` class and extends its functionalities with custom methods and attributes.

## Class Attributes

- `_metadata`: List of metadata attributes preserved in the class.

## Methods

### Initialization

- **`__init__(self, *args, **kwargs)`**:
  
  - Description: Initializes the `Wrangler` object, with an optional `my_attr` attribute.

### Data Preprocessing

- **`strip_dataframe(self)`**:
  
  - Description: Strips leading and trailing white spaces from column names and values.
    
- **`normalize(self, col)`**:
  
  - Description: Normalizes numerical column values.
  - Parameter(s):
    - `col`: String: name of numerical column
       
- **`complete_imputation(self)`**:
  
  - Description: Performs imputation for missing values in the DataFrame.
    
- **`remove_outlier_iqr(self, column)`**:
  
  - Description: Removes outliers using the interquartile range method.
    
- **`outlier_limits_iqr(self, column)`**:
  
  - Description: Calculates the upper and lower bounds for outliers using the interquartile range method.
  - Parameter(s):
    - `column`: String: name of column numerical column 
    
- **`show_outlier_rows(self)`**:
  
  - Description: Displays rows containing outliers in the DataFrame.

### Data Analysis

- **`dataframe_analysis(self)`**:
  
  - Description: Provides descriptive statistics, checks for null values, and duplicated rows.
    
- **`identify_columns(self)`**:
  
  - Description: Identifies numerical and categorical columns.
    
- **`column_analysis(self)`**:
  
  - Description: Provides an overview of the DataFrame, including the number of observations, variables, and types of columns.
  
- **`categorical_column_summary(self, column_name, plot=False)`**:
  
  - Description: Generates a summary of a categorical column and optionally plots a count plot.
  - Parameter(s):
    - `column`: String: Name of categorical column
    - `plot`: Boolean (Optional): Plot data, default is False.
      
- **`numerical_column_summary(self, column, plot=False)`**:
  
  - Description: Generates a summary of a numerical column and optionally plots a histogram.
  - Parameter(s):
    - `column`: String: Name of numerical column
    - `plot`: Boolean (Optional): Plot data, default is False.
  
- **`target_cross_analysis_cat(self, target, cat_col)`**:
  
  - Description: Cross-examines the relationship between a categorical column and a numerical target.
  - Parameter(s):
    - `target`: String: name of numerical column that will act as dependant variable
    - `cat_col`: String: name of the categorical column that will act as independant variable
       
- **`target_cross_analysis_num(self, target, num_col)`**:
  
  - Description: Cross-examines the relationship between a numerical column and  a target column of any datatype.
  - Parameter(s):
    - `target`: Any Datatype: column that will act as dependant variable
    - `num_col`: String: name of the numerical column that will act as independant variable

### Data Transformation

- **`category_datatype(self)`**:
  
  - Description: Converts object data types to category data types.
    
- **`turn_null(self, val)`**:
  
  - Description: Replaces specified values with null values.
  - Parameters:
    - `val`: any datatype: value(s) that will be replace in dataframe
      
- **`bool_datatype(self, column, true_value, false_value)`**:
  
  - Description: Converts object data types to boolean data types.
  - Parameter(s):
    -`column`: String: The name of the column
    -`true_value`: Any Datatype: The value in the column to be considered as True
    -`false_value`: Any Datatype: The value in the column to be considered as False.

### Miscellaneous

- **`counter(self, column)`**:
  
  - Description: Prints a dictionary with the unique values of a column and their occurrences.
  - Parameter(s):
    - `column`: String: The name of the column

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
  - Parameter(s):
    - `column`: String: The name of the column for which the histogram is to be plotted.

- **`categorical_boxplot(self, categorical_column, numerical_column)`**:

  - Description: Generates a boxplot for categorical vs numerical data.
  - Parameter(s):
    - `categorical_column`: String: The name of the categorical column.
    - `numerical_column`: String: The name of the numerical column.

- **`categorical_boxplot_with_hue(self, categorical_column, numerical_column, hue_column)`**:

  - Description: Generates a boxplot with hue encoding for categorical vs numerical data.
  - Parameter(s):
    - `categorical_column`: String: The name of the categorical column.
    - `numerical_column`: String: The name of the numerical column.
    - `hue_column`: String: The name of the column to encode with hues.

- **`categorical_barplot(self, cat_column, num_column, hue_col=None)`**

  - Description: Generates a barplot for categorical vs numerical data.
  - Parameter(s):
    - `cat_column`: String: The name of the categorical column.
    - `num_column`: String: The name of the numerical column.
    - `hue_col`: String (optional): The name of the column for hue encoding.

- **`scatterplot(self, num_col1, num_col2, hue_col=None)`**:

  - Description: Generates a scatterplot for two numerical columns.
  - Parameter(s):
    - `num_col1`: String: The name of the first numerical column.
    - `num_col2`: String: The name of the second numerical column.
    - `hue_col`: String (optional): The name of the column for hue encoding.

- **`jointplot(self, num_col1, num_col2)`**:

  - Description: Generates a joint plot with regression line for two numerical columns.
  - Parameter(s):
    - `num_col1`: String: The name of the first numerical column.
    - `num_col2`: String: The name of the second numerical column.

- **`list_heatmap(self, columns)`**:

  - Description: Generates a heatmap for a list of numerical columns.
  - Parameter(s):
    - `columns`: List of Strings: The names of the numerical columns.

- **`countplot(self, column, hue_col=None)`**:

  - Description: Generates a countplot for a single column.
  - Parameter(s):
    - `column`: String: The name of the column.
    - `hue_col`: String (optional): The name of the column for hue encoding.

- **`lineplot(self, x_column, y_column, hue_column=None, errors=None)`**:

  - Description: Generates a lineplot for two columns.
  - Parameter(s):
    - `x_column`: String: The name of the x-axis column.
    - `y_column`: String: The name of the y-axis column.
    - `hue_column`: String (optional): The name of the column for hue encoding.
    - `errors`: bool (optional): Whether to plot error bars.

- **`pie_chart(self, column)`**:

  - Description: Generates a pie chart for a categorical column.
  - Parameter(s):
    - `column`: String: The name of the categorical column.

- **`donut_pie_chart(self, column)`**:

  - Description: Generates a donut pie chart for a categorical column.
  - Parameter(s):
    - `column`: String: The name of the categorical column.

- **`violinplot(self, cat_col, num_col)`**:

  - Description: Generates a violin plot for categorical vs numerical data.
  - Parameter(s):
    - `cat_col`: String: The name of the categorical column.
    - `num_col`: String: The name of the numerical column.

- **`violinplot_with_hue(self, cat_col, num_col, hue_col)`**:

  - Description: Generates a violin plot with hue encoding for categorical vs numerical data.
  - Parameter(s):
    - `cat_col`: String: The name of the categorical column.
    - `num_col`: String: The name of the numerical column.
    - `hue_col`: String: The name of the column to encode with hues.

- **`circular_barplot(self, cat_col, num_col, bar_color)`**

  - Description: Generates a circular bar plot for categorical vs numerical data.
  - Parameter(s):
    - `cat_col`: String: The name of the categorical column.
    - `num_col`: String: The name of the numerical column.
    - `bar_color`: String: The color palette for the bars.

- **`race_track_plot(self, cat_col, num_col, bar_color)`**

  - Description: Generates a race track bar plot for categorical vs numerical data.
  - Parameter(s):
    - `cat_col`: String: The name of the categorical column.
    - `num_col`: String: The name of the numerical column.
    - `bar_color`: String: The color palette for the bars.

### Interactive Visualization Charts using Plotly library

- **`treemap(self, cat_col, num_col, color_scale)`**

  - Description: This method generates an interactive treemap visualization based on categorical and numerical data..
  
  - Parameter(s):
    - `cat_col`: String: The name of the categorical column.
    - `num_col`: String: The name of the numerical column.
    - `color_scale`: String: The color palette for the treemap bars.

- **`percentage_pie_chart(self, cat_col, num_col, bar_col)`**

  - Description: This method generates an interactive pie chart visualization representing the percentage distribution of numerical data across categories in a categorical column.
  
  - Parameter(s):
    - `cat_col`: String: The name of the categorical column.
    - `num_col`: String: The name of the numerical column.
    - `bar_col`: String: The name of the color palette to be used for the pie chart slices.

- **`interactive_bar_chart(self, cat_col, num_col, bar_col)`**

  - Description: This method generates an interactive bar chart visualization depicting the average numerical values across categories in a categorical column.
  
  - Parameter(s):
    - `cat_col`: String: The name of the categorical column.
    - `num_col`: String: The name of the numerical column.
    - `bar_col`: String: The name of the color palette to be used for the bar chart bars.

- **`polar_line_chart(self, cat_col, num_col, bar_col)`**

  - Description: This method generates an interactive polar line chart visualization illustrating the average numerical values across categories in a categorical column.
  
  - Parameter(s):
    - `cat_col`: String: The name of the categorical column.
    - `num_col`: String: The name of the numerical column.
    - `bar_col`: String: The name of the color palette to be used for the line chart.

- **`circular_bubble_chart(self, cat_col, num_col, bar_col)`**

  - Description: This method generates an interactive circular bubble chart visualization representing the average numerical values across categories in a categorical column.
  
  - Parameter(s):
    - `cat_col`: String: The name of the categorical column.
    - `num_col`: String: The name of the numerical column.
    - `bar_col`: String: The name of the color palette to be used for the bubble chart.

### Multi-graph subplots

- **`regression_subplots(self, cat_col, num_col1, num_col2, sub_1, sub_2)`**

  - Description: This method creates subplots of regression plots based on categorical and numerical data. It takes two numerical columns (`num_col1` and `num_col2`) and a categorical column (`cat_col`) as input and generates subplots where each subplot corresponds to a unique category in the categorical column. Each subplot contains a scatter plot with a regression line representing the relationship between the two numerical columns for the specific category. Annotations indicating the category name and the correlation coefficient are added to each subplot.
  
  - Parameter(s):
    - `cat_col`: String: The name of the categorical column.
    - `num_col1`: String: The name of the first numerical column.
    - `num_col2`: String: The name of the second numerical column.
    - `sub_1`: Integer: The number of rows in the subplot grid.
    - `sub_2`: Integer: The number of columns in the subplot grid.

