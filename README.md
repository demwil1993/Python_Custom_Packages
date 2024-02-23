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
      
- **`bool_datatype(self, column, true_value, false_value)`**

  - Description: This method type casts an object datatype into a boolean datatype based on specified true and false values.

  - Parameters:
    - `column` (str): The name of the column in the DataFrame.
    - `true_value`: The value in the column to be considered as True.
    - `false_value`: The value in the column to be considered as False.

  - **Note**: This method modifies the DataFrame in place.

### Miscellaneous

- **`counter(self, column)`**

  - Description: This method prints a dictionary containing the unique values of a specified column along with the number of occurrences of each value.

  - Parameter(s):
    - `column`: String: The name of the column for which unique values and their occurrences are counted.

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

- **`categorical_boxplot(self, categorical_column, numerical_column)`**

  - Description: This method generates a seaborn box plot to visualize the distribution of a numerical column (`numerical_column`) grouped by a categorical column (`categorical_column`).

  - Parameter(s):
    - `categorical_column`: String: The name of the categorical column for grouping.
    - `numerical_column`: String: The name of the numerical column to be plotted.


- **`categorical_boxplot_with_hue(self, categorical_column, numerical_column, hue_column)`**

  - Description: This method generates a seaborn box plot to visualize the distribution of a numerical column (`numerical_column`) grouped by a categorical column (`categorical_column`). Additionally, it encodes another categorical column (`hue_column`) by color to represent different groups.

  - Parameter(s):
    - `categorical_column`: String: The name of the categorical column for grouping.
    - `numerical_column`: String: The name of the numerical column to be plotted.
    - `hue_column`: String: The name of the categorical column for color encoding.

- **`categorical_barplot(self, cat_column, num_column, hue_col=None)`**

  - Description: This method generates a seaborn bivariate bar plot to visualize the relationship between a categorical column (`cat_column`) and a numerical column (`num_column`). Optionally, it can encode a third categorical column (`hue_col`) by color to represent different groups.

  - Parameter(s):
    - `cat_column`: String: The name of the categorical column.
    - `num_column`: String: The name of the numerical column.
    - `hue_col`: String or None: The name of the categorical column for color encoding (optional).

- **`scatterplot(self, num_col1, num_col2, hue_col=None)`**

  - Description: This method generates a seaborn scatterplot to visualize the relationship between two numerical columns (`num_col1` and `num_col2`). Optionally, it can encode a third categorical column (`hue_col`) by color to represent different groups.

  - Parameter(s):
    - `num_col1`: String: The name of the first numerical column.
    - `num_col2`: String: The name of the second numerical column.
    - `hue_col`: String or None: The name of the categorical column for color encoding (optional).

- **`jointplot(self, num_col1, num_col2)`**

  - Description: This method creates a seaborn jointplot with a regression line to visualize the relationship between two numerical columns.

  - Parameters:
    - `num_col1`: String: The name of the first numerical column.
    - `num_col2`: String: The name of the second numerical column.

- **`list_heatmap(self, columns)`**

  - Description: This method creates a seaborn heatmap to visualize the correlation matrix between the numerical columns specified in the input list (`columns`).

  - Parameter(s):
    - `columns`: List of strings: The list of numerical column names for which the correlation matrix will be calculated and visualized.

- **`countplot(self, column, hue_col=None)`**

  - Description: This method creates a seaborn bar plot to visualize the distribution of values in a single categorical column (`column`). Optionally, it can group the data based on another categorical column (`hue_col`), encoding different groups with different colors.

  - Parameter(s):
    - `column`: String: The name of the categorical column to be plotted.
    - `hue_col`: String or None: The name of the categorical column for grouping (optional).

- **`lineplot(self, x_column, y_column, hue_column=None, errors=None)`**

  - Description: This method creates a seaborn line plot to visualize the relationship between a numerical column (`y_column`) and a categorical column (`x_column`). Optionally, it can encode another categorical column (`hue_column`) using different colors. Error bars can also be included if desired.
  
  - Parameter(s):
    - `x_column`: String: The name of the categorical column on the x-axis.
    - `y_column`: String: The name of the numerical column on the y-axis.
    - `hue_column`: String or None: The name of the categorical column for color encoding (optional).
    - `errors`: String or None: The type of error bars to include (optional).

- **`pie_chart(self, column)`**

  - Description: This method creates a pie chart to visualize the distribution of categorical data in a specified column. Each category in the column is represented by a wedge in the pie chart, and the size of each wedge corresponds to the proportion of that category in the dataset.
  
  - Parameter(s):
    - `column`: String: The name of the categorical column for which the pie chart will be created.

- **`donut_pie_chart(self, column)`**

  - Description: This method creates a donut pie chart to visualize the distribution of categorical data in a specified column. Each category in the column is represented by a wedge in the pie chart, and the size of each wedge corresponds to the proportion of that category in the dataset.
  
  - Parameter(s):
    - `column`: String: The name of the categorical column for which the donut pie chart will be created.

- **`violinplot(self, cat_col, num_col)`**

  - Description: This method creates a seaborn violin plot to visualize the distribution of numerical data across categories in a categorical column. Each category in the categorical column is represented by a violin plot, showing the distribution of the numerical data within that category.
  
  - Parameter(s):
    - `cat_col`: String: The name of the categorical column.
    - `num_col`: String: The name of the numerical column to be plotted.

- **`violinplot_with_hue(self, cat_col, num_col, hue_col)`**

  - Description: This method creates a seaborn violin plot to visualize the distribution of numerical data across categories in a categorical column, with the data grouped by another categorical column represented by hue. Each category in the primary categorical column is represented by a violin plot, and within each category, the distribution is further differentiated by hue.
  
  - Parameter(s):
    - `cat_col`: String: The name of the primary categorical column.
    - `num_col`: String: The name of the numerical column to be plotted.
    - `hue_col`: String: The name of the categorical column used for grouping and differentiating the data in the violin plot.

- **`circular_barplot(self, cat_col, num_col, bar_color)`**

  - Description: This method creates a circular bar plot to visualize the values of a numerical column across categories in a categorical column. Each category is represented by a bar, and the length of the bar corresponds to the average value of the numerical column for that category. The bars are arranged in a circular manner around the plot, resembling a circular histogram.
  
  - Parameter(s):
    - `cat_col`: String: The name of the categorical column.
    - `num_col`: String: The name of the numerical column.
    - `bar_color`: String: The color palette to be used for the bars in the plot.

- **`race_track_plot(self, cat_col, num_col, bar_color)`**

  - Description: This method creates a race track bar plot to visualize the values of a numerical column across categories in a categorical column. Each category is represented by a bar, and the length of the bar corresponds to the average value of the numerical column for that category. The bars are arranged in a circular manner resembling a race track.
  
  - Parameter(s):
    - `cat_col`: String: The name of the categorical column.
    - `num_col`: String: The name of the numerical column.
    - `bar_color`: String: The color palette to be used for the bars in the plot.

### Interactive Visualization Charts using Plotly library

- **`treemap(self, cat_col, num_col, color_scale)`**

  - Description: This method generates an interactive treemap visualization based on categorical and numerical data. It groups the data by the categorical column, calculates the mean of the numerical column for each category, and visualizes the result as a treemap. Each category is represented by a rectangle, with the area of the rectangle proportional to the average value of the numerical column for that category. The color of the rectangles is determined by the values of the numerical column, using a specified color scale.
  
  - Parameter(s):
    - `cat_col`: String: The name of the categorical column.
    - `num_col`: String: The name of the numerical column.
    - `color_scale`: String: The color palette for the treemap bars.

- **`percentage_pie_chart(self, cat_col, num_col, bar_col)`**

  - Description: This method generates an interactive pie chart visualization representing the percentage distribution of numerical data across categories in a categorical column. It calculates the mean of the numerical column for each category, rounds the values to two decimal places, and then sorts the categories based on these values. Each category is represented in the pie chart, with the size of each slice proportional to the percentage of the total numerical values it represents. The color of each slice can be customized using a specified color palette.
  
  - Parameter(s):
    - `cat_col`: String: The name of the categorical column.
    - `num_col`: String: The name of the numerical column.
    - `bar_col`: String: The name of the color palette to be used for the pie chart slices.

- **`interactive_bar_chart(self, cat_col, num_col, bar_col)`**

  - Description: This method generates an interactive bar chart visualization depicting the average numerical values across categories in a categorical column. It calculates the mean of the numerical column for each category, rounds the values to two decimal places, and sorts the categories based on these mean values. Each category is represented by a bar in the bar chart, with the height of the bar corresponding to the average numerical value. The color of each bar can be customized using a specified color palette.
  
  - Parameter(s):
    - `cat_col`: String: The name of the categorical column.
    - `num_col`: String: The name of the numerical column.
    - `bar_col`: String: The name of the color palette to be used for the bar chart bars.

- **`polar_line_chart(self, cat_col, num_col, bar_col)`**

  - Description: This method generates an interactive polar line chart visualization illustrating the average numerical values across categories in a categorical column. It calculates the mean of the numerical column for each category, rounds the values to two decimal places, and sorts the categories based on these mean values. Each category is represented by a point on the polar chart, and the lines connecting these points create a line chart. The color of the lines can be customized using a specified color palette.
  
  - Parameter(s):
    - `cat_col`: String: The name of the categorical column.
    - `num_col`: String: The name of the numerical column.
    - `bar_col`: String: The name of the color palette to be used for the line chart.

- **`circular_bubble_chart(self, cat_col, num_col, bar_col)`**

  - Description: This method generates an interactive circular bubble chart visualization representing the average numerical values across categories in a categorical column. It calculates the mean of the numerical column for each category, rounds the values to two decimal places, and sorts the categories based on these mean values. Each category is represented by a bubble, positioned in a circular pattern, with the size of the bubble corresponding to the average numerical value. The color of each bubble can be customized using a specified color palette.
  
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

- **`histogram_subplots(self, sub_1, sub_2)`**

  - Description: This method generates subplots of histograms for numerical columns in the DataFrame. It creates a grid of subplots with dimensions specified by `sub_1` (number of rows) and `sub_2` (number of columns), and plots histograms for each numerical column in the DataFrame. The histograms include kernel density estimation (KDE) curves for better visualization of the data distribution.
  
  - Parameter(s):
    - `sub_1`: Integer: The number of rows in the subplot grid.
    - `sub_2`: Integer: The number of columns in the subplot grid.

- **`cat_count_subplots(self, sub_1, sub_2)`**

  - Description: This method generates subplots of countplots for categorical columns in the DataFrame. It creates a grid of subplots with dimensions specified by `sub_1` (number of rows) and `sub_2` (number of columns), and plots countplots for each categorical column in the DataFrame. Each countplot displays the count of occurrences of each category in the respective categorical column. The bars in the countplots are annotated with the percentage of occurrences they represent.
  
  - Parameter(s):
    - `sub_1`: Integer: The number of rows in the subplot grid.
    - `sub_2`: Integer: The number of columns in the subplot grid.

- **`scatter_subplots(self, num_col, sub_1, sub_2, hue_col=None)`**

  - Description: This method generates subplots of scatter plots with a specified numerical column (`num_col`) on the y-axis and other numerical columns in the DataFrame on the x-axis. It creates a grid of subplots with dimensions specified by `sub_1` (number of rows) and `sub_2` (number of columns), and plots scatter plots for each pair of numerical columns in the DataFrame. If a `hue_col` is provided, the data points will be colored based on the values in the specified categorical column.
  
  - Parameter(s):
    - `num_col`: String: The name of the numerical column to be plotted on the y-axis.
    - `sub_1`: Integer: The number of rows in the subplot grid.
    - `sub_2`: Integer: The number of columns in the subplot grid.
    - `hue_col`: String (optional): The name of the categorical column used for coloring the data points in the scatter plots.

- **`box_subplots(self, sub_1, sub_2)`**

  - Description: This method generates subplots of boxplots for numerical columns in the DataFrame. It creates a grid of subplots with dimensions specified by `sub_1` (number of rows) and `sub_2` (number of columns), and plots boxplots for each numerical column in the DataFrame. Each boxplot displays the distribution of values for the respective numerical column.
  
  - Parameter(s):
    - `sub_1`: Integer: The number of rows in the subplot grid.
    - `sub_2`: Integer: The number of columns in the subplot grid.

- **`bar_subplots(self, cat_col, sub_1, sub_2)`**

  - Description: This method generates subplots of bar plots for comparing numerical columns across categories in a specified categorical column. It creates a grid of subplots with dimensions specified by `sub_1` (number of rows) and `sub_2` (number of columns), and plots bar plots for each numerical column in the DataFrame. Each bar plot displays the average value of the respective numerical column for each category in the specified categorical column.
  
  - Parameter(s):
    - `cat_col`: String: The name of the categorical column for grouping the data.
    - `sub_1`: Integer: The number of rows in the subplot grid.
    - `sub_2`: Integer: The number of columns in the subplot grid.
