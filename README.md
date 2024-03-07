<details>
<summary><strong>Wrangler Class Documentation</strong></summary>

The <strong>Wrangler</strong> class is a custom extension of the <strong>pd.DataFrame</strong> class with additional functionalities for data preprocessing and analysis.

### Class Attributes

- **`_metadata`**: List of metadata attributes preserved in the class.

- **`_constructor`**

  - Description: This property returns a constructor for the class. The constructor is a function that can create a new instance of the class.

  - Returns:
    - Function: A constructor function.


### Methods

### Initialization

- **`__init__(self, *args, **kwargs)`**:
  
  - Description: Initializes the <strong>`Wrangler`</strong> object, with an optional `my_attr` attribute.

### Data Preprocessing

- **`strip_dataframe(self)`**

  - Description: This method strips leading and trailing white spaces from both the DataFrame columns and the values within those columns.

  - Note: This method modifies the DataFrame in place.
    
- **`normalize(self, col)`**

  - Description: This method performs normalization on a numerical column in the DataFrame.

  - Parameters:
    - `col`: String: The name of the numerical column to be normalized.

  - Note: This method modifies the DataFrame in place by replacing the values in the specified numerical column with their normalized counterparts.
       
- **`complete_imputation(self)`**

  - Description: This method performs imputation on the entire DataFrame, filling missing values regardless of datatype. Numerical columns are imputed with the mean of the column, while categorical columns are imputed with the mode of the column.

  - Note: This method modifies the DataFrame in place.
    
- **`remove_outlier_iqr(self, column)`**

  - Description: This method uses the Interquartile Range (IQR) method to identify and remove outliers from a numerical column in the DataFrame.

  - Parameters:
    - `column`: String: The name of the numerical column from which outliers are to be removed.

  - Note: This method modifies the DataFrame in place.
    
- **`outlier_limits_iqr(self, column)`**

  - Description: This method produces the upper and lower bound rows of a DataFrame using the Interquartile Range (IQR) method based on a given column.

  - Parameters:
    - `column`: String: The name of the numerical column used to calculate outliers.

  - Returns:
    - DataFrame: Rows of the DataFrame where values are considered outliers based on the IQR method. 
    
- **`show_outlier_rows(self)`**

  - Description: This method displays all rows in the DataFrame that contain outliers using the Interquartile Range (IQR) method.

  - Note: Outliers are determined separately for each numerical column in the DataFrame.

- **`category_datatype(self)`**

  - Description: This method type casts columns with an object datatype into a category datatype.

  - Note: This method modifies the DataFrame in place.
    
- **`turn_null(self, val)`**

  - Description: This method replaces values in a DataFrame that represent an unknown value but are not recorded as null (e.g., '-', '?', '*') with NaN.

  - Parameters:
    - `val`: The value to be replaced with NaN.

  - Note: This method modifies the DataFrame in place.

- **`null_percent(self)`**

  - Description: Outputs the percentage of null values in each column of the DataFrame.

- **`drop_null_by_percent(self, percent)`**

  - Description: Drops columns in a DataFrame that have a percentage of null values greater than or equal to the specified threshold.

  - Parameters:
    - `percent`: Float: The percentage threshold above which columns will be dropped. Expressed as a value between 0 and 100.

  - Note: This method modifies the DataFrame in place.
      
- **`bool_datatype(self, column, true_value, false_value)`**

  - Description: This method type casts an object datatype into a boolean datatype based on specified true and false values.

  - Parameters:
    - `column` (str): The name of the column in the DataFrame.
    - `true_value`: The value in the column to be considered as True.
    - `false_value`: The value in the column to be considered as False.

  - **Note**: This method modifies the DataFrame in place.

### Data Analysis

- **`dataframe_analysis(self)`**:
  
  - Description: Provides descriptive statistics, checks for null values, and duplicated rows.
    
- **`identify_columns(self)`**:
  
  - Description: Identifies numerical and categorical columns.
    
- **`column_analysis(self)`**

  - Description: This method identifies and analyzes columns in the DataFrame, categorizing them into numerical and categorical columns, and provides additional insights such as high cardinality categorical columns and numerical columns treated as categorical.

  - Returns:
    - `category_columns`: List of Strings: Names of categorical columns.
    - `numerical_columns`: List of Strings: Names of purely numerical columns.
    - `category_with_hi_cardinality`: List of Strings: Names of categorical columns with high cardinality.
    
  - Note: High cardinality is defined as having more than 50 unique values.
  
- **`categorical_column_summary(self, column_name, plot=False)`**

  - Description: This method provides a summary of a categorical column in the DataFrame.

  - Parameters:
    - `column_name`: String: The name of the categorical column to summarize.
    - `plot`: Boolean (default: False): If True, a count plot of the categorical column will be displayed.

  - Note: This method prints the count and percentage ratio of each category in the specified categorical column.
      
- **`numerical_column_summary(self, column, plot=False)`**

  - Description: This method provides a summary of a numerical column in the DataFrame.

  - Parameters:
    - `column`: String: The name of the numerical column to summarize.
    - `plot`: Boolean (default: False): If True, a histogram of the numerical column will be displayed along with the summary statistics.

  - Note: This method prints the summary statistics including count, mean, standard deviation, minimum, 5th, 10th, 20th, 30th, 40th, 50th (median), 60th, 70th, 80th, 90th, 95th, and 99th percentiles.

  
- **`target_cross_analysis_cat(self, target, cat_col)`**

  - Description: This method cross-examines the relationship between a target column (numerical) and a categorical column.

  - Parameters:
    - `target`: String: The name of the numerical target column.
    - `cat_col`: String: The name of the categorical column.

  - Note: This method prints the mean of the target column grouped by the categories in the specified categorical column.
       
- **`target_cross_analysis_num(self, target, num_col)`**

  - Description: This method cross-examines the relationship between a target column and a numerical column, regardless of the datatype of the target column.

  - Parameters:
    - `target`: String: The name of the target column.
    - `num_col`: String: The name of the numerical column.

- **`counter(self, column)`**

  - Description: This method prints a dictionary containing the unique values of a specified column along with the number of occurrences of each value.

  - Parameter(s):
    - `column`: String: The name of the column for which unique values and their occurrences are counted.
</details>

<details>
<summary><strong>Graphs Class Documentation</strong></summary>

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

- **`categorical_barplot(self, cat_column, num_column, hue_col=None, limit=None)`**

  - Description: This method generates a seaborn bivariate bar plot to visualize the relationship between a categorical column (`cat_column`) and a numerical column (`num_column`). Optionally, it can encode a third categorical column (`hue_col`) by color to represent different groups. Optionally, can also limit the number of group columns shown with the `limit' parameter.

  - Parameter(s):
    - `cat_column`: String: The name of the categorical column.
    - `num_column`: String: The name of the numerical column.
    - `hue_col`: String or None: The name of the categorical column for color encoding (optional).
    - `limit`: Integer or None: The number of group columns to be shown in graph (optional).

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

- **`countplot(self, column, hue_col=None, limit=None)`**

  - Description: This method creates a seaborn bar plot to visualize the distribution of values in a single categorical column (`column`). Optionally, it can group the data based on another categorical column (`hue_col`), encoding different groups with different colors. Optionally, can also limit the number of group columns shown with the `limit' parameter.

  - Parameter(s):
    - `column`: String: The name of the categorical column to be plotted.
    - `hue_col`: String or None: The name of the categorical column for grouping (optional).
    - `limit`: Integer or None: The number of group columns to be shown in graph (optional).

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

### Multi-plot Graphs

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

- **`cat_count_subplots(self, sub_1, sub_2, limit=None)`**

  - Description: This method generates subplots of countplots for categorical columns in the DataFrame. It creates a grid of subplots with dimensions specified by `sub_1` (number of rows) and `sub_2` (number of columns), and plots countplots for each categorical column in the DataFrame. Each countplot displays the count of occurrences of each category in the respective categorical column. The bars in the countplots are annotated with the percentage of occurrences they represent. Optionally, can limit the number of group columns shown in each subplot with the `limit' parameter.
  
  - Parameter(s):
    - `sub_1`: Integer: The number of rows in the subplot grid.
    - `sub_2`: Integer: The number of columns in the subplot grid.
    - `limit`: Integer or None: The number of group columns to be shown in graph (optional).

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

- **`bar_subplots(self, cat_col, sub_1, sub_2, limit=None)`**

  - Description: This method generates subplots of bar plots for comparing numerical columns across categories in a specified categorical column. It creates a grid of subplots with dimensions specified by `sub_1` (number of rows) and `sub_2` (number of columns), and plots bar plots for each numerical column in the DataFrame. Each bar plot displays the average value of the respective numerical column for each category in the specified categorical column. Optionally, can limit the number of group columns shown in each subplot with the `limit' parameter.
  
  - Parameter(s):
    - `cat_col`: String: The name of the categorical column for grouping the data.
    - `sub_1`: Integer: The number of rows in the subplot grid.
    - `sub_2`: Integer: The number of columns in the subplot grid.
    - `limit`: Integer or None: The number of group columns to be shown in graph (optional).
</details>
