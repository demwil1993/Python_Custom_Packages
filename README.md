# Python_Custom_Packages
Custom Packages for manipulating and graphing data

# Data Wrangler Package
### Package containg 2 custom classes for manipulating and graphing data:
- `wrangler`
- `graphs`

**class `wrangler` contains 18 methods that entend the Pandas Dataframe package:**

- [X] `_constructor`: returns class name of same type, the function calls another function also to excecute `__finalize__` every excecution to ensure all the attributes of pandas are imported over every execution.
- [X] `__init__`: initializes the wrangler class.
- [X] `strip_dataframe`: strips leading and trailing whitespaces from DataFrame columns and column values.
- [X] `dataframe_analylsis`: gives descriptive statistics of numerical and categorical columns, while also checking for null values and duplicate rows in dataframe.
- [X] `identify_columns`: outputs the names of teh numerical and categorical columns in a DataFrame.
- [X] `column_analysis`:outputs the number of numerical and categorical columns, while also identifying the number of columns that are numerical but act as categorical and columns with high cardinality.
- [X] `categorical_column_summary`: provides summary of categorical column, as far as value counts and ratio of data.
- [X] `numerical_column_summary`: provides summary of numerical colum as far as quantitles.
- [X] `target_cross_analysis_cat`: accepts two parameters (target column, categorical column) then cross examines the relationship between a categorical column and target column, then incorporate the information into a DataFrame.
- [X] `target_cross_analysis_num`: accepts two parameters (target column, numerical column) then cross examines the relationship between  a numerical column and target column, then incorporate the information into a DataFrame.
- [X] `normalize`: accepts a numerical column and normalizes it.
- [X] `complete_imputation`: performs imputation on entire DataFrame regardless of datatypes.
- [X] `remove_outlier_iqr`: accepts a numerical column and uses the interquartile range to identify and remove outliers in column.
- [X] `outlier_limits_iqr`: accepts a numerical column and uses the interquartile range to identify upper and lower bound limits.
- [X] `show_outliers_rows`: show rows in a DataFrame that have outliers.
- [X] `category_datatype`: type casts an object datatype into a category datatype.
- [X] `bool_datatype`: accepts 3 parameters: column, first value, and second value. Checks for if there are only 2 unique values in a column, if they are the column is type cast to a boolean datatype.
- [X] `counter`: accepts a column as a parameter, then prints a dictionary with unique values and the number of occurences.

**class `graphs` contain # of methods that use the seaborn, matplotlib, and plotly libraries to graph data:**

- [X] `__init__`: initializes the graphs class.
- [X] `histogram`: accepts 3 parameters (data, column, matplotlib theme). Outputs a single image Seaborn histogram plot from column in DataFrame.
- [X] `categorical_boxplot`: accepts 4 parameters (data, categorical column, numerical column, matplotlib theme). Outputs seaborn boxplot.
- [X] `categorical_boxplot_with_hue`: accepts 5 parameters (data, categorical column, numerical column, hue, matplotlib theme). Outputs seaborn boxplot with nested groups.
- [X] `categorical_barplot`: accepts 4 parameters (data, categorical column, numerical column, matplotlib theme). Outputs a seaborn bar graph with sorted columns in the bar graph.
- [X] `categorical_barplot_with_hue`: accepts 5 parameters (data, categorical column, numerical column, hue, matplotlib theme). Outputs seaborn bar graph with grouped columns.
- [X] `scatterplot_with_hue`: accepts 5 parameters (data, numerical column, numerical column, hue, matplotlib theme). Outputs seaborn scatterplot with points colored to match a value.
- [X] `scatterplot`: accepts 4 parameters (data, numerical column, numerical column, matplotlib theme). Outputs seaborn scatterplot.
- [X] `jointplot`: accepts 4 parameters (data, numerical column, numerical column, matplotlib theme). Outputs seaborn jointplot with regression line.
- [X] `list_heatmap`: accepts 3 parameters (data, list of columns, matplotlib theme): Output seaborn heatmap.
- [X] `multi_heatmap`: accepts 5 parameters (data, index, column, values, matplotlib theme). Outputs seaborn heatmap derived from pivot table.
- [X] `countplot`: accepts 3 parameters (data, column, matplotlib theme). Outputs a univariate seaborn univariate bar graph with sorted columns.
- [X] `countplot_with_hue`: accepts 4 parameters (data, column, hue, matplotlib theme). Outputs a univarite seaborn bar graph with grouped columns.
- [X] `lineplot`: accepts 4 parameters and 1 default parameter (data, categorical column, numerical column, matplotlib). Outputs seaborn line graph.
- [X] `lineplot_with_hue`: accepts 5 parameters and 1 default parameter (data, categorical column, numerical column, hue, matplotlib). Outputs seaborn line graph.
- [X] `pie_chart`: accepts 3 parameters (data, column, matplotlib theme). Outputs pie chart with noted percentages.
- [X] `donut_pie_chart`: accepts 3 parameters (data, column, matplotlib theme). Outputs donut pie chart with noted percentages.

