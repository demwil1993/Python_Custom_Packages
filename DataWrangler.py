import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from collections import Counter
import math
import plotly.express as px
import scipy.stats as stats

class Wrangler(pd.DataFrame):
    """ A custom DataFrame class with additional attributes. """

    # List of metadata attributes to be preserved.
    _metadata = ['my_attr']

    @property
    def _constructor(self):
        """ Returns a constructor for the class."""
        def _c(*args, **kwargs):
            return Wrangler(*args, **kwargs).__finalize__(self)
        return _c

    def __init__(self, *args, **kwargs):
        # Pop the 'my_attr' keyword argument if provided.
        self.my_attr = kwargs.pop('my_attr', None)
        super().__init__(*args, **kwargs)

    # This method strips leading and trailing white spaces from DataFrame columns and column values
    def strip_dataframe(self):
        self.columns = self.columns.str.strip() # strip whitespaces from column names

        # strip whitespaces from columns' values where the column is not a numerical datatype
        for col in self.select_dtypes(include=['object', 'category']):
            self[col] = self[col].str.strip()

    # This method provides summary statistics for categorical and numerical data columns
    def dataframe_anaylsis(self):
        try:
            # Descriptive Statistics
            print("Descriptive Statistics of Data:")
            print(self.describe(include=['object', 'float', 'int', 'category', 'bool']).T)
            print("-" * 60)

            # Check for Null Values
            print("\nCheck if any Columns have Null Values:")
            print(self.isnull().sum())
            print("-" * 60)

            # Check for Duplicated Rows
            print("\nCheck for Duplicated Rows in Dataframe:")
            print(self.duplicated().sum())

        except Exception as e:
            print(f"An error occurred: {e}")

    # This method displays seperated numerical and categorical columns
    def identify_columns(self):
        num_cols = [col for col in self.columns if self[col].dtypes in ['int64', 'float64']] # numerical columns
        cat_cols = [col for col in self.columns if self[col].dtypes not in ['int64', 'float64']] # categorical columns

        # Output
        print(f"Numerical columns are: {num_cols}.")
        print(f"Categorical columns are: {cat_cols}.")

    # This method identifies numerical and categorical columns
    def column_analysis(self):
        # Identify categorical columns
        category_columns = [col for col in self.columns if self[col].dtype in ['object', 'category', 'bool']]

        # Identify numerical columns treated as categorical
        numerical_but_categorical = [col for col in self.columns if self[col].nunique() < 10 and self[col].dtype in ['int64', 'float64']]

        # Identify categorical columns with high cardinality
        category_with_hi_cardinality = [col for col in self.columns if self[col].nunique() > 50 and self[col].dtype in ['category', 'object']]

        # Filter out numerical_but_categorical from categorical_columns
        categorical_columns = [col for col in category_columns if col not in numerical_but_categorical]

        # Identify purely numerical columns
        numerical_columns = [col for col in self.columns if self[col].dtype in ['int64','float64']]
        numerical_columns = [col for col in numerical_columns if col not in category_columns]

        # print analysis
        print(f'Observations : {self.shape[0]}')
        print(f'Variables : {self.shape[1]}')
        print(f'Categorical Columns : {len(category_columns)}')
        print(f'Numerical Columns : {len(numerical_columns)}')
        print(f'Categorical Columns with High Cardinality : {len(category_with_hi_cardinality)}')
        print(f'Numerical Columns that are Categorical: {len(numerical_but_categorical)}')

        return category_columns, numerical_columns, category_with_hi_cardinality

    # Method provides summary of categorical column
    def categorical_column_summary(self, column_name, plot=False):
        # Calculate value counts and ratios
        value_counts = self[column_name].value_counts()
        ratios = round(100 * value_counts / len(self), 2)

        # Create a summary DataFrame
        summary_df = pd.DataFrame({column_name: value_counts, 'Ratio (%)': ratios})

        # Print the summary
        print(summary_df)
        print('-' * 40)

        if plot:
            # Check if the column is boolean
            if self[column_name].dtype == 'bool':
                sns.countplot(x=self[column_name].astype(int), data=self)
            else:
                sns.countplot(x=self[column_name], data=self)

            # Show the plot
            plt.show(block=True)

    # Method provides summary of numerical column
    def numerical_column_summary(self, column, plot=False):
        if self[column].dtype not in ['int64', 'float64']:
            raise ValueError(f"{column} is not a numerical datatype")
        quantiles = [0.05,0.10,0.20,0.30,0.40,0.50,0.60,0.70,0.80,0.90,0.95,0.99]
        summary_df = self[column].describe(quantiles).to_frame()
        print(summary_df)

        if plot:
            self[column].hist()
            plt.xlabel(column)
            plt.title(column)
            plt.show()
        print('-' * 40)

    # This method cross examines the relationship between categorical columns and target column that is numerical
    def target_cross_analysis_cat(self, target, cat_col):
        if self[target].dtype not in ['int64', 'float64']:
            raise ValueError(f"{target} column is not numerical")
        elif self[cat_col].dtype not in ['object', 'category', 'bool']:
            raise ValueError(f"{cat_col} column is not categorical")
        print(pd.DataFrame({'TARGET MEAN': self.groupby(cat_col)[target].mean().sort_values(ascending=False)}))

    # This method cross examines the relationship between numerical columns and target column regardless of target datatype
    def target_cross_analysis_num(self, target, num_col):
        if self[num_col].dtype not in ['int64', 'float64']:
            raise ValueError(f"{num_col} column is not numerical")
        result = self.groupby(target)[num_col].mean().sort_index(ascending=False)
        print(result)

    # Normalization of numerical column
    def normalize(self, col):
        if col not in self.columns:
            raise ValueError(f"'{col}' not a column in dataframe")
        if self[col].dtype not in ['int64', 'float64']:
            raise  ValueError(f"'{col}' column not numerical")
        self[col] = (self[col] - self[col].mean()) / self[col].std()

    # This method performs imputation an entire DataFrame regardless of datatype
    def complete_imputation(self):
        for col in self.columns:
            if self[col].dtype in ['int64', 'float64'] and self[col].isna().any():
                self[col].fillna(self[col].mean(), inplace=True) # fill numerical missing data with mean of column
            elif self[col].dtype in ['object', 'category', 'bool'] and self[col].isna().any():
                self[col].fillna(self[col].mode().iloc[0], inplace=True) # fill categorical missing data with mode of column

    # This method uses the inter quartile range method to identify and remove outliers in column
    def remove_outlier_iqr(self, column):
        if self[column].dtype not in ['int64', 'float64']:
            raise ValueError(f'[{column}] not a numerical datatype')

        Q1 = np.percentile(self[column], 25, interpolation='midpoint')
        Q3 = np.percentile(self[column], 75, interpolation='midpoint')
        IQR = Q3 - Q1

        # Calculate the bounds for outliers
        upper_bound = Q3 + 1.5 * IQR
        lower_bound = Q1 - 1.5 * IQR

        # Identify the outliers
        #outliers = (self[column] >= upper_bound) | (self[column] <= lower_bound)
        outliers = self[(self[column] < lower_bound) | (self[column] > upper_bound)]

        # Remove the outliers
        self.drop(outliers.index, inplace=True)

    # This method produces the upper and lower bound rows of a dataframe using the IQR method of a given column
    def outlier_limits_iqr(self, column):
        if self[column].dtype not in ['int64', 'float64']:
            raise ValueError(f'[{column}] not a numerical datatype')

        Q1, Q3 = self[column].quantile([0.25, 0.75])
        IQR = Q3 - Q1
        upper_limit = Q3 + 1.5 * IQR
        lower_limit = Q1 - 1.5 * IQR
        return self[(self[column] < lower_limit) | (self[column] > upper_limit)]

    # This method displays all outliers in dataframe using the IQR method
    def show_outlier_rows(self):
        num_cols = [col for col in self.columns if self[col].dtype in ['int64', 'float64']]

        for col in num_cols:
            print('-'*40, col, '-'*40)
            print(self.outlier_limits_iqr(col))

    # This method type casts an object datatype into a category datatype
    def category_datatype(self):
        self[self.select_dtypes(include=['object']).columns] = self.select_dtypes(include=['object']).astype('category')

    # This method replaces values in a DataFrame that represent an unknown value but are not recorded as null (e.g. -, ?, *)
    def turn_null(self, val):
        self[self.columns] = self.apply(lambda col: col.replace({val: np.nan}))

    # This method outputs the percentage of null values in each column of dataframe
    def null_percent(self):
        print(self.isnull().mean().round(4).mul(100).sort_values(ascending=False))

    # Drop columns in a DataFrame with a certian percentage of null values
    def drop_null_by_percent(self, percent):
        min_count = int(((100-percent)/100)*self.shape[0]+1)
        self.dropna(axis=1, thresh=min_count, inplace=True)

    # This method type casts an object datatype into a boolean datatype
    def bool_datatype(self, column, true_value, false_value):
        """
        This method type casts an object datatype into a boolean datatype.

        Parameters:
        - column (str): The column name in the DataFrame.
        - true_value: The value in the column to be considered as True.
        - false_value: The value in the column to be considered as False.
        """
        if not isinstance(column, str):
            raise ValueError(f'[{column}] parameter is not a string datatype')

        if not all(self[column].isin([true_value, false_value])):
            raise ValueError('One or more values not in column')

        encoded_values = {true_value: True, false_value: False}
        self[column] = self[column].map(encoded_values).astype(bool)

    # This method print a dictionary with the unique values of a column and the number of occurences
    def counter(self, column):
        # Check if the column exists in the DataFrame
        if column not in self.columns:
            raise ValueError(f"Column '{column}' not found in the DataFrame")

        # Use Counter to count occurrences of unique values in the column
        counts = dict(Counter(self[column]))

        # Print the dictionary
        print(counts)

""" New class for graphing values of dataset"""
class Graphs ():
    def __init__(self, df, style='ggplot'):
        self.df = df
        self.style = style

    """ Single Visualization Graphs """
    # This method returns seaborn histogram
    def histogram(self, column):
        with plt.style.context(self.style):
            fig, ax = plt.subplots(figsize=(17, 8))
            sns.histplot(data=self.df, x = self.df[column], kde = True, ax=ax)
            plt.grid(False)
            ax.set_title(f"{column.title().replace('_', ' ')} Histogram")

        # Display the plot
        plt.show()

    # This method returns seaborn boxplot
    def categorical_boxplot(self, categorical_column, numerical_column):
        # Check data types
        cat_dtype = self.df[categorical_column].dtype
        num_dtype = self.df[numerical_column].dtype

        if cat_dtype not in ['object', 'category', 'bool']:
            raise ValueError(f'[{categorical_column}] not a categorical datatype')
        elif num_dtype not in ['int64', 'float64']:
            raise ValueError(f'[{numerical_column}] not a numerical datatype')

        # Plotting
        with plt.style.context(self.style):
            fig, ax = plt.subplots(figsize=(17, 8))
            sns.boxplot(data=self.df, x=categorical_column, y=numerical_column, ax=ax)
            ax.set_title(f"{numerical_column.title().replace('_', ' ')} by {categorical_column.title().replace('_', ' ')} boxplot")

            # Calculate medians and observations
            medians = self.df.groupby(categorical_column)[numerical_column].median()
            obs = self.df[categorical_column].value_counts().reindex(medians.index).values

            # Add observations to the plot
            pos = range(len(obs))
            for tick, label in zip(pos, ax.get_xticklabels()):
                ax.text(pos[tick], medians.iloc[tick] + 0.03, f'n: {obs[tick]}',
                        ha='center', size='large', color='black', weight='semibold')

            ax.grid(False)

        # Display the plot
        plt.show()

    # This method returns seaborn boxplot with color encoding from a certian column
    def categorical_boxplot_with_hue(self, categorical_column, numerical_column, hue_column):
        # Validate input data types
        cat_dtype = self.df[categorical_column].dtype
        num_dtype = self.df[numerical_column].dtype
        hue_dtype = self.df[hue_column].dtype

        valid_cat_types = ['object', 'category', 'bool']
        valid_num_types = ['int64', 'float64']

        if cat_dtype not in valid_cat_types:
            raise ValueError(f'[{categorical_column}] not a categorical datatype')
        elif num_dtype not in valid_num_types:
            raise ValueError(f'[{numerical_column}] not a numerical datatype')
        elif hue_dtype not in valid_cat_types:
            raise ValueError(f'[{hue_column}] not a categorical datatype')

        # Plotting
        with plt.style.context(self.style):
            fig, ax = plt.subplots(figsize=(17, 8))
            sns.boxplot(data=self.df, x=categorical_column, y=numerical_column, hue=hue_column, showmeans=True, ax=ax)
            ax.set_title(f"{numerical_column.title().replace('_', ' ')} by {categorical_column.title().replace('_', ' ')} with {hue_column.title().replace('_', ' ')} grouping boxplot")
            plt.grid(False)

        # Display the plot
        plt.show()

    # This method returns seaborn bivariate barplot with color encoding from a certian column if desired
    def categorical_barplot(self, cat_column, num_column, hue_col=None):
        # Validate input data types
        cat_dtype = self.df[cat_column].dtype
        num_dtype = self.df[num_column].dtype
        hue_dtype = self.df[hue_col].dtype if hue_col else None

        valid_cat_types = ['object', 'category', 'bool']
        valid_num_types = ['int64', 'float64']

        if cat_dtype not in valid_cat_types:
            raise ValueError(f'[{cat_column}] not a categorical datatype')
        elif num_dtype not in valid_num_types:
            raise ValueError(f'[{num_column}] not a numerical datatype')
        elif hue_col and hue_dtype not in valid_cat_types:
            raise ValueError(f'[{hue_col}] not a categorical datatype')

        with plt.style.context(self.style):
            # Create the barplot
            fig, ax = plt.subplots(figsize=(17, 8))
            if limit is not None and type(limit) == int:
                order = self.df.groupby(cat_column).mean(numeric_only=True).sort_values(num_column, ascending=False).iloc[:limit].index
            else:
                order = self.df.groupby(cat_column).mean(numeric_only=True).sort_values(num_column, ascending=False).index
            sns.barplot(data=self.df, x=cat_column, y=num_column, hue=hue_col, order=order, errwidth=0, ax=ax)

            # Customize plot
            ax.grid(False)
            if limit is not None:
                title = f"Average {num_column.replace('_', ' ').title()} by {cat_column.replace('_', ' ').title()} Barplot [Top {limit}]"
            else:
                title = f"Average {num_column.replace('_', ' ').title()} by {cat_column.replace('_', ' ').title()} Barplot"
            if hue_col:
                title += f" with {hue_col.replace('_', ' ').title()} Grouping"
            ax.set_title(title)

            # Annotate the bars with the numeric values
            for p in ax.patches:
                ax.annotate(format(p.get_height(), '.1f'),
                            (p.get_x() + p.get_width() / 2, p.get_height()),
                            ha='center', va='center',
                            size=15, xytext=(0, 8),
                            textcoords='offset points')

        # Display the plot
        plt.show()

    # This method returns seaborn scatterplot with color encoding from a certian column if desired
    def scatterplot(self, num_col1, num_col2, hue_col=None):
        # Validate input data types
        num_dtype1 = self.df[num_col1].dtype
        num_dtype2 = self.df[num_col2].dtype
        hue_dtype = self.df[hue_col].dtype if hue_col else None

        valid_cat_types = ['object', 'category', 'bool']
        valid_num_types = ['int64', 'float64']

        if num_dtype1 not in valid_num_types:
            raise ValueError(f'[{num_col1}] not a numerical datatype')
        elif num_dtype2 not in valid_num_types:
            raise ValueError(f'[{num_col2}] not a numerical datatype')
        elif hue_col and hue_dtype not in valid_cat_types:
            raise ValueError(f'[{hue_col}] not a categorical datatype')

        with plt.style.context(self.style):
            # Create the barplot
            fig, ax = plt.subplots(figsize=(17, 8))
            sns.scatterplot(data=self.df, x=num_col1, y=num_col2, hue=hue_col, ax=ax)

            # Customize plot
            if hue_col:
                ax.legend(loc='upper center')
            ax.grid(False)
            title = f"{num_col1.replace('_', ' ').title()} vs {num_col2.replace('_', ' ').title()} Scatterplot"
            if hue_col:
                title += f" with {hue_col.replace('_', ' ').title()} Grouping"
            ax.set_title(title)

        # Display the plot
        plt.show()

    # This method returns seaborn joiintplot with regression line
    def jointplot(self, num_col1, num_col2):
        # Check if columns have numerical data types
        numerical_types = ['int64', 'float64']
        if self.df[num_col1].dtype not in numerical_types:
            raise ValueError(f'Column [{num_col1}] is not a numerical data type')
        elif self.df[num_col2].dtype not in numerical_types:
            raise ValueError(f'Column [{num_col2}] is not a numerical data type')

        # Plotting
        with plt.style.context(self.style):
            g = sns.jointplot(data=self.df, x=num_col1, y=num_col2, kind='reg')

            # Calculate and display Pearson correlation coefficient
            r, p = stats.pearsonr(self.df[num_col1].values, self.df[num_col2].values)
            annotation_text = f'$\\rho = {r:.3f}$'
            g.ax_joint.annotate(annotation_text, xy=(0.1, 0.9), xycoords='axes fraction',
                                ha='left', va='center', bbox={'boxstyle': 'round'}, fontsize=20)

            # Scatter plot on the joint axis
            g.ax_joint.scatter(self.df[num_col1], self.df[num_col2])
            g.fig.set_figwidth(17)
            g.fig.set_figheight(8)

            # Set axis labels and adjust layout
            g.set_axis_labels(xlabel=num_col1, ylabel=num_col2, size=15)
            plt.tight_layout()

            # Remove grid
            plt.grid(False)

        # Display the plot
        plt.show()

    # This method returns seaborn heatmap
    def list_heatmap(self, columns):
        if not isinstance(columns, list):
            raise ValueError('Parameter must be a list')
        for col in columns:
            if self.df[col].dtype not in ['int64', 'float64']:
                raise ValueError(f'{col} is not a numerical datatype')
        with plt.style.context(self.style):
            fig, ax = plt.subplots(figsize=(17, 8))

            selected_columns = self.df[columns]
            correlation_matrix = selected_columns.corr()

            sns.heatmap(correlation_matrix, annot=True, cmap='winter', ax=ax)
            ax.set_title(f"Heatmap of {columns}")

        # Display the plot
        plt.show()

    # This method returns seaborn univariate barplot with grouping if desired
    def countplot(self, column, hue_col=None, limit=None):
        if column not in self.df.columns:
            raise ValueError(f'Column "{column}" not in the dataframe')
        elif hue_col not in self.df.columns and hue_col is not None:
            raise ValueError(f'Column "{hue_col}" not in the dataframe')

        with plt.style.context(self.style):
            fig, ax = plt.subplots(figsize=(17, 8))
            if limit is not None and type(limit) == int:
                order = self.df[column].value_counts(normalize=True).iloc[:limit].index
            else:
                order = self.df[column].value_counts(normalize=True).index
            sns.countplot(data=self.df, x=column, hue=hue_col, order=order, ax=ax)

            ax.grid(False)
            if limit is not None:
                title = f"{column.title().replace('_', ' ')} Countplot [Top {limit}]"
            else:
                title = f"{column.title().replace('_', ' ')} Countplot"
            if hue_col is not None:
                title += f' with {hue_col.title().replace("_", " ")} Categories'
            ax.set_title(title)

            total = len(self.df[column])
            for p in ax.patches:
                percentage = f'{100 * p.get_height() / total:.1f}%\n'
                x = p.get_x() + p.get_width() / 2
                y = p.get_height()
                ax.annotate(percentage, (x, y), ha='center', va='center', fontsize=11)

        # Display the plot
        plt.show()

    # This method returns seaborn line graph with color encoding from a certian column if desired
    def lineplot(self, x_column, y_column, hue_column=None, errors=None):
        # Validate input data types
        valid_cat_types = ['object', 'category', 'bool']

        if self.df[x_column].dtype not in valid_cat_types:
            raise ValueError(f'[{x_column}] not a categorical datatype')
        elif self.df[y_column].dtype not in ['int64', 'float64']:
            raise ValueError(f'[{y_column}] not a numerical datatype')
        elif hue_column is not None and self.df[hue_column].dtype not in valid_cat_types:
            raise ValueError(f'[{hue_column}] not a categorical datatype')

        # plot graph
        with plt.style.context(self.style):
                fig, ax = plt.subplots(figsize=(17, 8))
                sns.lineplot(data = self.df, x = x_column, y = y_column, hue = hue_column, errorbar = errors, ax=ax)
                ax.grid(False)

                title = f"{x_column.title().replace('_', ' ')} vs {y_column.title().replace('_', ' ')} lineplot"
                if hue_column is not None:
                    title += f' with {hue_column.title().replace("_", " ")} Categories'
                ax.set_title(title)

        # Display the plot
        plt.show()

    # This method returns a pie chart
    def pie_chart(self, column):
        # Check if the column is a categorical datatype
        if self.df[column].dtype not in ['object','category','bool']:
            raise ValueError(f'[{column}] not a categorical datatype')

        # Create the pie chart
        with plt.style.context(self.style):
            sorted_counts = self.df[column].value_counts()
            fig, ax = plt.subplots(figsize=(8,8))

            # Customize the pie chart appearance
            wedges_props = {'linewidth': 3, 'edgecolor': 'white'}
            ax.pie(sorted_counts, labels= sorted_counts.index, startangle=90, counterclock=False,
                    autopct='%1.1f%%', textprops={'fontsize':15}, wedgeprops=wedges_props)

            # Set equal aspect ratio and title
            ax.axis('equal')
            ax.set_title(f"{column.title().replace('_', ' ')} Pie Chart")

            # Adjust layout and display the chart
            plt.tight_layout()

        # Display the plot
        plt.show()

    # This method returns a donut pie chart
    def donut_pie_chart(self, column):
        # Check if the column is a categorical data type
        if self.df[column].dtype not in ['object','category','bool']:
            raise ValueError(f'[{column}] not a categorical datatype')

        # Set the style for the plot
        with plt.style.context(self.style):

            # Create a figure and axis with a specified size
            fig, ax = plt.subplots(figsize=(12,12))

            # Create a circle at the center of the plot
            my_circle = plt.Circle((0, 0), 0.7, color='black')

            # Get sorted value counts of the categorical column
            sorted_counts = self.df[column].value_counts()

            # Set properties for the wedges in the pie chart
            wedge_props = {'width':0.5, 'linewidth':7, 'edgecolor':'black'}

            # Plot the donut chart
            ax.pie(sorted_counts, labels=sorted_counts.index, startangle=90,
                    counterclock=False, wedgeprops=wedge_props,
                    autopct='%1.1f%%', pctdistance=0.85, textprops={'fontsize':14})

            # Add the circle to create the donut effect
            p = plt.gcf()
            p.gca().add_artist(my_circle)

            # Set the title of the chart
            ax.set_title(f"{column.title().replace('_', ' ')} Donut Chart")

        # Display the plot
        plt.show()

    # This method shows seaborn violinplot
    def violinplot(self, cat_col, num_col):
        # Check if the categorical column is of the right datatype
        if self.df[cat_col].dtype not in ['object', 'category', 'bool']:
            raise ValueError(f'[{cat_col}] is not a categorical datatype')

        # Check if the numerical column is of the right datatype
        elif self.df[num_col].dtype not in ['int64', 'float64']:
            raise ValueError(f'[{num_col}] is not a numerical datatype')

        # Set the plotting style
        with plt.style.context(self.style):
            # Create a figure and axis for the violin plot
            fig, ax = plt.subplots(figsize=(17, 8))

            # Create the violin plot
            sns.violinplot(data=self.df, x=cat_col, y=num_col, ax=ax)

            # Calculate medians and observations for each group
            medians = self.df.groupby([cat_col])[num_col].median()
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
    def violinplot_with_hue(self, cat_col, num_col, hue_col):
        # validate catergoical columns datatypes
        valid_cat_dtype = ['object','category','bool']

        # Check if the categorical column is of the right datatype
        if self.df[cat_col].dtype not in valid_cat_dtype:
            raise ValueError(f'[{cat_col}] not a categorical datatype')

        # Check if the numerical column is of the right datatype
        elif self.df[num_col].dtype not in ['int64', 'float64']:
            raise ValueError(f'[{num_col}] not a numerical datatype')

        # Check if the hue column is of the right datatype
        elif self.df[hue_col].dtype not in valid_cat_dtype:
            raise ValueError(f'[{hue_col}] not a categorical datatype')

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
    def circular_barplot(self, cat_col, num_col, bar_color):
        # Check if the categorical column is of the right datatype
        if self.df[cat_col].dtype not in ['object','category','bool']:
            raise ValueError(f'[{cat_col}] not a categorical datatype')

        # Check if the numerical column is of the right datatype
        elif self.df[num_col].dtype not in ['int64', 'float64']:
            raise ValueError(f'[{num_col}] not a numerical datatype')

        # Set plotting style
        with plt.style.context(self.style):

            # Reorder dataframe
            df_grouped = self.df.groupby([cat_col])[num_col].mean(numeric_only=True).round(0).reset_index()
            df_sorted = df_grouped.sort_values(by=[num_col])

            pal = list(sns.color_palette(palette=bar_color, n_colors=len(df_sorted)).as_hex())

            # initialize figure
            plt.figure(figsize=(20,10))
            ax = plt.subplot(111, polar=True)
            plt.axis('off')

            # constraints = parameters controling plot layout
            #upper_limit = 100
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
            width = 2*np.pi / len(df_sorted.index)

            # compute the angle each bar is centered on:
            indexes = list(range(1, len(df_sorted.index) + 1))
            angles = [item * width for item in indexes]
            angles

            # draw bars
            bars = ax.bar(x=angles, height=heights, width=width, bottom=lower_limit, linewidth=2,
                            edgecolor='white', color=pal)

            # Add labels
            for bar, angle, height, label in zip(bars, angles, heights, df_sorted[cat_col]):
                # labels are rotated. rotation must be specified in degrees
                rotation = np.rad2deg(angle)

                # flip some labels upside down
                alignment = ""
                if angle >= np.pi/2 and angle < 3*np.pi/2:
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
    def race_track_plot(self, cat_col, num_col, bar_color):
        # Check if the categorical column is of the right datatype
        if self.df[cat_col].dtype not in ['object', 'category', 'bool']:
            raise ValueError(f'[{cat_col}] not a categorical datatype')

        # Check if the numerical column is of the right datatype
        elif self.df[num_col].dtype not in ['int64', 'float64']:
            raise ValueError(f'[{num_col}] not a numerical datatype')

        # Set plotting theme
        with plt.style.context(self.style):
            # Reorder dataframe
            df_grouped = self.df.groupby([cat_col])[num_col].mean().round(0).reset_index()
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
    def treemap(self, cat_col, num_col, color_scale):
        # Check if the categorical column is of the right datatype
        if self.df[cat_col].dtype not in ['object', 'category', 'bool']:
            raise ValueError(f'[{cat_col}] not a categorical datatype')

        # Check if the numerical column is of the right datatype
        elif self.df[num_col].dtype not in ['int64', 'float64']:
            raise ValueError(f'[{num_col}] not a numerical datatype')

        df_grouped = self.df.groupby([cat_col])[num_col].mean(numeric_only=True).round().reset_index()
        fig = px.treemap(df_grouped, path=[px.Constant(f'{cat_col.title()} Categories'), cat_col],
                            values=df_grouped[num_col],
                            color=df_grouped[num_col],
                            color_continuous_scale=color_scale,
                            color_continuous_midpoint=np.average(df_grouped[num_col]),
                            title= f"Treemap of {cat_col.title()} Categories and the Average {num_col.title().replace('_', ' ')} for Each Category."
                            )
        fig.update_layout(margin= dict(t=50, l=25, r=25, b=25), title_x=0.5)
        fig.show()
        plt.show()

    # This method produces interactive pie chart with percentages
    def percentage_pie_chart(self, cat_col, num_col, bar_col):
        # Check if the categorical column is of the right datatype
        if self.df[cat_col].dtype not in ['object', 'category', 'bool']:
            raise ValueError(f'[{cat_col}] not a categorical datatype')

        # Check if the numerical column is of the right datatype
        elif self.df[num_col].dtype not in ['int64', 'float64']:
            raise ValueError(f'[{num_col}] not a numerical datatype')

        # Group by and calculate mean
        df_grouped = self.df.groupby([cat_col])[num_col].mean(numeric_only=True).round(2).reset_index()

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
    def interactive_bar_chart(self, cat_col, num_col, bar_col):
        # Check if the categorical column is of the right datatype
        if self.df[cat_col].dtype not in ['object', 'category', 'bool']:
            raise ValueError(f'[{cat_col}] not a categorical datatype')

        # Check if the numerical column is of the right datatype
        elif self.df[num_col].dtype not in ['int64', 'float64']:
            raise ValueError(f'[{num_col}] not a numerical datatype')

        # Group by and calculate mean
        df_grouped = self.df.groupby([cat_col])[num_col].mean(numeric_only=True).round(2).reset_index()

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
    def polar_line_chart(self, cat_col, num_col, bar_col):
        # Check if the categorical column is of the right datatype
        if self.df[cat_col].dtype not in ['object', 'category', 'bool']:
            raise ValueError(f'[{cat_col}] not a categorical datatype')

        # Check if the numerical column is of the right datatype
        elif self.df[num_col].dtype not in ['int64', 'float64']:
            raise ValueError(f'[{num_col}] not a numerical datatype')

        # Group by and calculate mean
        df_grouped = self.df.groupby([cat_col])[num_col].mean(numeric_only=True).round(2).reset_index()

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
    def circular_bubble_chart(self, cat_col, num_col, bar_col):
        # Check if the categorical column is of the right datatype
        if self.df[cat_col].dtype not in ['object', 'category', 'bool']:
            raise ValueError(f'[{cat_col}] not a categorical datatype')

        # Check if the numerical column is of the right datatype
        elif self.df[num_col].dtype not in ['int64', 'float64']:
            raise ValueError(f'[{num_col}] not a numerical datatype')

        # Group by and calculate mean
        df_grouped = self.df.groupby([cat_col])[num_col].mean(numeric_only=True).round(2).reset_index()

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
    def regression_subplots(self, cat_col, num_col1, num_col2, sub_1, sub_2):
        # Validate input data type
        valid_cat_types = ['object', 'category', 'bool']
        valid_num_types = ['int64', 'float64']

        # Check if the categorical column is of the right datatype
        if self.df[cat_col].dtype not in valid_cat_types:
            raise ValueError(f'[{cat_col}] is not a categorical datatype')

        # Check if the numerical columns are of the right datatype
        for num_col in [num_col1, num_col2]:
            if self.df[num_col].dtype not in valid_num_types:
                raise ValueError(f'[{num_col}] is not a numerical datatype')

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
    def histogram_subplots(self, sub_1, sub_2):
        # Identify all the columns that are numerical
        num_cols = [col for col in self.df.columns if self.df[col].dtype in ['int64', 'float64']]

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
    def cat_count_subplots(self, sub_1, sub_2, limit=None):
        # Identify all the columns that are categorical
        cat_cols = [col for col in self.df.columns if self.df[col].dtype not in ['int64', 'float64']]

        # Initialize figure
        fig, axes = plt.subplots(sub_1, sub_2, figsize=(20, 20))
        axes = axes.flatten()

        # Iterate columns and plot corresponding data
        for i, col in enumerate(cat_cols):

            # Set plotting style
            with plt.style.context('ggplot'):

                # Seaborn count plot
                if limit is not None and type(limit) == int:
                    order = self.df[col].value_counts(normalize=True).iloc[:limit].index
                else:
                    order = self.df[col].value_counts(normalize=True).index
                sns.countplot(data=self.df, x=col, order=order, ax=axes[i])
                axes[i].set_xlabel(col)

                # Annotate each bar with its percentage
                total = len(self.df[col])
                for p in axes[i].patches:
                    percentage = '{:.1f}%'.format(100 * p.get_height() / total)
                    x = p.get_x() + p.get_width() / 2
                    y = p.get_height()
                    axes[i].annotate(percentage, (x, y), ha='center', va='bottom')

        # Remove extra subplots if there are more axes than needed
        for j in range(i + 1, len(axes)):
            fig.delaxes(axes[j])

        # Display the plot
        plt.show()

    # This method return subplots of scatter plots
    def scatter_subplots(self, num_col, sub_1, sub_2, hue_col=None):
        # Check if the numerical column is of the right datatype
        if self.df[num_col].dtype not in ['int64', 'float64']:
            raise ValueError(f'[{num_col}] not a numerical datatype')

        # Check if the categorical 'hue' column is of the right datatype
        elif hue_col and self.df[hue_col].dtype not in ['object', 'category', 'bool']:
            raise ValueError(f'[{hue_col}] not a categorical datatype')

        # Identify all the columns that are numerical
        num_cols = [col for col, dtype in zip(self.df.columns, self.df.dtypes)
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
    def box_subplots(self, sub_1, sub_2):
        # Identify all the columns that are numerical
        num_cols = [col for col in self.df.columns if self.df[col].dtype in ['int64', 'float64']]

        # Initialize figure
        fig, axes = plt.subplots(sub_1, sub_2, figsize=(20, 20))
        axes = axes.flatten()

        # Iterate columns and plot corresponding data
        for i, col in enumerate(num_cols):

            # Set plotting style
            with plt.style.context(self.style):

                # Seaborn boxplot
                sns.boxplot(x=self.df[col], ax=axes[i])
                axes[i].set_xlabel(col)

        # Remove extra subplots if there are more axes than needed
        for j in range(len(num_cols), len(axes)):
            fig.delaxes(axes[j])

        # Display the plot
        plt.show()

    # This method returns subplots of barplots
    def bar_subplots(self, cat_col, sub_1, sub_2, limit=None):
        # Check if the categorical column has a valid datatype
        if self.df[cat_col].dtype not in ['object', 'category', 'bool']:
            raise ValueError(f'[{cat_col}] not a valid categorical datatype')

        # Identify all the columns that are numerical
        num_cols = self.df.select_dtypes(include=['int64', 'float64']).columns

        # Set plotting style
        with plt.style.context(self.style):

            # Initialize figure
            fig, axes = plt.subplots(sub_1, sub_2, figsize=(20, 20))
            axes = axes.flatten()

            # Iterate columns and plot corresponding data
            for j, ax in zip(num_cols, axes):

                # Use seaborn barplot
                if limit is not None and type(limit) == int:
                    order = self.df.groupby(cat_col).mean(numeric_only=True).sort_values(j, ascending=False).iloc[:limit].index
                else:
                    order = self.df.groupby(cat_col).mean(numeric_only=True).sort_values(j, ascending=False).index
                sns.barplot(x=cat_col, y=j, order=order, palette='magma', errwidth=0, data=self.df, ax=ax)

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
