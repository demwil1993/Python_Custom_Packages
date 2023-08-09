import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from collections import Counter
import math
import plotly.express as px
import scipy.stats as stats

class wrangler(pd.DataFrame):
    """ Class variable tells pandas the names of each attributes
     that is to be imported over to the derivative DataFrame.
     The __finalize__ method is used to get attributes and assign
     them to newly created class. """
    _metadata = ['my_attr']

    @property
    # a _constructor is a private member function that returns an empty DataFrame object.
    def _constructor(self):
        """ Returns the class name of same type but there are some cases
         where __finalize__ is not called the the attributes of pandas are not
         imported over. to fix this the function below calls __finalize__ every excecution. """
        def _c(*args, **kwargs):
            return wrangler(*args, **kwargs).__finalize__(self)
        return _c

    # called every time an object is created from a class, lets class initialize the object's attributes
    def __init__(self, *args, **kwargs):
        # grab the keyword argument that is supposed to be my_attr
        self.my_attr = kwargs.pop('my_attr', None)
        super().__init__(*args, **kwargs) # Return a proxy object which represents the parent’s class

    # This method strips leading and trailing white spaces from DataFrame columns and column values
    def strip_dataframe(self):
        self.columns = self.columns.str.strip()
        for x in self.columns:
            if str(self[x].dtypes) in ['object', 'category']:
                self[x] = self[x].str.strip()
            else:
                continue
    # This method provides summary statistics for categorical and numerical data columns
    def dataframe_anaylsis(self):
        print("Descriptive Statistics of Numerical Data:", end="\n\n")
        print(self.describe().T, end="\n--------------------------------------------------------------\n")
        print("Descriptive Statistics of Categorical Data:", end="\n\n")
        for x in self.columns:
            if str(self[x].dtypes) not in ['object', 'category', 'bool']:
                continue
            else:
                print(self.describe(include="O").T, end="\n--------------------------------------------------------------\n")
                break
        print("Check if any Columns have Null Values:", end="\n\n")
        print(self.isnull().sum(), end="\n--------------------------------------------------------------\n")
        print("Check for Duplicated Rows in Dataframe:", end="\n\n")
        print(self.duplicated().sum())

    # This method displays seperated numerical and categorical columns
    def identify_columns(self):
        num_cols = []
        cat_cols = []

        for i in self.columns:
            if str(self[i].dtypes) in ['int64','float64']:
                num_cols.append(i)
            else:
                cat_cols.append(i)
        print(f"Numerical columns are: {num_cols}.", end="\n\n")
        print(f"Categorical columns are: {cat_cols}.")

    # This method identifies numerical and categorical columns
    def column_analysis(self):
        category_columns = [col for col in self.columns if str(self[col].dtypes) in ['object', 'category', 'bool']]
        numerical_but_categorical = [col for col in self.columns if self[col].nunique() < 10 and str(self[col].dtypes) in ['int64', 'float64']]
        category_with_hi_cardinality = [col for col in self.columns if self[col].nunique() > 50 and str(self[col].dtypes) in ['category', 'object']]
        category_columns = category_columns + numerical_but_categorical
        category_columns = [col for col in category_columns if col not in category_with_hi_cardinality]

        numerical_columns = [col for col in self.columns if (self[col].dtypes) in ['int64','float64']]
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
    def categorical_column_summary(self, data, column_name, plot=False):
        print(pd.DataFrame({column_name:data[column_name].value_counts(),
                            'Ratio':round(100*data[column_name].value_counts() / len(data), 2)}))
        print('-'*40)
        if plot:
            if str(self[column_name].dtypes) == 'bool':
                sns.countplot(x=self[column_name].astype(int), data=data)
                plt.show(block=True)
            else:
                sns.countplot(x=data[column_name],data=data)
                plt.show(block=True)

    # Method provides summary of numerical column
    def numerical_column_summary(self, data, column, plot=False):
        quantiles = [0.05,0.10,0.20,0.30,0.40,0.50,0.60,0.70,0.80,0.90,0.95,0.99]
        print(pd.DataFrame(data[column].describe(quantiles)))
        if plot:
            data[column].hist()
            plt.xlabel(column)
            plt.title(column)
            plt.show()
        print("____________________________________________________")

    # This method cross examines the relationship between categorical columns and target column
    def target_cross_analysis_cat(self, target, cat_col):
        print(pd.DataFrame({'TARGET MEAN': self.groupby(cat_col)[target].mean().sort_values(ascending=False)}))

    # This method cross examines the relationship between numerical columns and target column
    def target_cross_analysis_num(self, target, num_col):
        print(self.groupby(target).agg({num_col: 'mean'}).sort_values(by=target, ascending=False))

    # Normalization of numerical column
    def normalize(self, col):
        if col not in self.columns:
            raise TypeError(f"'{col}' not a column in dataframe")
        elif str(self[x].dtypes) not in ['int64', 'float64']:
            raise  TypeError(f"'{col}' column not numerical")
        else:
            self[col] = (self[col] - self[col].mean()) / self[col].std()

    # This method performs imputation an entire DataFrame regardless of datatype
    def complete_imputation(self):
        for x in self.columns:
            if str(self[x].dtypes) in ['int64', 'float64'] and (self[x].isna().any() == True):
                mean = self[x].mean()
                self[x].fillna(mean, inplace=True) # fill numerical missing data with mean of column
            elif str(self[x].dtypes) in ['object', 'category', 'bool'] and (self[x].isna().any() == True):
                mode = self[x].mode().iloc[0]
                self[x].fillna(mode, inplace=True) # fill categorical missing data with mode of column

    # This method uses the inter quartile range method to identify and remove outliers in column
    def remove_outlier_iqr(self, column):
        if str(self[column].dtypes) not in ['int64', 'float64']:
            raise TypeError(f'[{column}] not a numerical datatype')
        else:
            """Detection"""
            Q1 = np.percentile(self[column], 25, interpolation='midpoint')
            Q3 = np.percentile(self[column], 75, interpolation='midpoint')
            IQR = Q3 - Q1

            # Upper bound
            upper = np.where(self[column] >= (Q3 + 1.5 * IQR))
            # Lower bound
            lower = np.where(self[column] <= (Q1 - 1.5 * IQR))

            """Removing the Outliers"""
            self.drop(upper[0], inplace=True)
            self.drop(lower[0], inplace=True)

    # This method produces the upper and lower bound of the IQR method of a given column
    def outlier_limits_iqr(self, column):
        if str(self[column].dtypes) not in ['int64', 'float64']:
            raise TypeError(f'[{column}] not a numerical datatype')
        else:
            Q1 = self[column].quantile(0.25)
            Q3 = self[column].quantile(0.75)
            IQR = Q3 - Q1
            upper_limit = Q3 + 1.5 * IQR
            lower_limit = Q1 - 1.5 * IQR
            return self[(self[column] < lower_limit) | (self[column] > upper_limit)]

    # This method displays outliers using the IQR method
    def show_outlier_rows(self):
        num_cols = []
        for x in self.columns:
            if str(self[x].dtypes) not in ['int64', 'float64']:
                continue
            else:
                num_cols.append(x)

        for col in num_cols:
            print('-'*40, col, '-'*40)
            print(self.outlier_limits_iqr(col))

    # This method type casts an object datatype into a category datatype
    def category_datatype(self):
        for x in self.columns:
            if self[x].dtype == 'object':
                self[x] = self[x].astype('category')

    # This method replaces values in a DataFrame that represent an unknown value but are not recorded as null (e.g. -, ?, *)
    def turn_null(self, val):
        for col in self.columns:
            self[col] = self[col].replace({val:np.nan})

    # This method outputs the percentage of null values in each column of dataframe
    def null_percent(self, data):
        data.isnull().mean().round(4).mul(100).sort_values(ascending=False)

    # Drop columns in a DataFrame with a certian percentage of null values
    def drop_null_by_percent(self, data, percent):
        percentage = percent
        min_count = int(((100-percentage)/100)*data.shape[0]+1)
        data.dropna(axis=1, thresh=min_count, inplace=True)

    # This method type casts an object datatype into a boolean datatype
    def bool_datatype(self, column, true_value, false_value):
        if type(column) != str:
            raise TypeError(f'[{column}] parameter is not a string datatype')
        elif ~self[column].isin([true_value, false_value]).all():
            raise TypeError('One or more values not in column')
        else:
            encoded_values = {true_value : 1, false_value : 0}
            self[column].replace(encoded_values, inplace=True)
            self[column] = self[column].astype('bool')

    # This method print a dictionary with the unique values of a column and the number of occurences
    def counter(self, column):
        if type(column) != str and type(column) != bool:
            raise TypeError(f'[{column}] parameter is not a string datatype')
        else:
            print(dict(Counter(self[column])))

""" New class for graphing values of dataset"""
class graphs ():
    def __init__(self):
        pass

    """ Single Visualization Graphs """
    # This method returns seaborn histogram
    def histogram(self, df, column, style):
        if type(column) != str:
            raise TypeError(f'[{column}] parameter is not a string datatype')
        else:
            with plt.style.context(style):
                fig = plt.figure(figsize=(17, 8))
                sns.histplot(data=df, x = df[column], kde = True)
                plt.grid(False)
                plt.title(f"{column.title().replace('_', ' ')} Histogram")
            plt.show()

    # This method returns seaborn boxplot
    def categorical_boxplot(self, df, categorical_column, numerical_column, style):
        if str(df[categorical_column].dtypes) not in ['object', 'category', 'bool']:
            raise TypeError(f'[{categorical_column}] not a categorical datatype')
        elif str(df[numerical_column].dtypes) not in ['int64', 'float64']:
            raise TypeError(f'[{numerical_column}] not a numerical datatype')
        else:
            with plt.style.context(style):
                fig = plt.figure(figsize=(17, 8))
                ax = sns.boxplot(data= df, x = categorical_column, y = numerical_column)

                # calculate the number of obserations for each group and median to position labels
                medians = df.groupby([categorical_column])[numerical_column].median()
                obs = df[categorical_column].value_counts().reindex(medians.index).values
                medians = medians.values
                obs = [str(x) for x in obs.tolist()]
                obs = ['n: '+i for i in obs]

                # Add observations to plot
                pos = range(len(obs))
                for tick,label in zip(pos, ax.get_xticklabels()):
                    ax.text(pos[tick], medians[tick] + 0.03, obs[tick], horizontalalignment='center',
                            size='large', color='black', weight='semibold')
                plt.grid(False)
            plt.show()

    # This method returns seaborn boxplot with color encoding from a certian column
    def categorical_boxplot_with_hue(self, df, categorical_column, numerical_column, hue_column, style):
        if str(df[categorical_column].dtypes) not in ['object', 'category', 'bool']:
            raise TypeError(f'[{categorical_column}] not a categorical datatype')
        elif str(df[numerical_column].dtypes) not in ['int64', 'float64']:
            raise TypeError(f'[{numerical_column}] not a numerical datatype')
        elif str(df[hue_column].dtypes) not in ['object', 'category', 'bool']:
            raise TypeError(f'[{hue_column}] not a categorical datatype')
        else:
            with plt.style.context(style):
                fig = plt.figure(figsize=(17,8))
                sns.boxplot(data = df, x = categorical_column, y = numerical_column,
                            hue= hue_column, showmeans=True)
                plt.grid(False)
            plt.show()

    # This method returns seaborn bivariate barplot
    def categorical_barplot(self, df, categorical_column, numerical_column, style):
        if str(df[categorical_column].dtypes) not in ['object', 'category', 'bool']:
            raise TypeError(f'[{categorical_column}] not a categorical datatype')
        elif str(df[numerical_column].dtypes) not in ['int64', 'float64']:
            raise TypeError(f'[{numerical_column}] not a numerical datatype')
        else:
            with plt.style.context(style):
                fig = plt.figure(figsize=(17, 8))
                order = df.groupby(categorical_column).mean(numeric_only=True).sort_values(numerical_column, ascending=False).index
                ax = sns.barplot(data = df, x=categorical_column, y=numerical_column, order=order,
                                errwidth=0)
                plt.grid(False)

                for p in ax.patches:
                    ax.annotate(format(p.get_height(), '.1f'),
                                (p.get_x() + p.get_width() / 2,
                                p.get_height()), ha='center', va='center',
                                size=15, xytext=(0,8),
                                textcoords='offset points')
            plt.show()

    # This method returns seaborn bivariate barplot with color encoding from a certian column
    def categorical_barplot_with_hue(self, df, categorical_column, numerical_column, hue_column, style):
        if str(df[categorical_column].dtypes) not in ['object', 'category', 'bool']:
            raise TypeError(f'[{categorical_column}] not a categorical datatype')
        elif str(df[numerical_column].dtypes) not in ['int64', 'float64']:
            raise TypeError(f'[{numerical_column}] not a numerical datatype')
        elif str(df[hue_column].dtypes) not in ['object', 'category', 'bool']:
            raise TypeError(f'[{hue_column}] not a categorical datatype')
        else:
            with plt.style.context(style):
                fig = plt.figure(figsize=(17, 8))
                order = df.groupby(categorical_column).mean(numeric_only=True).sort_values(numerical_column, ascending=False).index
                ax = sns.barplot(data = df, x=categorical_column, y=numerical_column, hue=hue_column, order=order,
                             errwidth=0)
                plt.grid(False)

                for p in ax.patches:
                    ax.annotate(format(p.get_height(), '.1f'),
                                (p.get_x() + p.get_width() / 2,
                                p.get_height()), ha='center', va='center',
                                size=15, xytext=(0,8),
                                textcoords='offset points')
            plt.show()

    # This method returns seaborn scatterplot with color encoding from a certian column
    def scatterplot_with_hue(self, df, num_col1, num_col2, hue_col, style):
        if str(df[num_col1].dtypes) not in ['int64', 'float64']:
            raise TypeError(f'[{num_col1}] not a numerical datatype')
        elif str(df[num_col2].dtypes) not in ['int64', 'float64']:
            raise TypeError(f'[{num_col2}] not a numerical datatype')
        elif str(df[hue_col].dtypes) not in ['object','category','bool']:
            raise TypeError(f'[{hue_col}] not a categorical datatype')
        else:
            with plt.style.context(style):
                fig = plt.figure(figsize=(17,8))
                sns.scatterplot(data = df, x=num_col1, y=num_col2, hue=hue_col)
                plt.grid(False)
                plt.legend(loc='upper center', title=hue_col)
            plt.show()

    # This method returns seaborn scatterplot
    def scatterplot(self, df, num_col1, num_col2, style):
        if str(df[num_col1].dtypes) not in ['int64', 'float64']:
            raise TypeError(f'[{num_col1}] not a numerical datatype')
        elif str(df[num_col2].dtypes) not in ['int64', 'float64']:
            raise TypeError(f'[{num_col2}] not a numerical datatype')
        else:
            with plt.style.context(style):
                fig = plt.figure(figsize=(17,8))
                sns.scatterplot(data = df, x=num_col1, y=num_col2)
                plt.grid(False)
                plt.title(f"{num_col1.title().replace('_', ' ')} vs. {num_col2.title().replace('_', ' ')} Scatterplot")
            plt.show()

    # This method returns seaborn joiintplot with regression line
    def jointplot(self, df, num_col1, num_col2, style):
        if str(df[num_col1].dtypes) not in ['int64', 'float64']:
            raise TypeError(f'[{num_col1}] not a numerical datatype')
        elif str(df[num_col2].dtypes) not in ['int64', 'float64']:
            raise TypeError(f'[{num_col2}] not a numerical datatype')
        else:
            with plt.style.context(style):
                fig = plt.figure(figsize=(17,8))
                g = sns.jointplot(data=df, x=num_col1, y=num_col2, kind='reg')
                r, p = stats.pearsonr(df[num_col1].values, df[num_col2].values)
                g.ax_joint.annotate(f'$\\rho = {r:.3f}, p = {p:.3f}$',
                                    xy=(0.1, 0.9), xycoords='axes fraction',
                                    ha='left', va='center',
                                    bbox={'boxstyle':'round'})
                g.ax_joint.scatter(num_col1, num_col2)
                g.set_axis_labels(xlabel=num_col1, ylabel=num_col2, size=15)
                plt.tight_layout()
                plt.grid(False)
            plt.show()

    # This method returns seaborn heatmap
    def list_heatmap(self, df, columns, style):
        if type(columns) != list:
            raise TypeError('parameter not a list')
        else:
            with plt.style.context(style):
                fig = plt.figure(figsize=(17, 8))
                df_new = df[columns]
                sns.heatmap(df_new.corr(), annot=True, cmap='winter')
                plt.title(f"Heatmap of {columns}")
            plt.show()

    # This method takes a pivot table and makes a multivarite heatmap
    def multi_heatmap(self, df, index, column, values, style):
        if index not in df.columns:
            raise TypeError(f"{index} column not in dataframe")
        elif str(df[index].dtypes) not in ['object','category','bool']:
            raise TypeError(f'[{x_column}] not a categorical datatype')
        elif column not in df.columns:
            raise TypeError(f"{column} column not in dataframe")
        elif values not in df.columns:
            raise TypeError(f"{values} column not in dataframe")
        elif str(df[values].dtypes) not in ['int64','float64']:
            raise TypeError(f'[{values}] not a numerical datatype')
        else:
            with plt.style.context(style):
                fig = plt.figure(figsize=(17, 8))
                sns.heatmap(df.pivot_table(index=index, columns=column, values=values))
            plt.show()

    # This method returns seaborn univariate barplot
    def countplot(self, df, column, style):
        if column not in df.columns:
            raise TypeError(f'{column} column not in dataframe')
        else:
            with plt.style.context(style):
                fig = plt.figure(figsize=(17, 8))
                ax = sns.countplot(data = df, x=column, order=df[column].value_counts(normalize=True).index)
                plt.grid(False)
                plt.title(f"{column.title().replace('_', ' ')} Countplot")

                total = len(df[column])
                for p in ax.patches:
                    percentage = f'{100 * p.get_height() / total:.1f}%\n'
                    x = p.get_x() + p.get_width() / 2
                    y = p.get_height()
                    ax.annotate(percentage, (x, y), ha='center', va='center', fontsize=11)
            plt.show()

    # This method returns seaborn univariate barplot with grouping
    def countplot_with_hue(self, df, column, hue_col, style):
        if column not in df.columns:
            raise TypeError(f'{column} column not in dataframe')
        elif hue_col not in df.columns:
            raise TypeError(f'{hue_col} column not in dataframe')
        else:
            with plt.style.context(style):
                fig = plt.figure(figsize=(17, 8))
                ax = sns.countplot(data = df, x=column, hue = hue_col, order = df[column].value_counts(normalize=True).index)
                plt.grid(False)
                plt.title(f"{column.title().replace('_', ' ')} Countplot with {hue_col.title().replace('_', ' ')} Categories")

                total = len(df[column])
                for p in ax.patches:
                    percentage = f'{100 * p.get_height() / total:.1f}%\n'
                    x = p.get_x() + p.get_width() / 2
                    y = p.get_height()
                    ax.annotate(percentage, (x, y), ha='center', va='center', fontsize= 11)

    # This method returns seaborn line graph
    def lineplot(self, df, x_column, y_column, style, ci=None):
        if str(df[x_column].dtypes) not in ['object','category','bool']:
            raise TypeError(f'[{x_column}] not a categorical datatype')
        elif str(df[y_column].dtypes) not in ['int64','float64']:
            raise TypeError(f'[{y_column}] not a numerical datatype')
        else:
            with plt.style.context(style):
                fig = plt.figure(figsize=(17, 8))
                sns.lineplot(data = df, x = x_column, y = y_column, ci=None)
                plt.grid(False)
                plt.title(f"{x_column.title().replace('_', ' ')} vs. "
                          f"{y_column.title().replace('_', ' ')} Lineplot")
            plt.show()

    # This method returns seaborn line graph with color encoding from a certian column
    def lineplot_with_hue(self, df, x_column, y_column, hue_column, style, ci=None):
        if str(df[x_column].dtypes) not in ['object','category','bool']:
            raise TypeError(f'[{x_column}] not a categorical datatype')
        elif str(df[y_column].dtypes) not in ['int64','float64']:
            raise TypeError(f'[{y_column}] not a numerical datatype')
        elif str(df[hue_column].dtypes) not in ['object','category','bool']:
            raise TypeError(f'[{hue_column}] not a categorical datatype')
        else:
            with plt.style.context(style):
                fig = plt.figure(figsize=(17, 8))
                sns.lineplot(data = df, x = x_column, y = y_column, hue = hue_column, ci = None)
                plt.grid(False)
                plt.title(f"{x_column.title().replace('_', ' ')} vs. "
                          f"{y_column.title().replace('_', ' ')} Lineplot")
            plt.show()

    # This method returns a pie chart
    def pie_chart(self, df, column, style):
        if str(df[column].dtypes) not in ['object','category','bool']:
            raise TypeError(f'[{column}] not a categorical datatype')
        else:
            with plt.style.context(style):
                sorted_counts = df[column].value_counts()
                fig, ax = plt.subplots(figsize=(8,8))
                ax.pie(sorted_counts, labels= sorted_counts.index, startangle=90, counterclock=False,
                        autopct='%1.1f%%', textprops={'fontsize':15}, wedgeprops={'linewidth':3, 'edgecolor':'white'})
                ax.axis('equal')
                ax.set_title(f"{column.title().replace('_', ' ')} Pie Chart")
                plt.tight_layout()
            plt.show()

    # This method returns a donut pie chart
    def donut_pie_chart(self, df, column, style):
        if str(df[column].dtypes) not in ['object','category','bool']:
            raise TypeError(f'[{column}] not a categorical datatype')
        else:
            with plt.style.context(style):
                fig, ax = plt.subplots(figsize=(12,12))
                # Create a circle at the center of the plot
                my_circle = plt.Circle((0, 0), 0.7, color='black')
                sorted_counts = df[column].value_counts()
                ax.pie(sorted_counts, labels=sorted_counts.index, startangle=90,
                       counterclock=False, wedgeprops={'width':0.5, 'linewidth':7, 'edgecolor':'black'},
                       autopct='%1.1f%%', pctdistance=0.85, textprops={'fontsize':14})
                p = plt.gcf()
                p.gca().add_artist(my_circle)
                ax.set_title(f"{column.title().replace('_', ' ')} Donut Chart")
            plt.show()

    # This method shows seaborn violinplot
    def violinplot(self, df, cat_col, num_col, style):
        if str(df[cat_col].dtypes) not in ['object','category','bool']:
            raise TypeError(f'[{cat_col}] not a categorical datatype')
        elif str(df[num_col].dtypes) not in ['int64','float64']:
            raise TypeError(f'[{num_col}] not a numerical datatype')
        else:
            with plt.style.context(style):
                fig = plt.figure(figsize=(17,8))
                ax = sns.violinplot(data = df, x=cat_col, y=num_col)

                # calculate the number of obserations for each group and median to position labels
                medians = df.groupby([cat_col])[num_col].median()
                obs = df[cat_col].value_counts().reindex(medians.index).values
                medians = medians.values
                obs = [str(x) for x in obs.tolist()]
                obs = ['n: ' + i for i in obs]

                # Add observations to plot
                pos = range(len(obs))
                for tick, label in zip(pos, ax.get_xticklabels()):
                    ax.text(pos[tick], medians[tick] + 0.03, obs[tick], horizontalalignment='center',
                            size='large', color='black', weight='semibold')
                plt.grid(False)
            plt.show()

    # This method shows seaborn violinplot with grouping
    def violinplot_with_hue(self, df, cat_col, num_col, hue_col, style):
        if str(df[cat_col].dtypes) not in ['object','category','bool']:
            raise TypeError(f'[{cat_col}] not a categorical datatype')
        elif str(df[num_col].dtypes) not in ['int64', 'float64']:
            raise TypeError(f'[{num_col}] not a numerical datatype')
        elif str(df[hue_col].dtypes) not in ['object','category','bool']:
            raise TypeError(f'[{hue_col}] not a categorical datatype')
        else:
            with plt.style.context(style):
                fig  = plt.figure(figsize=(17,8))
                sns.violinplot(data = df, x=cat_col, y=num_col, hue=hue_col)
                plt.grid(False)
                plt.legend(loc='upper center', title=hue_col.title().replace('_', ' '))
            plt.show()

    # The method creates circular bar plot
    def circular_barplot(self, df, cat_col, num_col, style, bar_color):
        if str(df[cat_col].dtypes) not in ['object','category','bool']:
            raise TypeError(f'[{cat_col}] not a categorical datatype')
        elif str(df[num_col].dtypes) not in ['int64', 'float64']:
            raise TypeError(f'[{num_col}] not a numerical datatype')
        else:
            with plt.style.context(style):
                # Reorder dataframe
                df = df.groupby([cat_col])[num_col].mean().round(0).reset_index()
                df = df.sort_values(by=[num_col])

                pal = list(sns.color_palette(palette=bar_color, n_colors=len(self)).as_hex())

                # initialize figure
                plt.figure(figsize=(20,10))
                ax = plt.subplot(111, polar=True)
                plt.axis('off')

                # constraints = parameters controling plot layout
                #upper_limit = 100
                lower_limit = 30
                label_padding = 4

                # compute max and min in the dataset
                max_vl = df[num_col].max()

                # Let's compute heights: they are a conversion of each item value in those new coordinates
                # In our example, 0 in the dataset will be converted to the lowerLimit (10)
                # The maximum will be converted to the upperLimit (100)
                slope = (max_vl - lower_limit) / max_vl
                heights = slope * df[num_col] + lower_limit

                # compute the width of each bar. In total we have 2*pi = 360 degrees
                width = 2*np.pi / len(df.index)

                # compute the angle each bar is centered on:
                indexes = list(range(1, len(df.index) + 1))
                angles = [item * width for item in indexes]
                angles

                # draw bars
                bars = ax.bar(x=angles, height=heights, width=width, bottom=lower_limit, linewidth=2,
                              edgecolor='white', color=pal)

                # Add labels
                for bar, angle, height, label in zip(bars, angles, heights, df[cat_col]):
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
    def race_track_plot(self, df, cat_col, num_col, style, bar_color):
        if str(df[cat_col].dtypes) not in ['object','category','bool']:
            raise TypeError(f'[{cat_col}] not a categorical datatype')
        elif str(df[num_col].dtypes) not in ['int64', 'float64']:
            raise TypeError(f'[{num_col}] not a numerical datatype')
        else:
            with plt.style.context(style):
                # Reorder dataframe
                df = df.groupby([cat_col])[num_col].mean().round(0).reset_index()
                df = df.sort_values(by=[num_col])

                pal = list(sns.color_palette(palette=bar_color, n_colors=len(self)).as_hex())

                # initialize figure
                plt.gcf().set_size_inches(12,12)
                sns.set_style('darkgrid')

                # set max value
                max_v1 = max(df[num_col]) * 1.01
                ax = plt.subplot(projection= 'polar')

                for i in range(len(df)):
                    ax.barh(i, list(df[num_col])[i]*2*np.pi/max_v1,
                            label=list(df[cat_col])[i], color=pal[i])

                # Set subplot
                ax.set_theta_zero_location('N')
                ax.set_theta_direction(1)
                ax.set_rlabel_position(0)
                ax.set_thetagrids([], labels=[])
                ax.set_rgrids(range(len(df)), labels=df[cat_col])

                #set the pojection
                plt.legend(bbox_to_anchor=(1, 1), loc=2)
            plt.show()

    """ Interactive Visualization Charts """

    # This method produces an interactive Treemap
    def treemap(self, df, cat_col, num_col, style, color_scale):
        if str(df[cat_col].dtypes) not in ['object','category','bool']:
            raise TypeError(f'[{cat_col}] not a categorical datatype')
        elif str(df[num_col].dtypes) not in ['int64', 'float64']:
            raise TypeError(f'[{num_col}] not a numerical datatype')
        else:
            with plt.style.context(style):
                df = df.groupby([cat_col])[num_col].mean().round().reset_index()
                fig = px.treemap(df, path=[px.Constant(f'{cat_col.title()} Categories'), cat_col],
                                 values=df[num_col],
                                 color=df[num_col],
                                 color_continuous_scale=color_scale,
                                 color_continuous_midpoint=np.average(df[num_col]),
                                 title= f"Treemap of {cat_col.title()} Categories and the Average {num_col.title().replace('_', ' ')} for Each Category."
                                 )
                fig.update_layout(margin= dict(t=50, l=25, r=25, b=25), title_x=0.5)
                fig.show()
            plt.show()

    # This method produces interactive pie chart with percentages
    def percentage_pie_chart(self, df, cat_col, num_col, bar_col):
        if str(df[cat_col].dtypes) not in ['object','category','bool']:
            raise TypeError(f'[{cat_col}] not a categorical datatype')
        elif str(df[num_col].dtypes) not in ['int64', 'float64']:
            raise TypeError(f'[{num_col}] not a numerical datatype')
        else:
            df = df.groupby([cat_col])[num_col].mean().round(0).reset_index()
            df = df.sort_values(by=[num_col])

            pal = list(sns.color_palette(palette=bar_col, n_colors=len(df)).as_hex())
            fig = px.pie(df, values=num_col, names=cat_col, color=cat_col, color_discrete_sequence=pal)
            fig.update_traces(textposition='inside', textinfo='percent+label', sort=False)
            fig.update_layout(width=1000, height=550)
            fig.show()

    # This method produces interactive bar chart
    def interactive_bar_chart(self, df, cat_col, num_col, bar_col):
        if str(df[cat_col].dtypes) not in ['object','category','bool']:
            raise TypeError(f'[{cat_col}] not a categorical datatype')
        elif str(df[num_col].dtypes) not in ['int64', 'float64']:
            raise TypeError(f'[{num_col}] not a numerical datatype')
        else:
            df = df.groupby([cat_col])[num_col].mean().round(0).reset_index()
            df = df.sort_values(by=[num_col])

            pal = list(sns.color_palette(palette=bar_col, n_colors=len(df)).as_hex())

            fig = px.bar(df, x=cat_col, y=num_col, text=num_col,
                         color=cat_col, color_discrete_sequence=pal)

            fig.update_traces(texttemplate='%{text:.3s}', textposition='outside')
            fig.update_layout({'plot_bgcolor':'white', 'paper_bgcolor':'white'})
            fig.update_layout(width=1100, height=500, margin=dict(t=15, l=15, r=15, b=15))
            fig.show()

    # This method creates interactive polar chart
    def polar_line_chart(self, df, cat_col, num_col, bar_col):
        if str(df[cat_col].dtypes) not in ['object','category','bool']:
            raise TypeError(f'[{cat_col}] not a categorical datatype')
        elif str(df[num_col].dtypes) not in ['int64', 'float64']:
            raise TypeError(f'[{num_col}] not a numerical datatype')
        else:
            df = df.groupby([cat_col])[num_col].mean().round(0).reset_index()
            df = df.sort_values(by=[num_col])
            pal = list(sns.color_palette(palette=bar_col, n_colors=len(df)).as_hex())
            fig = px.line_polar(df, r=num_col, theta=cat_col, line_close=True)
            fig.update_traces(fill='toself', line = dict(color=pal[-5]))
            fig.update_traces(mode="markers+lines")
            fig.update_layout({'plot_bgcolor': 'rgba(0,0,0,0)', 'paper_bgcolor': 'rgba(0,0,0,0)'})
            fig.show()

    # This method creates an interactive circular bubble chart
    def circular_bubble_chart(self, df, cat_col, num_col, bar_col):
        if str(df[cat_col].dtypes) not in ['object','category','bool']:
            raise TypeError(f'[{cat_col}] not a categorical datatype')
        elif str(df[num_col].dtypes) not in ['int64', 'float64']:
            raise TypeError(f'[{num_col}] not a numerical datatype')
        else:
            df = df.groupby([cat_col])[num_col].mean().round(0).reset_index()
            df = df.sort_values(by=[num_col])
            self['labels'] = ['<b>'+i+'<b>'+format(j,',') for i,j in zip(df[cat_col], df[num_col])]

            e = 360 / len(df)
            deg = [i * e for i in list(range(len(df)))]
            df['x'] = [math.cos(i * math.pi / 180) for i in deg]
            df['y'] = [math.sin(i * math.pi / 180) for i in deg]
            pal = list(sns.color_palette(palette=bar_col, n_colors=len(df)).as_hex())

            fig = px.scatter(df, x='x', y='y', color=cat_col, color_discrete_sequence=pal,
                             size=num_col, text='labels', size_max=40)
            fig.update_layout(width=800, height=800, margin= dict(t=0, l=0, r=0, b=0),
                              showlegend=False)
            fig.update_traces(textposition='bottom center')
            fig.update_xaxes(showgrid=False, zeroline=False, visible=False)
            fig.update_yaxes(showgrid=False, zeroline=False, visible=False)

            fig.update_layout({'plot_bgcolor':'white', 'paper_bgcolor':'white'})
            fig.show()

    """ Multi-graph subplots """

    # This method makes subplots of regresssion plots
    def regression_subplots(self, df, cat_col, num_col1, num_col2, style, sub_1, sub_2):
        if cat_col not in df.columns:
            raise TypeError(f'{cat_col} column not in dataframe')
        elif str(df[cat_col].dtypes) not in ['object','category','bool']:
            raise TypeError(f'[{cat_col}] not a categorical datatype')
        elif num_col1 not in df.columns:
            raise TypeError(f'{num_col1} column not in dataframe')
        elif str(df[num_col1].dtypes) not in ['int64','float64']:
            raise TypeError(f'[{num_col1}] not a numerical datatype')
        elif num_col2 not in df.columns:
            raise TypeError(f'{num_col2} column not in dataframe')
        elif str(df[num_col2].dtypes) not in ['int64', 'float64']:
            raise TypeError(f'[{num_col2}] not a numerical datatype')
        else:
            with plt.style.context(style):
                cat_val = df[cat_col].unique().tolist()
                fig = plt.figure(figsize=(20,20))
                k = 1
                for x in cat_val:
                    plt.subplot(sub_1, sub_2, k)
                    data = df[df[cat_col] == x]
                    g = sns.regplot(data=data, x=num_col1, y=num_col2, ci=False)
                    plt.annotate(x, xy=(0.05, 0.9), xycoords='axes fraction',
                                 bbox=dict(boxstyle="round", fc='tab:red', alpha=0.6))
                    r = stats.pearsonr(data[num_col1], data[num_col2])[0]
                    plt.annotate(f'$\\rho = {r :.2f}$', xy=(0.05, 0.8), xycoords='axes fraction')
                    plt.grid(False)
                    k += 1
            plt.show()

    # This method returns subplots of histograms for numerical columns
    def histogram_subplots(self, df, style, sub_1, sub_2):
        num_cols = []
        for x in df.columns:
            if str(df[x].dtypes) in ['int64','float64']:
                num_cols.append(x)
            else:
                continue

        fig = plt.figure(figsize=(20,20))
        k = 1
        for i in num_cols:
            with plt.style.context(style):
                plt.subplot(sub_1, sub_2, k)
                plt.xlabel(i)
                sns.histplot(x=df[i], kde=True)
                k += 1
        plt.show()

    # This method return subplots of countplots
    def count_subplots(self, df, style, sub_1, sub_2):
        cat_cols = []
        for x in df.columns:
            if str(df[x].dtypes) in ['int64','float64']:
                continue
            else:
                cat_cols.append(x)

        fig = plt.figure(figsize=(20,20))
        k = 1
        for i in cat_cols:
            with plt.style.context(style):
                plt.subplot(sub_1, sub_2, k)
                plt.xlabel(i)
                sns.countplot(data=df, x=i, order=df[i].value_counts(normalize=True).index)
                k += 1
        plt.show()

    # This method return subplots of scatter plots
    def scatter_subplots(self, df, num_col, hue_col, style, sub_1, sub_2):
        if num_col not in df.columns:
            raise TypeError(f"{num_col} not a column in dataframe")
        elif str(df[num_col].dtypes) not in ['int64','float64']:
            raise TypeError(f'[{num_col1}] not a numerical datatype')
        elif hue_col not in df.columns:
            raise TypeError(f"{hue_col} not a column in dataframe")
        elif str(df[hue_col].dtypes) not in ['object','category','bool']:
            raise TypeError(f'[{hue_col}] not a categorical datatype')
        else:
            with plt.style.context(style):
                num_cols = []
                for x in df.columns:
                    if str(df[x].dtypes) in ['int64','float64']:
                        num_cols.append(x)
                    else:
                        continue
                fig = plt.figure(figsize=(20, 20))
                k = 1
                for j in num_cols:
                    if j != num_col:
                        plt.subplot(sub_1, sub_2, k)
                        plt.xlabel(j)
                        sns.scatterplot(x=j, y=num_col, hue=hue_col, palette='magma', data=df)
                        k += 1
                        plt.grid(False)
                plt.show()

    # This method returns subplots of histograms for numerical columns
    def box_subplots(self, df, style, sub_1, sub_2):
        num_cols = []
        for x in df.columns:
            if str(df[x].dtypes) in ['int64','float64']:
                num_cols.append(x)
            else:
                continue
        fig = plt.figure(figsize=(20, 20))
        k = 1
        for i in num_cols:
            with plt.style.context(style):
                plt.subplot(sub_1, sub_2, k)
                plt.xlabel(i)
                sns.boxplot(x=df[i])
                k += 1
        plt.show()

    # This method returns subplots of barplots
    def bar_subplots(self, df, cat_col, style, sub_1, sub_2):
        if cat_col not in df.columns:
            raise TypeError(f"{cat_col} not a column in dataframe")
        elif str(df[cat_col].dtypes) not in ['object','category','bool']:
            raise TypeError(f'[{cat_col}] not a categorical datatype')
        else:
            with plt.style.context(style):
                num_cols = []
                for x in df.columns:
                    if str(df[x].dtypes) in ['int64','float64']:
                        num_cols.append(x)
                    else:
                        continue
                fig = plt.figure(figsize=(20, 20))
                k = 1

                for j in num_cols:
                    plt.subplot(sub_1, sub_2, k)
                    plt.xlabel(j)
                    order = df.groupby(cat_col).mean(numeric_only=True).sort_values(j, ascending=False).index
                    ax = sns.barplot(x=cat_col, y=j, order=order, palette='magma', errwidth=0, data=df)
                    ax.set_title(f"{cat_col.title().replace('_', ' ')} vs. {j.title().replace('_', ' ')} Bar Chart")
                    plt.grid(False)
                    k += 1

                    for p in ax.patches:
                        ax.annotate(format(p.get_height(), '.1f'),
                                    (p.get_x() + p.get_width() / 2,
                                     p.get_height()), ha='center', va='center',
                                    size=15, xytext=(0, 8),
                                    textcoords='offset points')
            plt.show()
