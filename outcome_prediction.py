import polars as pl
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import statsmodels.formula.api as smf
import statsmodels as sm
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import classification_report, mean_squared_error, r2_score
from sklearn.feature_selection import VarianceThreshold
from statsmodels.stats.outliers_influence import variance_inflation_factor
from sklearn.ensemble import GradientBoostingRegressor

def load_and_preprocess_data(file_path):
    """
    Load and preprocess the pet adoption data.

    Args:
        file_path (str): Path to the CSV file.

    Returns:
        pl.DataFrame: Preprocessed DataFrame.
    """
    df = pl.read_csv(file_path)
    df = df.drop_nulls(subset=["outcome_type", "sex_upon_intake"])
    
    columns_to_drop = [
        "age_upon_outcome", "age_upon_outcome_(days)", "age_upon_outcome_age_group",
        "outcome_month", "outcome_monthyear", "outcome_weekday", "outcome_hour",
        "dob_year", "dob_monthyear", "age_upon_intake", "age_upon_intake_(days)",
        "age_upon_intake_age_group", "intake_month", "intake_monthyear",
        "intake_weekday", "intake_hour", "time_in_shelter"
    ]
    df = df.drop(columns_to_drop)
    
    df = df.with_columns(
        pl.col("date_of_birth").str.to_datetime(),
        pl.col("outcome_datetime").str.to_datetime(),
        pl.col("intake_datetime").str.to_datetime(),
    )
    
    # assert df.null_count().sum().sum() == 0, "Preprocessed data contains null values"

    return df

def plot_outcome_distribution(df):
    """
    Plot the distribution of outcome types.

    Args:
        df (pl.DataFrame): Preprocessed DataFrame.
    """
    outcome_counts = df['outcome_type'].value_counts().sort('count', descending=True)
    plt.figure(figsize=(12, 6))
    plt.bar(outcome_counts['outcome_type'], outcome_counts['count'])
    plt.title('Distribution of Outcome Types')
    plt.xlabel('Outcome Type')
    plt.ylabel('Count')
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    plt.show()

def analyze_euthanasia_cases(df):
    """
    Analyze euthanasia cases.

    Args:
        df (pl.DataFrame): Preprocessed DataFrame.

    Returns:
        pl.DataFrame: Euthanasia analysis results.
    """
    euth_analysis = df.filter(pl.col('outcome_type') == 'Euthanasia')\
        .group_by('outcome_subtype')\
        .agg([
            pl.count('outcome_subtype').alias('count'),
            pl.mean('time_in_shelter_days').alias('avg_time_in_shelter'),
        ])\
        .sort('count', descending=True)
    
    assert len(euth_analysis) > 0, "No euthanasia cases found"
    return euth_analysis

def plot_euthanasia_analysis(euth_analysis):
    """
    Plot euthanasia analysis results.

    Args:
        euth_analysis (pl.DataFrame): Euthanasia analysis results.
    """
    euth_analysis_df = euth_analysis.to_pandas()
    fig, ax1 = plt.subplots(figsize=(10, 6))
    sns.barplot(
        x='count', y='outcome_subtype', data=euth_analysis_df,
        hue='outcome_subtype', palette='viridis', ax=ax1
    )
    ax1.set_xlabel('Count')
    ax1.set_ylabel('Outcome Subtype')
    ax1.set_title('Euthanasia Subtype Analysis: Count and Avg Time in Shelter')
    
    ax2 = ax1.twiny()
    sns.scatterplot(
        x='avg_time_in_shelter', y='outcome_subtype', data=euth_analysis_df,
        color='red', marker='o', ax=ax2
    )
    ax2.set_xlabel('Avg Time in Shelter (days)', color='red')
    ax2.tick_params(axis='x', colors='red')
    plt.tight_layout()
    plt.show()

def plot_outcome_by_animal_type(df):
    """
    Plot outcome types by animal type.

    Args:
        df (pl.DataFrame): Preprocessed DataFrame.
    """
    sns.countplot(data=df.to_pandas(), x='animal_type', hue='outcome_type')
    plt.title('Outcome Types by Animal Type')
    plt.xticks(rotation=45)
    plt.show()

def analyze_specific_animal(df, animal_type):
    """
    Analyze outcome distribution for a specific animal type.

    Args:
        df (pl.DataFrame): Preprocessed DataFrame.
        animal_type (str): Type of animal to analyze.

    Returns:
        pl.DataFrame: Outcome distribution for the specified animal type.
    """
    animal_outcome = df.filter(pl.col('animal_type') == animal_type)\
        .group_by('outcome_type')\
        .agg([pl.count('outcome_type').alias('count'),])\
        .sort('count', descending=True)
    
    assert len(animal_outcome) > 0, f"No data found for {animal_type}"
    return animal_outcome

def plot_animal_outcome_distribution(animal_outcome, animal_type):
    """
    Plot outcome distribution for a specific animal type.

    Args:
        animal_outcome (pl.DataFrame): Outcome distribution data.
        animal_type (str): Type of animal being analyzed.
    """

    # Set up the plot
    plt.figure(figsize=(10, 8))
    sns.set(font_scale=1.2)

    # Create the pie chart with customizations
    wedges, texts, autotexts = plt.pie(
        animal_outcome['count'],
        labels=animal_outcome['outcome_type'],
        autopct='%1.1f%%',
        pctdistance=0.85,
        startangle=90,
        colors=sns.color_palette('Set2'),
        wedgeprops=dict(width=0.5),  # Create a donut chart
        textprops={'fontweight': 'bold'}
    )

    # Customize label positions
    for text in texts:
        text.set_horizontalalignment('center')

    # Customize percentage labels
    for autotext in autotexts:
        autotext.set_horizontalalignment('center')
        autotext.set_fontsize(10)

    # Add a title
    plt.title(f'Outcome Distribution for {animal_type}', fontsize=16, pad=20)

    # Add a legend
    plt.legend(wedges, animal_outcome['outcome_type'],
            title="Outcome Types",
            loc="center left",
            bbox_to_anchor=(1, 0, 0.5, 1))

    plt.tight_layout()
    plt.show()

def plot_age_vs_time_in_shelter(df):
    """
    Create a scatter plot of age upon intake vs time in shelter.

    Args:
        df (pl.DataFrame): The input DataFrame containing pet data.
    """
    sns.scatterplot(data=df.to_pandas(), x='age_upon_intake_(years)', y='time_in_shelter_days', hue='animal_type')
    plt.title('Age upon Intake vs Time in Shelter')
    plt.show()

def get_top_breeds(df, animal_type, n=10):
    """
    Get the top n breeds for a specific animal type.

    Args:
        df (pl.DataFrame): The input DataFrame containing pet data.
        animal_type (str): The type of animal to filter for (e.g., 'Dog', 'Cat').
        n (int): The number of top breeds to return. Default is 10.

    Returns:
        pl.DataFrame: A DataFrame containing the top n breeds and their counts.
    """
    top_breeds = (
        df.filter(pl.col('animal_type') == animal_type)
        .group_by('breed')
        .agg(pl.count('breed').alias('count'))
        .sort('count', descending=True)
        .limit(n)
    )
    assert len(top_breeds) <= n, f"Number of top breeds exceeds {n}"
    return top_breeds

def plot_top_breeds(top_breeds, animal_type):
    """
    Create a bar plot of the top breeds for a specific animal type.

    Args:
        top_breeds (pl.DataFrame): DataFrame containing breed and count information.
        animal_type (str): The type of animal (e.g., 'Dog', 'Cat').
    """
    plt.figure(figsize=(12, 6))
    plt.bar(top_breeds['breed'].to_list(), top_breeds['count'].to_list(), color='skyblue')
    plt.title(f'Top 10 {animal_type} Breeds')
    plt.xlabel(f'Breed of {animal_type}s')
    plt.ylabel('Count')
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    plt.show()

def get_adopted_animals(df, animal_type):
    """
    Filter the DataFrame for adopted animals of a specific type.

    Args:
        df (pl.DataFrame): The input DataFrame containing pet data.
        animal_type (str): The type of animal to filter for (e.g., 'Dog', 'Cat').

    Returns:
        pl.DataFrame: A DataFrame containing only adopted animals of the specified type.
    """
    adopted = df.filter(
        (pl.col('animal_type') == animal_type) & 
        (pl.col('outcome_type') == 'Adoption')
    )
    assert len(adopted) > 0, f"No adopted {animal_type}s found in the data"
    return adopted

def plot_age_distribution(adopted_animals, animal_type):
    """
    Create a histogram of the age distribution for adopted animals.

    Args:
        adopted_animals (pl.DataFrame): DataFrame containing adopted animal data.
        animal_type (str): The type of animal (e.g., 'Dog', 'Cat').
    """
    plt.figure(figsize=(10, 6))
    sns.histplot(adopted_animals['age_upon_outcome_(years)'].to_numpy(), bins=20, kde=True, color='skyblue')
    plt.title(f'Age Distribution of Adopted {animal_type}s')
    plt.xlabel('Age (years)')
    plt.ylabel('Count')
    plt.tight_layout()
    plt.show()

def get_top_colors(adopted_animals, n=10):
    """
    Get the top n colors for adopted animals.

    Args:
        adopted_animals (pl.DataFrame): DataFrame containing adopted animal data.
        n (int): The number of top colors to return. Default is 10.

    Returns:
        pl.DataFrame: A DataFrame containing the top n colors and their counts.
    """
    top_colors = (
        adopted_animals
        .group_by('color')
        .agg(pl.len().alias('count'))
        .sort('count', descending=True)
        .limit(n)
    )
    assert len(top_colors) <= n, f"Number of top colors exceeds {n}"
    return top_colors

def plot_top_colors(top_colors, animal_type):
    """
    Create a bar plot of the top colors for adopted animals.

    Args:
        top_colors (pl.DataFrame): DataFrame containing color and count information.
        animal_type (str): The type of animal (e.g., 'Dog', 'Cat').
    """
    plt.figure(figsize=(12, 6))
    plt.bar(top_colors['color'].to_list(), top_colors['count'].to_list(), color='salmon')
    plt.title(f'Top 10 Colors of Adopted {animal_type}s')
    plt.xlabel('Color')
    plt.ylabel('Count')
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    plt.show()

def get_sex_distribution(adopted_animals):
    """
    Get the sex distribution of adopted animals.

    Args:
        adopted_animals (pl.DataFrame): DataFrame containing adopted animal data.

    Returns:
        pl.DataFrame: A DataFrame containing the sex distribution counts.
    """
    sex_counts = (
        adopted_animals
        .group_by('sex_upon_outcome')
        .agg(pl.len().alias('count'))
        .sort('count', descending=True)
    )
    assert len(sex_counts) > 0, "No sex distribution data found"
    return sex_counts

def plot_sex_distribution(sex_counts, animal_type):
    """
    Create a pie chart of the sex distribution for adopted animals.

    Args:
        sex_counts (pl.DataFrame): DataFrame containing sex distribution data.
        animal_type (str): The type of animal (e.g., 'Dog', 'Cat').
    """
    plt.figure(figsize=(6, 6))
    plt.pie(
        sex_counts['count'].to_list(), 
        labels=sex_counts['sex_upon_outcome'].to_list(), 
        autopct='%1.1f%%', 
        colors=plt.cm.Paired.colors
    )
    plt.title(f'Sex Distribution of Adopted {animal_type}s')
    plt.axis('equal')
    plt.tight_layout()
    plt.show()

def one_hot_adoption(adopted):
    '''
    One-hot encodes adoption

    Arg: 
    adopted: pd.Series to apply to
    '''
    if adopted == "Adoption":
        return 1
    else:
        return 0

def one_hot_gender(gender):
    '''
    One hot encodes gender

    Args: 
    gender: pd.Series to apply to
    '''
    
    if "Fe" in gender:
        return 'F'
    else:
        return 'M'

def age_vs_adoption(filepath):
    '''
    Fits a linear regression model for age of dogs versus adoption likelihood

    Args:
    filepath(str): filepath of csv
    '''
    assert isinstance(filepath, str), "filepath must be a string"
    df = pd.read_csv(filepath)
    df.dropna(inplace=True)
    df['date_of_birth'] = pd.to_datetime(df['date_of_birth'])
    df['outcome_datetime'] = pd.to_datetime(df['outcome_datetime'])
    df['intake_datetime'] = pd.to_datetime(df['intake_datetime'])

    df_dogs = df[df['animal_type']=='Dog']
    df_dogs['age_upon_intake_years'] = df_dogs["age_upon_intake_(years)"]
    df_dogs['encoded_adoption'] = df_dogs['outcome_type'].apply(one_hot_adoption)

    model_age = smf.ols(formula='encoded_adoption ~ age_upon_intake_years + 1', data=df_dogs)
    results_age = model_age.fit()

    return results_age

def gender_vs_adoption(filepath):
    '''
    Fits a linear regression model for gender of dogs versus adoption likelihood

    Args:
    filepath (str): filepath of csv
    '''
    assert isinstance(filepath, str), "filepath must be a string"
    df = pd.read_csv(filepath)
    df.dropna(inplace=True)
    df['date_of_birth'] = pd.to_datetime(df['date_of_birth'])
    df['outcome_datetime'] = pd.to_datetime(df['outcome_datetime'])
    df['intake_datetime'] = pd.to_datetime(df['intake_datetime'])

    df_dogs = df[df['animal_type']=='Dog']
    df_dogs['encoded_adoption'] = df_dogs['outcome_type'].apply(one_hot_adoption)
    df_dogs['gender_classified'] = df_dogs['sex_upon_intake'].apply(one_hot_gender)
    df_dogs = pd.get_dummies(df_dogs, columns = ['gender_classified'], drop_first=False)
    model_gender = smf.ols(formula='encoded_adoption ~ gender_classified_F + gender_classified_M + 1', data=df_dogs)
    results_gender = model_gender.fit()

    return results_gender

def color_vs_adoption(filepath):
    '''
    Fits a linear regression model for color of dogs versus adoption likelihood

    Args:
    filepath (str): filepath of csv
    '''
    assert isinstance(filepath, str), "filepath must be a string"
    df = pd.read_csv(filepath)
    df.dropna(inplace=True)
    df['date_of_birth'] = pd.to_datetime(df['date_of_birth'])
    df['outcome_datetime'] = pd.to_datetime(df['outcome_datetime'])
    df['intake_datetime'] = pd.to_datetime(df['intake_datetime'])

    df_dogs = df[df['animal_type']=='Dog']
    df_dogs['encoded_adoption'] = df_dogs['outcome_type'].apply(one_hot_adoption)
    df_dogs = pd.get_dummies(df_dogs, columns = ['color'], drop_first=False)
    model_color = smf.ols(formula='encoded_adoption ~ color_Apricot + color_Black + color_Blue + color_Brown + color_Gray + color_Tan + 1', data=df_dogs)
    results_color = model_color.fit()

    return results_color

def logistic_regression_cats(filepath):
    '''
    Performs logistic regression for cats using the features age, sex, breed, color, and intake condition to predict adoption likelihood

    Args
    filepath (str): filepath of csv
    '''
    assert isinstance(filepath, str), "filepath must be a string"
    df = load_and_preprocess_data(filepath)
    # Filter data for cats and create a binary target for adoption
    cats = df.filter(pl.col('animal_type') == 'Cat')
    cats = cats.with_columns(
        pl.Series((cats['outcome_type'] == 'Adoption').to_numpy().astype(int)).alias('adopted')
    )

    # Drop the row where the sex is unknown
    cats = cats.filter(pl.col("sex_upon_intake") != "Unknown")

    # Prepare features for logistic regression (adoption likelihood)
    features_classification = [
        'age_upon_intake_(years)', 'sex_upon_intake', 'breed', 'color', 'intake_condition'
    ]
    X_classification = pd.get_dummies(cats[features_classification].to_pandas(), drop_first=True)
    y_classification = cats['adopted'].to_pandas()

    # Split and train logistic regression
    X_train_clf, X_test_clf, y_train_clf, y_test_clf = train_test_split(
        X_classification, y_classification, test_size=0.2, random_state=42
    )

    logistic_model = LogisticRegression(max_iter=1000)
    logistic_model.fit(X_train_clf, y_train_clf)

    # Predict and evaluate logistic regression
    # change the threshold to 0.6 for predicting
    predictions_clf = logistic_model.predict_proba(X_test_clf)[:, 1] > 0.5
    classification_report_clf = classification_report(y_test_clf, predictions_clf)

    return classification_report_clf

def random_forest_cats(filepath):
    '''
    RandomForest model to predict time in shelter

    Args:
    filepath (str): filepath for csv
    '''
    assert isinstance(filepath, str), "filepath must be a string"
    df = load_and_preprocess_data(filepath)
    # Filter data for cats and create a binary target for adoption
    cats = df.filter(pl.col('animal_type') == 'Cat')
    cats = cats.with_columns(
        pl.Series((cats['outcome_type'] == 'Adoption').to_numpy().astype(int)).alias('adopted')
    )

    # Drop the row where the sex is unknown
    cats = cats.filter(pl.col("sex_upon_intake") != "Unknown")

    # Prepare features for regression (time in shelter prediction)
    features_regression = [
        'age_upon_intake_(years)', 'sex_upon_intake', 'breed', 'color', 'intake_condition'
    ]
    X_regression = pd.get_dummies(cats[features_regression].to_pandas(), drop_first=True)
    y_regression = cats['time_in_shelter_days'].to_pandas()

    # Split and train random forest regressor
    X_train_reg, X_test_reg, y_train_reg, y_test_reg = train_test_split(
        X_regression, y_regression, test_size=0.3, random_state=42
    )

    rf_regressor = RandomForestRegressor(n_estimators=100, random_state=42)
    rf_regressor.fit(X_train_reg, y_train_reg)

    # Predict and evaluate random forest regressor
    predictions_reg = rf_regressor.predict(X_test_reg)
    mse = mean_squared_error(y_test_reg, predictions_reg)
    r2 = r2_score(y_test_reg, predictions_reg)

    return mse, r2

def extended_regression(filepath):
    '''
    Extended Regression model for month of birth

    Args:
    filepath (str): filepath for csv
    '''
    assert isinstance(filepath, str), "filepath must be a string"
    df = load_and_preprocess_data(filepath)
    # Filter data for cats and create a binary target for adoption
    cats = df.filter(pl.col('animal_type') == 'Cat')
    cats = cats.with_columns(
        pl.Series((cats['outcome_type'] == 'Adoption').to_numpy().astype(int)).alias('adopted')
    )

    # Drop the row where the sex is unknown
    cats = cats.filter(pl.col("sex_upon_intake") != "Unknown")

    # Prepare features for regression (time in shelter prediction)
    features_regression = [
        'age_upon_intake_(years)', 'sex_upon_intake', 'breed', 'color', 'intake_condition'
    ]
    X_regression = pd.get_dummies(cats[features_regression].to_pandas(), drop_first=True)
    y_regression = cats['time_in_shelter_days'].to_pandas()

    # Split and train random forest regressor
    X_train_reg, X_test_reg, y_train_reg, y_test_reg = train_test_split(
        X_regression, y_regression, test_size=0.3, random_state=42
    )

    rf_regressor = RandomForestRegressor(n_estimators=100, random_state=42)
    rf_regressor.fit(X_train_reg, y_train_reg)

    # Predict and evaluate random forest regressor
    predictions_reg = rf_regressor.predict(X_test_reg)
    mse = mean_squared_error(y_test_reg, predictions_reg)
    r2 = r2_score(y_test_reg, predictions_reg)

    additional_features_regression = [
        'dob_month'
    ]
    X_regression_extended = pd.get_dummies(
        cats[features_regression + additional_features_regression].to_pandas(), drop_first=True
    )

    X_with_const = sm.add_constant(X_regression_extended.astype(float))  # Add constant for intercept
    ols_model = sm.OLS(y_regression, X_with_const).fit()
    print(ols_model.summary())

    # Fix significant features extraction
    significant_features = ols_model.pvalues[ols_model.pvalues < 0.05].index
    print("\nSignificant Features:")
    print(significant_features)
    X_regression_significant = X_regression_extended[significant_features]

    # Remove near-zero variance features
    variance_filter = VarianceThreshold(threshold=0.01)
    X_reduced_variance = variance_filter.fit_transform(X_regression_significant)
    columns_retained = X_regression_significant.columns[variance_filter.get_support()]

    # Recreate the DataFrame with reduced features
    X_regression_cleaned = pd.DataFrame(X_reduced_variance, columns=columns_retained)

    # Recompute VIF
    vif_data_cleaned = pd.DataFrame()
    vif_data_cleaned["feature"] = X_regression_cleaned.columns
    vif_data_cleaned["VIF"] = [
        variance_inflation_factor(X_regression_cleaned.values, i)
        for i in range(X_regression_cleaned.shape[1])
    ]
    print("\nVariance Inflation Factor (VIF) after removing near-zero variance features:")
    print(vif_data_cleaned)

    # Remove features with high VIF (>10)
    low_vif_features_cleaned = vif_data_cleaned[vif_data_cleaned["VIF"] < 10]["feature"]
    X_regression_low_vif_cleaned = X_regression_cleaned[low_vif_features_cleaned]

    # Train Gradient Boosting with cleaned features
    X_train_reg_cleaned, X_test_reg_cleaned, y_train_reg_cleaned, y_test_reg_cleaned = train_test_split(
        X_regression_low_vif_cleaned, y_regression, test_size=0.3, random_state=42
    )

    gb_regressor_cleaned = GradientBoostingRegressor(n_estimators=200, learning_rate=0.1, random_state=42)
    gb_regressor_cleaned.fit(X_train_reg_cleaned, y_train_reg_cleaned)

    # Predict and evaluate
    predictions_reg_cleaned = gb_regressor_cleaned.predict(X_test_reg_cleaned)
    mse_cleaned = mean_squared_error(y_test_reg_cleaned, predictions_reg_cleaned)
    r2_cleaned = r2_score(y_test_reg_cleaned, predictions_reg_cleaned)

    return mse_cleaned, r2_cleaned

def plot_feature_importance(df):
    '''
    Creates a Dataframe of the importance of features and plots them

    Args:
    df: pd.DataFrame to filter for cats
    '''
    assert isinstance(df, pd.DataFrame), "df must be a pandas DataFrame"
    # Filter data for cats and create a binary target for adoption
    cats = df.filter(pl.col('animal_type') == 'Cat')
    cats = cats.with_columns(
        pl.Series((cats['outcome_type'] == 'Adoption').to_numpy().astype(int)).alias('adopted')
    )

    # Drop the row where the sex is unknown
    cats = cats.filter(pl.col("sex_upon_intake") != "Unknown")

    # Prepare features for regression (time in shelter prediction)
    features_regression = [
        'age_upon_intake_(years)', 'sex_upon_intake', 'breed', 'color', 'intake_condition'
    ]
    X_regression = pd.get_dummies(cats[features_regression].to_pandas(), drop_first=True)
    y_regression = cats['time_in_shelter_days'].to_pandas()

    # Split and train random forest regressor
    X_train_reg, X_test_reg, y_train_reg, y_test_reg = train_test_split(
        X_regression, y_regression, test_size=0.3, random_state=42
    )

    rf_regressor = RandomForestRegressor(n_estimators=100, random_state=42)
    rf_regressor.fit(X_train_reg, y_train_reg)

    # Predict and evaluate random forest regressor
    predictions_reg = rf_regressor.predict(X_test_reg)
    mse = mean_squared_error(y_test_reg, predictions_reg)
    r2 = r2_score(y_test_reg, predictions_reg)

    additional_features_regression = [
        'dob_month'
    ]
    X_regression_extended = pd.get_dummies(
        cats[features_regression + additional_features_regression].to_pandas(), drop_first=True
    )

    X_with_const = sm.add_constant(X_regression_extended.astype(float))  # Add constant for intercept
    ols_model = sm.OLS(y_regression, X_with_const).fit()
    print(ols_model.summary())

    # Fix significant features extraction
    significant_features = ols_model.pvalues[ols_model.pvalues < 0.05].index
    print("\nSignificant Features:")
    print(significant_features)
    X_regression_significant = X_regression_extended[significant_features]

    # Remove near-zero variance features
    variance_filter = VarianceThreshold(threshold=0.01)
    X_reduced_variance = variance_filter.fit_transform(X_regression_significant)
    columns_retained = X_regression_significant.columns[variance_filter.get_support()]

    # Recreate the DataFrame with reduced features
    X_regression_cleaned = pd.DataFrame(X_reduced_variance, columns=columns_retained)

    # Recompute VIF
    vif_data_cleaned = pd.DataFrame()
    vif_data_cleaned["feature"] = X_regression_cleaned.columns
    vif_data_cleaned["VIF"] = [
        variance_inflation_factor(X_regression_cleaned.values, i)
        for i in range(X_regression_cleaned.shape[1])
    ]
    print("\nVariance Inflation Factor (VIF) after removing near-zero variance features:")
    print(vif_data_cleaned)

    # Remove features with high VIF (>10)
    low_vif_features_cleaned = vif_data_cleaned[vif_data_cleaned["VIF"] < 10]["feature"]
    X_regression_low_vif_cleaned = X_regression_cleaned[low_vif_features_cleaned]

    # Train Gradient Boosting with cleaned features
    X_train_reg_cleaned, X_test_reg_cleaned, y_train_reg_cleaned, y_test_reg_cleaned = train_test_split(
        X_regression_low_vif_cleaned, y_regression, test_size=0.3, random_state=42
    )

    gb_regressor_cleaned = GradientBoostingRegressor(n_estimators=200, learning_rate=0.1, random_state=42)
    gb_regressor_cleaned.fit(X_train_reg_cleaned, y_train_reg_cleaned)

    # Retrieve feature importances
    feature_importances = gb_regressor_cleaned.feature_importances_

    # Create a DataFrame for feature importance
    importance_df = pd.DataFrame({
        'Feature': X_regression_low_vif_cleaned.columns,
        'Importance': feature_importances
    }).sort_values(by='Importance', ascending=False)

    # Display the feature importance
    top_important_features = importance_df.head(10)
    print("Important Features for Predicting Time in Shelter:")
    print(top_important_features)

    # Plot the feature importance
    plt.figure(figsize=(10, 6))
    plt.barh(top_important_features['Feature'], top_important_features['Importance'])
    plt.xlabel('Importance')
    plt.ylabel('Feature')
    plt.title('Important Features')
    plt.gca().invert_yaxis()  # Reverse the y-axis for better readability
    plt.show()

    return importance_df

    # Function to make predictions based on the regression model
def predict_time_in_shelter(age, dob_month, sex_upon_intake, breed, color, intake_condition, df):
    '''
    Predict time in shelter for scenarios based on regression model

    Args:
    age (int): age of cat
    dob (int): date of birth of cat
    sex_upon_intake (str): sex of cat upon intake
    breed (str): breed of cat
    color (str): color of cat
    intake_condition (str): intake condition of cat
    df (pd.DataFrame): dataframe to use
    '''

    assert isinstance(df, pd.DataFrame), "df must be a pandas DataFrame"
    assert isinstance(age, int), "age must be an int"
    assert isinstance(dob_month, int), "dob_month must be an int"
    assert isinstance(breed, str), "breed must be a str"
    assert isinstance(color, str), "color must be a str"
    assert isinstance(intake_condition, str), "intake_condition must be a str"
    # Filter data for cats and create a binary target for adoption
    cats = df.filter(pl.col('animal_type') == 'Cat')
    cats = cats.with_columns(
        pl.Series((cats['outcome_type'] == 'Adoption').to_numpy().astype(int)).alias('adopted')
    )

    # Drop the row where the sex is unknown
    cats = cats.filter(pl.col("sex_upon_intake") != "Unknown")

    # Prepare features for regression (time in shelter prediction)
    features_regression = [
        'age_upon_intake_(years)', 'sex_upon_intake', 'breed', 'color', 'intake_condition'
    ]
    X_regression = pd.get_dummies(cats[features_regression].to_pandas(), drop_first=True)
    y_regression = cats['time_in_shelter_days'].to_pandas()

    # Split and train random forest regressor
    X_train_reg, X_test_reg, y_train_reg, y_test_reg = train_test_split(
        X_regression, y_regression, test_size=0.3, random_state=42
    )

    rf_regressor = RandomForestRegressor(n_estimators=100, random_state=42)
    rf_regressor.fit(X_train_reg, y_train_reg)

    # Predict and evaluate random forest regressor
    predictions_reg = rf_regressor.predict(X_test_reg)
    mse = mean_squared_error(y_test_reg, predictions_reg)
    r2 = r2_score(y_test_reg, predictions_reg)

    additional_features_regression = [
        'dob_month'
    ]
    X_regression_extended = pd.get_dummies(
        cats[features_regression + additional_features_regression].to_pandas(), drop_first=True
    )

    X_with_const = sm.add_constant(X_regression_extended.astype(float))  # Add constant for intercept
    ols_model = sm.OLS(y_regression, X_with_const).fit()
    print(ols_model.summary())

    # Fix significant features extraction
    significant_features = ols_model.pvalues[ols_model.pvalues < 0.05].index
    print("\nSignificant Features:")
    print(significant_features)
    X_regression_significant = X_regression_extended[significant_features]

    # Remove near-zero variance features
    variance_filter = VarianceThreshold(threshold=0.01)
    X_reduced_variance = variance_filter.fit_transform(X_regression_significant)
    columns_retained = X_regression_significant.columns[variance_filter.get_support()]

    # Recreate the DataFrame with reduced features
    X_regression_cleaned = pd.DataFrame(X_reduced_variance, columns=columns_retained)

    # Recompute VIF
    vif_data_cleaned = pd.DataFrame()
    vif_data_cleaned["feature"] = X_regression_cleaned.columns
    vif_data_cleaned["VIF"] = [
        variance_inflation_factor(X_regression_cleaned.values, i)
        for i in range(X_regression_cleaned.shape[1])
    ]
    print("\nVariance Inflation Factor (VIF) after removing near-zero variance features:")
    print(vif_data_cleaned)

    # Remove features with high VIF (>10)
    low_vif_features_cleaned = vif_data_cleaned[vif_data_cleaned["VIF"] < 10]["feature"]
    X_regression_low_vif_cleaned = X_regression_cleaned[low_vif_features_cleaned]

    # Train Gradient Boosting with cleaned features
    X_train_reg_cleaned, X_test_reg_cleaned, y_train_reg_cleaned, y_test_reg_cleaned = train_test_split(
        X_regression_low_vif_cleaned, y_regression, test_size=0.3, random_state=42
    )

    gb_regressor_cleaned = GradientBoostingRegressor(n_estimators=200, learning_rate=0.1, random_state=42)
    gb_regressor_cleaned.fit(X_train_reg_cleaned, y_train_reg_cleaned)

    # Create a feature dictionary with default values (0) for one-hot encoded features
    feature_dict = {feature: 0 for feature in X_regression_low_vif_cleaned.columns}
    
    # Assign values to the specified features
    feature_dict['age_upon_intake_(years)'] = age
    feature_dict['dob_month'] = dob_month
    feature_dict[f'sex_upon_intake_{sex_upon_intake}'] = 1
    feature_dict[f'breed_{breed}'] = 1
    feature_dict[f'color_{color}'] = 1
    feature_dict[f'intake_condition_{intake_condition}'] = 1
    
    # Convert dictionary to a DataFrame for prediction
    feature_vector = pd.DataFrame([feature_dict])
    
    # Ensure columns match the training set
    feature_vector = feature_vector.reindex(columns=X_regression_low_vif_cleaned.columns, fill_value=0)
    
    # Predict using the trained Gradient Boosting model
    predicted_days = gb_regressor_cleaned.predict(feature_vector)
    return predicted_days[0]

def predict_adoption(age, dob_month, sex_upon_intake, breed, color, intake_condition, df):
    '''
    Predict adoption outcome for scenarios based on regression model

    Args:
    age (int): age of cat
    dob (int): date of birth of cat
    sex_upon_intake (str): sex of cat upon intake
    breed (str): breed of cat
    color (str): color of cat
    intake_condition (str): intake condition of cat
    df (pd.DataFrame): dataframe to use
    '''
    assert isinstance(df, pd.DataFrame), "df must be a pandas DataFrame"
    assert isinstance(age, int), "age must be an int"
    assert isinstance(dob_month, int), "dob_month must be an int"
    assert isinstance(breed, str), "breed must be a str"
    assert isinstance(color, str), "color must be a str"
    assert isinstance(intake_condition, str), "intake_condition must be a str"

    cats = df.filter(pl.col('animal_type') == 'Cat')
    cats = cats.with_columns(
        pl.Series((cats['outcome_type'] == 'Adoption').to_numpy().astype(int)).alias('adopted')
    )

    # Drop the row where the sex is unknown
    cats = cats.filter(pl.col("sex_upon_intake") != "Unknown")

    # Prepare features for logistic regression (adoption likelihood)
    features_classification = [
        'age_upon_intake_(years)', 'sex_upon_intake', 'breed', 'color', 'intake_condition'
    ]
    X_classification = pd.get_dummies(cats[features_classification].to_pandas(), drop_first=True)
    y_classification = cats['adopted'].to_pandas()

    # Split and train logistic regression
    X_train_clf, X_test_clf, y_train_clf, y_test_clf = train_test_split(
        X_classification, y_classification, test_size=0.2, random_state=42
    )

    logistic_model = LogisticRegression(max_iter=1000)
    logistic_model.fit(X_train_clf, y_train_clf)

    # Predict and evaluate logistic regression
    # change the threshold to 0.6 for predicting
    predictions_clf = logistic_model.predict_proba(X_test_clf)[:, 1] > 0.5
    # Create a feature dictionary with default values (0) for one-hot encoded features
    feature_dict = {feature: 0 for feature in X_classification.columns}
    
    # Assign values to the specified features
    feature_dict['age_upon_intake_(years)'] = age
    feature_dict[f'sex_upon_intake_{sex_upon_intake}'] = 1
    feature_dict[f'breed_{breed}'] = 1
    feature_dict[f'color_{color}'] = 1
    feature_dict[f'intake_condition_{intake_condition}'] = 1
    
    # Convert dictionary to a DataFrame for prediction
    feature_vector = pd.DataFrame([feature_dict])
    
    # Ensure columns match the training set
    feature_vector = feature_vector.reindex(columns=X_classification.columns, fill_value=0)
    
    # Predict using the trained logistic regression model
    adoption_prob = logistic_model.predict_proba(feature_vector)[0, 1]
    return adoption_prob

def plot_seasonal_effect(df):
    '''
    Plot effects of seasons on cat adoption

    Args:
    df (pd.DataFrame): dataframe to use
    '''
    assert isinstance(df, pd.DataFrame), "df must be a pandas DataFrame"

    cats = df.filter(pl.col('animal_type') == 'Cat')
    cats = cats.with_columns(
        pl.Series((cats['outcome_type'] == 'Adoption').to_numpy().astype(int)).alias('adopted')
    )

    # Drop the row where the sex is unknown
    cats = cats.filter(pl.col("sex_upon_intake") != "Unknown")

    adopted_cats = cats.filter(pl.col('adopted') == 1).to_pandas()

    # Ensure outcome_datetime is in datetime format
    adopted_cats['outcome_datetime'] = pd.to_datetime(adopted_cats['outcome_datetime'])

    # Extract the month of adoption
    adopted_cats['outcome_month'] = adopted_cats['outcome_datetime'].dt.month

    # Aggregate adoptions by month
    monthly_adoption_counts = adopted_cats['outcome_month'].value_counts().sort_index()

    # Plot seasonal trends in cat adoptions
    plt.figure(figsize=(10, 6))
    plt.bar(monthly_adoption_counts.index,
            monthly_adoption_counts.values,
            tick_label=['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec'])
    plt.title('Seasonal Effect on Cat Adoptions')
    plt.xlabel('Month')
    plt.ylabel('Number of Adoptions')
    plt.xticks(rotation=45)
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.show()

    # Print monthly adoption counts for further analysis
    print(monthly_adoption_counts)









def main():
    df = load_and_preprocess_data("./aac_intakes_outcomes.csv")
    plot_outcome_distribution(df)
    euth_analysis = analyze_euthanasia_cases(df)
    plot_euthanasia_analysis(euth_analysis)
    plot_outcome_by_animal_type(df)
    plot_age_vs_time_in_shelter(df)

    
    for animal_type in ['Dog', 'Cat']:
        top_breeds = get_top_breeds(df, animal_type)
        plot_top_breeds(top_breeds, animal_type)
        
        adopted_animals = get_adopted_animals(df, animal_type)
        plot_age_distribution(adopted_animals, animal_type)
        
        top_colors = get_top_colors(adopted_animals)
        plot_top_colors(top_colors, animal_type)
        
        sex_counts = get_sex_distribution(adopted_animals)
        plot_sex_distribution(sex_counts, animal_type)

        # animal_outcome = analyze_specific_animal(df, animal_type)
        # plot_animal_outcome_distribution(animal_outcome, animal_type)

    model_age = age_vs_adoption("./aac_intakes_outcomes.csv")
    print(model_age.summary())
    # print(model_age.pvalues[1])

    model_gender = gender_vs_adoption("./aac_intakes_outcomes.csv")
    print(model_gender.summary())
    # print(model_gender.pvalues[1])
    # print(model_gender.pvalues[2])

    results_color = color_vs_adoption("./aac_intakes_outcomes.csv")
    print(results_color.summary())

    classification_report_cats = logistic_regression_cats("./aac_intakes_outcomes.csv")
    print(classification_report_cats)

    mse, r2 = random_forest_cats("./aac_intakes_outcomes.csv")
    print("\nRegression Metrics (Time in Shelter):")
    print(f"Mean Squared Error: {mse:.2f}")
    print(f"R^2 Score: {r2:.2f}")

    mse_cleaned, r2_cleaned = extended_regression("./aac_intakes_outcomes.csv")
    print("\nRegression Metrics (Cleaned Features):")
    print(f"Mean Squared Error: {mse_cleaned:.2f}")
    print(f"R^2 Score: {r2_cleaned:.2f}")

    plot_feature_importance(df)

    # Example scenarios
    scenarios = [
        {'age': 2, 'dob_month': 5, 'sex_upon_intake': 'Neutered Male', 'breed': 'Chartreux Mix', 'color': 'Blue/Tortie', 'intake_condition': 'Normal', 'df':df},
        {'age': 2, 'dob_month': 3, 'sex_upon_intake': 'Spayed Female', 'breed': 'Cymric Mix', 'color': 'Lynx Point/Tortie Point', 'intake_condition': 'Normal', 'df':df},
        {'age': 2, 'dob_month': 3, 'sex_upon_intake': 'Intact Female', 'breed': 'Cymric Mix', 'color': 'Lynx Point/Tortie Point', 'intake_condition': 'Normal', 'df':df},
        {'age': 15, 'dob_month': 3, 'sex_upon_intake': 'Spayed Female', 'breed': 'Cymric Mix', 'color': 'Lynx Point/Tortie Point', 'intake_condition': 'Normal', 'df':df},
        {'age': 5, 'dob_month': 10, 'sex_upon_intake': 'Neutered Male', 'breed': 'Pixiebob Shorthair Mix', 'color': 'Orange Tabby/Brown', 'intake_condition': 'Injured', 'df':df},
    ]

    # Make predictions for each scenario
    for i, scenario in enumerate(scenarios):
        predicted_days = predict_time_in_shelter(**scenario)
        print(f"Scenario {i+1}: Predicted time in shelter = {predicted_days:.2f} days")

    # Make predictions for each scenario
    for i, scenario in enumerate(scenarios):
        adoption_prob = predict_adoption(**scenario)
        print(f"Scenario {i+1}: Probability of adoption = {adoption_prob:.2f}")

    plot_seasonal_effect(df)



    



if __name__ == "__main__":
    main()