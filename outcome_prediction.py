import polars as pl
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

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

if __name__ == "__main__":
    main()