import pandas as pd

def sample_rows(df, percent_rows):
    '''
    Chooses some percentage of the dataframe's rows at random.
    Note that class balance is preserved. 
    
    Parameters
    ----------------------
    df (type: pd.DataFrame)
    percent_rows (type: float, range: 0-1)
    
    Returns
    ----------------------
    pd.DataFrame
    - Contains percent_rows of each class in input df
    '''
    
    labels = df['label'].unique()
    dfs_condensed = []
    
    # Select rows with chosen label
    for label in labels:
        mask = df['label'] == label
        df_by_label = df[mask]
        
        # Randomly sample some percentage of rows in current class
        sample = df_by_label.sample(frac = percent_rows)
        dfs_condensed.append(sample)
    
    # Shuffle all samples
    return pd.concat(dfs_condensed, ignore_index = True).sample(frac = 1)



def sample_rows2(dfx, dfy, percent_rows):
    '''
    Chooses some percentage of the DataFrame's rows at random.
    The class balance is preserved. 
    
    Parameters
    ----------------------
    dfx : pd.DataFrame
        DataFrame containing features.
    dfy : pd.Series
        Series containing labels.
    percent_rows : float
        Percentage of rows to sample (range: 0-1).
    
    Returns
    ----------------------
    pd.DataFrame, pd.Series
        Sampled DataFrame and corresponding Series.
    '''
    # Combine dfx and dfy into a single DataFrame
    df = pd.concat([dfx, dfy], axis=1)
    
    # Identify unique labels
    labels = dfy.unique()
    
    dfs_condensed = []
    
    # Select rows with chosen label
    for label in labels:
        mask = dfy == label
        df_by_label = df[mask]
        
        # Randomly sample some percentage of rows in current class
        sample = df_by_label.sample(frac=percent_rows)
        dfs_condensed.append(sample)
    
    # Concatenate sampled subsets and shuffle
    sampled_df = pd.concat(dfs_condensed).sample(frac=1).reset_index(drop=True)
    
    # Split sampled DataFrame back into dfx and dfy
    sampled_dfx = sampled_df.drop(columns=[dfy.name])
    sampled_dfy = sampled_df[dfy.name]
    
    return sampled_dfx, sampled_dfy
