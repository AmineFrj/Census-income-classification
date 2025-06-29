import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder
from scipy.stats import chi2_contingency

def clean_string_columns(df):
    """Strip whitespace from all string columns."""
    for col in df.select_dtypes(include='object').columns:
        df[col] = df[col].str.strip()
    return df

def replace_missing_values(df, missing_values):
    """Replace specified non-informative values with np.nan."""
    return df.replace(missing_values, np.nan)

def drop_high_na_columns(df, threshold=0.85):
    """Drop columns with missing ratio above threshold."""
    missing_ratio = df.isna().mean()
    cols_to_drop = missing_ratio[missing_ratio > threshold].index.tolist()
    df = df.drop(columns=cols_to_drop)
    return df, cols_to_drop

def drop_duplicates(df, verbose=False):
    """Remove exact duplicate rows from DataFrame."""
    shape_before = df.shape[0]
    df = df.drop_duplicates()
    shape_after = df.shape[0]
    if verbose:    
        print(f"   - Removed {shape_before - shape_after} duplicate rows.")
    return df

def convert_to_category(df, columns):
    """Convert selected columns to pandas 'category' dtype."""
    for col in columns:
        if col in df.columns:
            df[col] = df[col].astype('category')
    return df

def group_rare_categories(df, column, min_freq=0.01):
    """Group rare categories (freq < min_freq) into 'Other'."""
    freq = df[column].value_counts(normalize=True)
    rare = freq[freq < min_freq].index
    df[column] = df[column].astype(str).replace(rare, "Other")
    return df

def get_high_cardinality_cols(df, cat_features, threshold=10):
    """Return categorical columns with more than 'threshold' unique values."""
    return [col for col in cat_features if df[col].nunique() > threshold]

def impute_missing_values(train_df, test_df):
    """Impute missing values for numeric (median) and categorical ('Missing')."""
    num_cols = train_df.select_dtypes(include=['int64', 'float64']).columns.tolist()
    num_imputer = SimpleImputer(strategy='median')
    train_df[num_cols] = num_imputer.fit_transform(train_df[num_cols])
    test_df[num_cols] = num_imputer.transform(test_df[num_cols])

    cat_cols = train_df.select_dtypes(include=['object', 'category']).columns.tolist()
    cat_imputer = SimpleImputer(strategy='constant', fill_value="Missing")
    train_df[cat_cols] = cat_imputer.fit_transform(train_df[cat_cols])
    test_df[cat_cols] = cat_imputer.transform(test_df[cat_cols])

    return train_df, test_df

def one_hot_encode(train_df, test_df, categorical_cols):
    """One-hot encode the categorical columns (returns X_train, X_test, encoder)"""
    for col in categorical_cols:
        train_df.loc[:, col] = train_df[col].astype(str)
        test_df.loc[:, col] = test_df[col].astype(str)
    encoder = OneHotEncoder(handle_unknown='ignore', sparse_output=False)
    encoder.fit(train_df[categorical_cols])
    X_train_ohe = encoder.transform(train_df[categorical_cols])
    X_test_ohe = encoder.transform(test_df[categorical_cols])
    ohe_cols = encoder.get_feature_names_out(categorical_cols)
    train_num = train_df.drop(columns=categorical_cols).reset_index(drop=True)
    test_num = test_df.drop(columns=categorical_cols).reset_index(drop=True)
    X_train_final = pd.concat([train_num, pd.DataFrame(X_train_ohe, columns=ohe_cols)], axis=1)
    X_test_final = pd.concat([test_num, pd.DataFrame(X_test_ohe, columns=ohe_cols)], axis=1)
    return X_train_final, X_test_final, encoder

def cramers_v(x, y):
    """Calculate Cramér's V statistic for categorical-categorical association."""
    confusion_matrix = pd.crosstab(x, y)
    chi2 = chi2_contingency(confusion_matrix)[0]
    n = confusion_matrix.sum().sum()
    phi2 = chi2 / n
    r, k = confusion_matrix.shape
    phi2corr = max(0, phi2 - ((k-1)*(r-1))/(n-1))
    rcorr = r - ((r-1)**2)/(n-1)
    kcorr = k - ((k-1)**2)/(n-1)
    return np.sqrt(phi2corr / min((kcorr-1), (rcorr-1)))

def cramers_v_matrix(df, cat_cols):
    """Compute a matrix of Cramér's V for all pairs of categorical columns."""
    cramers_matrix = pd.DataFrame(index=cat_cols, columns=cat_cols)
    for col1 in cat_cols:
        for col2 in cat_cols:
            if col1 == col2:
                cramers_matrix.loc[col1, col2] = 1.0
            else:
                try:
                    cramers_matrix.loc[col1, col2] = cramers_v(df[col1], df[col2])
                except Exception:
                    cramers_matrix.loc[col1, col2] = np.nan
    return cramers_matrix.astype(float)

def clean_column_names(df):
    df.columns = [col.replace('[', '_').replace(']', '_').replace('<', '_').replace(" ", "_") for col in df.columns]
    return df

def drop_columns(df, columns_to_drop):
    """Remove columns from a DataFrame."""
    return df.drop(columns=columns_to_drop, errors='ignore')

def preprocess_data(
    train_df,
    test_df=None,
    config=None,
    verbose=True
):
    """
    Preprocess train and (optionally) test DataFrames according to a config dict (YAML).
    Steps (all optional, controlled via config['preprocessing_steps']):
      - drop instance_weight column
      - clean string columns
      - replace missing values with np.nan
      - drop columns with too many missing
      - drop duplicate rows
      - convert categorical columns
      - group rare categories
      - impute missing values
      - one-hot encode (optionally)
    Returns: X_train, X_test, y_train, y_test, cat_features
    """
    steps = config.get('preprocessing_steps', {})
    col_names = config.get('col_names')
    missing_values = config.get('missing_values')
    cat_cols = config.get('cat_cols')
    columns_to_remove = config.get('columns_to_remove')

    # 0. Set column names
    if col_names is not None:
        train_df.columns = col_names
        if test_df is not None:
            test_df.columns = col_names

    # 1. Drop 'instance_weight'
    if steps.get('drop_instance_weight', True):
        for df in [train_df, test_df]:
            if df is not None and 'instance_weight' in df.columns:
                df.drop(columns=['instance_weight'], inplace=True)

    # 2. Clean string columns
    if steps.get('clean_strings', True):
        train_df = clean_string_columns(train_df)
        if test_df is not None:
            test_df = clean_string_columns(test_df)

    # 3. Replace missing values
    if steps.get('replace_missing', True):
        train_df = replace_missing_values(train_df, missing_values)
        if test_df is not None:
            test_df = replace_missing_values(test_df, missing_values)

    # 4. Drop high-NA columns
    if steps.get('drop_high_na_columns', True):
        train_df, dropped_cols = drop_high_na_columns(train_df)
        if test_df is not None:
            test_df = test_df.drop(columns=dropped_cols)

    # 5. Drop duplicates
    if steps.get('drop_duplicates', True):
        train_df = drop_duplicates(train_df)

    # 6. Convert to category
    if steps.get('convert_to_category', True):
        train_df = convert_to_category(train_df, cat_cols)
        for col in cat_cols:
            if col in train_df.columns:
                train_df[col] = train_df[col].astype(str)
        if test_df is not None:
            test_df = convert_to_category(test_df, cat_cols)
            for col in cat_cols:
                if col in test_df.columns:
                    test_df[col] = test_df[col].astype(str)
    # 7. Convert to category
    if steps.get('remove_correlated', True):
        train_df = drop_columns(train_df, columns_to_remove)
        if test_df is not None: 
            test_df = drop_columns(test_df, columns_to_remove)

    
    # 8. Group rare categories
    cat_features = train_df.select_dtypes(include=['category', 'object']).columns.tolist()
    if 'income' in cat_features:
        cat_features.remove('income')
    if steps.get('group_rare_categories', True):
        high_card_cols = get_high_cardinality_cols(train_df, cat_features, threshold=10)
        for col in high_card_cols:
            train_df = group_rare_categories(train_df, col, min_freq=0.01)
            if test_df is not None:
                test_df = group_rare_categories(test_df, col, min_freq=0.01)

    # 9. Impute missing values 
    if steps.get('impute_missing', True):
        train_df, test_df = impute_missing_values(train_df, test_df)
    else:
        for col in train_df.select_dtypes(include=['object','category']).columns.tolist():
            if col in train_df.columns:
                train_df[col] = train_df[col].astype(str).replace('nan', 'Missing').replace(np.nan, 'Missing')
                if test_df is not None:
                    test_df[col] = test_df[col].astype(str).replace('nan', 'Missing').replace(np.nan, 'Missing')
                    
    # 10. Create binary target
    train_df['income_binary'] = train_df['income'].apply(lambda x: 1 if '50000+' in str(x) else 0)
    if test_df is not None:
        test_df['income_binary'] = test_df['income'].apply(lambda x: 1 if '50000+' in str(x) else 0)

    # 11. Split X/y
    X_train = train_df.drop(columns=['income', 'income_binary'])
    y_train = train_df['income_binary']
    if test_df is not None:
        X_test = test_df.drop(columns=['income', 'income_binary'])
        y_test = test_df['income_binary']
    else:
        X_test, y_test = None, None

    # 12. One-hot encoding 
    if steps.get('one_hot_encode', False):
        X_train, X_test, encoder = one_hot_encode(X_train, X_test, cat_features)
        X_train = clean_column_names(X_train)
        X_test = clean_column_names(X_test)

    if verbose:
        print("✅ Preprocessing ended successfully.")

    return X_train, X_test, y_train, y_test, cat_features
