import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import Ridge, Lasso, ElasticNet
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import cross_val_score, KFold
from sklearn.metrics import r2_score, mean_squared_error
from scipy import stats
import warnings
warnings.filterwarnings('ignore')

# Set plotting style
sns.set_theme(style='whitegrid', context='talk', palette='colorblind')
plt.rcParams['figure.dpi'] = 300
plt.rcParams['savefig.dpi'] = 300

# ============================================================================
# SECTION 1: DATA LOADING AND PREPARATION
# ============================================================================

def load_and_prepare_data():
    """
    Load both datasets and perform initial cleaning and preparation.

    Returns:
        bat_df: DataFrame with bat-level observations
        site_df: DataFrame with site-level time-indexed summaries
    """
    print("=" * 80)
    print("SECTION 1: DATA LOADING AND PREPARATION")
    print("=" * 80)

    # Load datasets with European date format (day-first)
    bat_df = pd.read_csv('dataSet_1.csv')
    site_df = pd.read_csv('dataSet_2.csv')

    print(f"\nDataset 1 (Bat-level) shape: {bat_df.shape}")
    print(f"Dataset 2 (Site-level) shape: {site_df.shape}")

    # Parse datetime columns
    date_cols_bat = ['start_time', 'rat_period_start', 'rat_period_end', 'sunset_time']
    for col in date_cols_bat:
        if col in bat_df.columns:
            bat_df[col] = pd.to_datetime(bat_df[col], dayfirst=True, errors='coerce')

    if 'time' in site_df.columns:
        site_df['time'] = pd.to_datetime(site_df['time'], dayfirst=True, errors='coerce')

    print("\nDatetime columns parsed successfully.")

    return bat_df, site_df


def merge_datasets(bat_df, site_df):
    """
    Merge bat-level observations with nearest site-level context.

    Args:
        bat_df: Bat-level DataFrame
        site_df: Site-level DataFrame

    Returns:
        merged_df: Enriched DataFrame with both bat and site variables
    """
    print("\n" + "=" * 80)
    print("SECTION 2: DATASET MERGING")
    print("=" * 80)

    # Sort both datasets by time
    bat_sorted = bat_df.sort_values('start_time').copy()
    site_sorted = site_df.sort_values('time').copy()

    # Merge using nearest time match (30-minute tolerance)
    merged = pd.merge_asof(
        bat_sorted, 
        site_sorted,
        left_on='start_time', 
        right_on='time',
        direction='nearest',
        tolerance=pd.Timedelta('30min'),
        suffixes=('', '_site')
    )

    print(f"\nMerged dataset shape: {merged.shape}")
    print(f"Rows with successful merge: {merged['time'].notna().sum()}")

    return merged


def engineer_features(df):
    """
    Create advanced engineered features for modeling.

    Args:
        df: Merged DataFrame

    Returns:
        df: DataFrame with additional engineered features
    """
    print("\n" + "=" * 80)
    print("SECTION 3: FEATURE ENGINEERING")
    print("=" * 80)

    # Group sparse habit categories
    if 'habit' in df.columns:
        habit_counts = df['habit'].value_counts()
        rare_habits = habit_counts[habit_counts < 20].index
        df['habit_grouped'] = df['habit'].apply(
            lambda x: 'other' if x in rare_habits else x
        )
        print(f"\nHabit categories grouped. Unique values: {df['habit_grouped'].nunique()}")

    # Create interaction features
    if 'rat_minutes' in df.columns and 'food_availability' in df.columns:
        df['rat_food_interaction'] = df['rat_minutes'] * df['food_availability']
        print("Created rat_food_interaction feature")

    if 'hours_after_sunset' in df.columns and 'bat_landing_number' in df.columns:
        df['time_activity_interaction'] = df['hours_after_sunset'] * df['bat_landing_number']
        print("Created time_activity_interaction feature")

    # Log-transform skewed count variables
    count_vars = ['rat_arrival_number', 'bat_landing_number']
    for var in count_vars:
        if var in df.columns:
            df[f'{var}_log'] = np.log1p(df[var])
            print(f"Created log-transformed {var}")

    # Squared terms for potential nonlinearity
    if 'hours_after_sunset' in df.columns:
        df['hours_after_sunset_sq'] = df['hours_after_sunset'] ** 2
        print("Created hours_after_sunset_sq feature")

    print(f"\nFinal dataset shape after feature engineering: {df.shape}")

    return df


def clean_data(df, response_var='bat_landing_to_food'):
    """
    Clean data by handling missing values and outliers.

    Args:
        df: DataFrame to clean
        response_var: Name of response variable

    Returns:
        df_clean: Cleaned DataFrame
    """
    print("\n" + "=" * 80)
    print("SECTION 4: DATA CLEANING")
    print("=" * 80)

    initial_rows = len(df)

    # Drop rows with missing response variable
    df_clean = df.dropna(subset=[response_var]).copy()
    print(f"\nRows dropped due to missing response: {initial_rows - len(df_clean)}")

    # Handle extreme outliers in response (beyond 3 IQR)
    Q1 = df_clean[response_var].quantile(0.25)
    Q3 = df_clean[response_var].quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 3 * IQR
    upper_bound = Q3 + 3 * IQR

    outliers = (df_clean[response_var] < lower_bound) | (df_clean[response_var] > upper_bound)
    print(f"Extreme outliers identified: {outliers.sum()}")

    # Keep outliers but flag them for sensitivity analysis
    df_clean['is_outlier'] = outliers

    print(f"\nFinal clean dataset shape: {df_clean.shape}")
    print(f"Missing values per column:\n{df_clean.isnull().sum()[df_clean.isnull().sum() > 0]}")

    return df_clean


# ============================================================================
# SECTION 5: EXPLORATORY DATA ANALYSIS
# ============================================================================

def create_correlation_heatmap(df, output_file='correlation_heatmap.svg'):
    """
    Create correlation heatmap of numeric variables.

    Args:
        df: DataFrame
        output_file: Output filename for SVG
    """
    print("\n" + "=" * 80)
    print("SECTION 5: CORRELATION ANALYSIS")
    print("=" * 80)

    # Select numeric columns
    numeric_cols = df.select_dtypes(include=[np.number]).columns

    # Remove constant columns
    numeric_df = df[numeric_cols]
    constant_cols = [c for c in numeric_df.columns if numeric_df[c].nunique(dropna=True) <= 1]
    if constant_cols:
        numeric_df = numeric_df.drop(columns=constant_cols)

    # Limit to top 30 variables by variance for readability
    if numeric_df.shape[1] > 30:
        var_order = numeric_df.var().sort_values(ascending=False).index[:30]
        numeric_df = numeric_df[var_order]

    # Compute correlation matrix
    corr_matrix = numeric_df.corr(method='pearson')

    print(f"\nCorrelation matrix computed for {corr_matrix.shape[0]} variables")

    # Create heatmap
    mask = np.triu(np.ones_like(corr_matrix, dtype=bool))

    fig, ax = plt.subplots(figsize=(14, 12))
    sns.heatmap(
        corr_matrix,
        mask=mask,
        cmap='coolwarm',
        vmin=-1, vmax=1,
        center=0,
        square=True,
        linewidths=0.5,
        cbar_kws={'shrink': 0.8, 'label': 'Pearson Correlation'},
        annot=True,
        fmt='.2f',
        annot_kws={'size': 7}
    )

    plt.title('Correlation Heatmap of Numeric Variables', fontsize=16, weight='bold', pad=20)
    plt.tight_layout()
    plt.savefig(output_file, format='svg', bbox_inches='tight')
    plt.close()

    print(f"Correlation heatmap saved: {output_file}")

    return corr_matrix


def descriptive_statistics(df, response_var='bat_landing_to_food'):
    """
    Generate descriptive statistics for key variables.

    Args:
        df: DataFrame
        response_var: Response variable name
    """
    print("\n" + "=" * 80)
    print("SECTION 6: DESCRIPTIVE STATISTICS")
    print("=" * 80)

    print(f"\nResponse Variable: {response_var}")
    print(df[response_var].describe())

    # Key predictors
    key_vars = ['rat_minutes', 'rat_arrival_number', 'seconds_after_rat_arrival',
                'hours_after_sunset', 'food_availability', 'bat_landing_number']

    available_vars = [v for v in key_vars if v in df.columns]

    print("\nKey Predictor Variables:")
    print(df[available_vars].describe())

    # Categorical variables
    if 'season' in df.columns:
        print("\nSeason distribution:")
        print(df['season'].value_counts())

    if 'habit_grouped' in df.columns:
        print("\nHabit (grouped) distribution:")
        print(df['habit_grouped'].value_counts())


# ============================================================================
# SECTION 7: MODEL PREPARATION
# ============================================================================

def prepare_modeling_data(df, response_var='bat_landing_to_food'):
    """
    Prepare features and response for modeling.

    Args:
        df: DataFrame
        response_var: Response variable name

    Returns:
        X: Feature matrix
        y: Response vector
        feature_names: List of feature names
        scaler: Fitted StandardScaler
    """
    print("\n" + "=" * 80)
    print("SECTION 7: MODEL PREPARATION")
    print("=" * 80)

    # Define predictor variables
    numeric_predictors = [
        'rat_minutes', 'rat_arrival_number', 'rat_arrival_number_log',
        'seconds_after_rat_arrival', 'hours_after_sunset', 'hours_after_sunset_sq',
        'food_availability', 'bat_landing_number', 'bat_landing_number_log',
        'rat_food_interaction', 'time_activity_interaction'
    ]

    # Filter to available columns
    available_predictors = [p for p in numeric_predictors if p in df.columns]

    # Add categorical variables (one-hot encoded)
    categorical_predictors = []
    if 'habit_grouped' in df.columns:
        categorical_predictors.append('habit_grouped')
    if 'season' in df.columns:
        categorical_predictors.append('season')

    # Create feature matrix
    X_numeric = df[available_predictors].copy()

    # One-hot encode categoricals
    if categorical_predictors:
        X_categorical = pd.get_dummies(df[categorical_predictors], drop_first=True)
        X = pd.concat([X_numeric, X_categorical], axis=1)
    else:
        X = X_numeric

    # Handle missing values (fill with median for numeric)
    X = X.fillna(X.median())

    # Response variable
    y = df[response_var].values

    # Standardize features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    X_scaled = pd.DataFrame(X_scaled, columns=X.columns, index=X.index)

    print(f"\nFeature matrix shape: {X_scaled.shape}")
    print(f"Response vector shape: {y.shape}")
    print(f"\nFeatures included: {list(X_scaled.columns)}")

    return X_scaled, y, list(X.columns), scaler


# ============================================================================
# SECTION 8: MODEL TRAINING AND COMPARISON
# ============================================================================

def compare_models(X, y, cv_folds=5):
    """
    Compare Ridge, Lasso, and ElasticNet models using cross-validation.

    Args:
        X: Feature matrix
        y: Response vector
        cv_folds: Number of cross-validation folds

    Returns:
        results: Dictionary with model performance metrics
        best_model: Best performing model
    """
    print("\n" + "=" * 80)
    print("SECTION 8: MODEL COMPARISON")
    print("=" * 80)

    # Define models with different alpha values to test
    alphas = np.logspace(-3, 3, 50)

    models = {
        'Ridge': Ridge(),
        'Lasso': Lasso(max_iter=10000),
        'ElasticNet': ElasticNet(max_iter=10000)
    }

    results = {}
    cv = KFold(n_splits=cv_folds, shuffle=True, random_state=42)

    for name, model in models.items():
        print(f"\nEvaluating {name}...")

        best_alpha = None
        best_score = -np.inf

        for alpha in alphas:
            model.set_params(alpha=alpha)
            scores = cross_val_score(model, X, y, cv=cv, scoring='r2')
            mean_score = scores.mean()

            if mean_score > best_score:
                best_score = mean_score
                best_alpha = alpha

        # Refit with best alpha
        model.set_params(alpha=best_alpha)
        model.fit(X, y)

        # Calculate metrics
        y_pred = model.predict(X)
        train_r2 = r2_score(y, y_pred)
        train_rmse = np.sqrt(mean_squared_error(y, y_pred))

        # Cross-validated metrics
        cv_scores = cross_val_score(model, X, y, cv=cv, scoring='r2')
        cv_r2_mean = cv_scores.mean()
        cv_r2_std = cv_scores.std()

        results[name] = {
            'model': model,
            'best_alpha': best_alpha,
            'train_r2': train_r2,
            'train_rmse': train_rmse,
            'cv_r2_mean': cv_r2_mean,
            'cv_r2_std': cv_r2_std
        }

        print(f"  Best alpha: {best_alpha:.4f}")
        print(f"  Train R²: {train_r2:.4f}")
        print(f"  CV R² (mean ± std): {cv_r2_mean:.4f} ± {cv_r2_std:.4f}")

    # Select best model
    best_model_name = max(results, key=lambda k: results[k]['cv_r2_mean'])
    best_model = results[best_model_name]['model']

    print(f"\n{'='*80}")
    print(f"BEST MODEL: {best_model_name}")
    print(f"{'='*80}")

    return results, best_model, best_model_name


def create_model_comparison_plot(results, output_file='model_comparison_investigation_a.svg'):
    """
    Create visualization comparing model performance.

    Args:
        results: Dictionary with model results
        output_file: Output filename
    """
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    models = list(results.keys())
    cv_r2 = [results[m]['cv_r2_mean'] for m in models]
    cv_r2_std = [results[m]['cv_r2_std'] for m in models]
    train_r2 = [results[m]['train_r2'] for m in models]

    # R² comparison
    x = np.arange(len(models))
    width = 0.35

    axes[0].bar(x - width/2, train_r2, width, label='Train R²', alpha=0.8)
    axes[0].bar(x + width/2, cv_r2, width, label='CV R²', alpha=0.8, yerr=cv_r2_std)
    axes[0].set_xlabel('Model', fontsize=12)
    axes[0].set_ylabel('R² Score', fontsize=12)
    axes[0].set_title('Model Performance Comparison', fontsize=14, weight='bold')
    axes[0].set_xticks(x)
    axes[0].set_xticklabels(models)
    axes[0].legend()
    axes[0].grid(axis='y', alpha=0.3)

    # RMSE comparison
    train_rmse = [results[m]['train_rmse'] for m in models]
    axes[1].bar(models, train_rmse, alpha=0.8, color='coral')
    axes[1].set_xlabel('Model', fontsize=12)
    axes[1].set_ylabel('RMSE', fontsize=12)
    axes[1].set_title('Training RMSE Comparison', fontsize=14, weight='bold')
    axes[1].grid(axis='y', alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_file, format='svg', bbox_inches='tight')
    plt.close()

    print(f"\nModel comparison plot saved: {output_file}")


# ============================================================================
# SECTION 9: MODEL DIAGNOSTICS
# ============================================================================

def create_diagnostic_plots(model, X, y, output_file='model_diagnostics_investigation_a.svg'):
    """
    Create comprehensive diagnostic plots for the model.

    Args:
        model: Fitted model
        X: Feature matrix
        y: Response vector
        output_file: Output filename
    """
    print("\n" + "=" * 80)
    print("SECTION 9: MODEL DIAGNOSTICS")
    print("=" * 80)

    # Predictions and residuals
    y_pred = model.predict(X)
    residuals = y - y_pred

    # Standardized residuals
    residuals_std = residuals / np.std(residuals)

    # Create diagnostic plots
    fig, axes = plt.subplots(2, 2, figsize=(14, 12))

    # 1. Residuals vs Fitted
    axes[0, 0].scatter(y_pred, residuals, alpha=0.5, s=20)
    axes[0, 0].axhline(y=0, color='r', linestyle='--', linewidth=2)
    axes[0, 0].set_xlabel('Fitted Values', fontsize=11)
    axes[0, 0].set_ylabel('Residuals', fontsize=11)
    axes[0, 0].set_title('Residuals vs Fitted', fontsize=13, weight='bold')
    axes[0, 0].grid(alpha=0.3)

    # 2. Q-Q Plot
    stats.probplot(residuals, dist="norm", plot=axes[0, 1])
    axes[0, 1].set_title('Normal Q-Q Plot', fontsize=13, weight='bold')
    axes[0, 1].grid(alpha=0.3)

    # 3. Scale-Location
    axes[1, 0].scatter(y_pred, np.sqrt(np.abs(residuals_std)), alpha=0.5, s=20)
    axes[1, 0].set_xlabel('Fitted Values', fontsize=11)
    axes[1, 0].set_ylabel('√|Standardized Residuals|', fontsize=11)
    axes[1, 0].set_title('Scale-Location', fontsize=13, weight='bold')
    axes[1, 0].grid(alpha=0.3)

    # 4. Residuals vs Leverage
    # Calculate leverage (hat values)
    if hasattr(X, 'values'):
        X_array = X.values
    else:
        X_array = X

    # Add intercept column
    X_with_intercept = np.column_stack([np.ones(X_array.shape[0]), X_array])

    # Hat matrix diagonal
    try:
        hat_matrix = X_with_intercept @ np.linalg.pinv(X_with_intercept.T @ X_with_intercept) @ X_with_intercept.T
        leverage = np.diag(hat_matrix)
    except:
        leverage = np.ones(len(y)) * (X_array.shape[1] + 1) / len(y)

    axes[1, 1].scatter(leverage, residuals_std, alpha=0.5, s=20)
    axes[1, 1].axhline(y=0, color='r', linestyle='--', linewidth=2)
    axes[1, 1].set_xlabel('Leverage', fontsize=11)
    axes[1, 1].set_ylabel('Standardized Residuals', fontsize=11)
    axes[1, 1].set_title('Residuals vs Leverage', fontsize=13, weight='bold')
    axes[1, 1].grid(alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_file, format='svg', bbox_inches='tight')
    plt.close()

    print(f"Diagnostic plots saved: {output_file}")

    # Print diagnostic statistics
    print(f"\nDiagnostic Statistics:")
    print(f"  Mean residual: {np.mean(residuals):.6f}")
    print(f"  Std residual: {np.std(residuals):.4f}")
    print(f"  Skewness: {stats.skew(residuals):.4f}")
    print(f"  Kurtosis: {stats.kurtosis(residuals):.4f}")


def create_coefficient_plot(model, feature_names, output_file='investigation_a_coefficients_improved.svg'):
    """
    Create coefficient plot with confidence intervals.

    Args:
        model: Fitted Ridge model
        feature_names: List of feature names
        output_file: Output filename
    """
    print("\n" + "=" * 80)
    print("SECTION 10: COEFFICIENT ANALYSIS")
    print("=" * 80)

    # Get coefficients
    coefs = model.coef_

    # Sort by absolute value
    sorted_idx = np.argsort(np.abs(coefs))[::-1]
    sorted_coefs = coefs[sorted_idx]
    sorted_names = [feature_names[i] for i in sorted_idx]

    # Create plot
    fig, ax = plt.subplots(figsize=(10, max(8, len(feature_names) * 0.3)))

    colors = ['red' if c < 0 else 'blue' for c in sorted_coefs]

    y_pos = np.arange(len(sorted_names))
    ax.barh(y_pos, sorted_coefs, color=colors, alpha=0.7)
    ax.set_yticks(y_pos)
    ax.set_yticklabels(sorted_names, fontsize=9)
    ax.set_xlabel('Standardized Coefficient', fontsize=12)
    ax.set_title('Ridge Regression Coefficients (Investigation A)', fontsize=14, weight='bold')
    ax.axvline(x=0, color='black', linestyle='-', linewidth=0.8)
    ax.grid(axis='x', alpha=0.3)

    # Add legend
    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor='blue', alpha=0.7, label='Positive'),
        Patch(facecolor='red', alpha=0.7, label='Negative')
    ]
    ax.legend(handles=legend_elements, loc='best')

    plt.tight_layout()
    plt.savefig(output_file, format='svg', bbox_inches='tight')
    plt.close()

    print(f"Coefficient plot saved: {output_file}")

    # Print top coefficients
    print("\nTop 10 Coefficients by Absolute Value:")
    for i in range(min(10, len(sorted_names))):
        print(f"  {sorted_names[i]:40s}: {sorted_coefs[i]:8.4f}")


# ============================================================================
# SECTION 11: INVESTIGATION B - SEASONAL ANALYSIS
# ============================================================================

def seasonal_analysis(df, response_var='bat_landing_to_food'):
    """
    Perform seasonal analysis comparing winter and spring models.

    Args:
        df: DataFrame
        response_var: Response variable name

    Returns:
        seasonal_results: Dictionary with results for each season
    """
    print("\n" + "=" * 80)
    print("SECTION 11: INVESTIGATION B - SEASONAL ANALYSIS")
    print("=" * 80)

    if 'season' not in df.columns:
        print("Warning: 'season' column not found. Skipping seasonal analysis.")
        return None

    seasonal_results = {}

    # Get unique seasons
    seasons = df['season'].unique()
    print(f"\nSeasons found: {seasons}")

    for season in seasons:
        print(f"\n{'='*80}")
        print(f"ANALYZING SEASON: {season}")
        print(f"{'='*80}")

        # Subset data
        season_df = df[df['season'] == season].copy()
        print(f"Season {season} sample size: {len(season_df)}")

        if len(season_df) < 50:
            print(f"Warning: Sample size too small for season {season}. Skipping.")
            continue

        # Prepare data (without season as predictor)
        X, y, feature_names, scaler = prepare_modeling_data(season_df, response_var)

        # Remove season-related features if present
        season_cols = [col for col in X.columns if 'season' in col.lower()]
        if season_cols:
            X = X.drop(columns=season_cols)
            feature_names = [f for f in feature_names if f not in season_cols]

        # Train Ridge model
        alphas = np.logspace(-3, 3, 50)
        best_alpha = None
        best_score = -np.inf

        cv = KFold(n_splits=5, shuffle=True, random_state=42)

        for alpha in alphas:
            model = Ridge(alpha=alpha)
            scores = cross_val_score(model, X, y, cv=cv, scoring='r2')
            mean_score = scores.mean()

            if mean_score > best_score:
                best_score = mean_score
                best_alpha = alpha

        # Fit final model
        model = Ridge(alpha=best_alpha)
        model.fit(X, y)

        # Calculate metrics
        y_pred = model.predict(X)
        train_r2 = r2_score(y, y_pred)
        train_rmse = np.sqrt(mean_squared_error(y, y_pred))

        cv_scores = cross_val_score(model, X, y, cv=cv, scoring='r2')
        cv_r2_mean = cv_scores.mean()
        cv_r2_std = cv_scores.std()

        print(f"\nSeason {season} Results:")
        print(f"  Best alpha: {best_alpha:.4f}")
        print(f"  Train R²: {train_r2:.4f}")
        print(f"  CV R² (mean ± std): {cv_r2_mean:.4f} ± {cv_r2_std:.4f}")
        print(f"  Train RMSE: {train_rmse:.4f}")

        seasonal_results[season] = {
            'model': model,
            'X': X,
            'y': y,
            'feature_names': feature_names,
            'best_alpha': best_alpha,
            'train_r2': train_r2,
            'train_rmse': train_rmse,
            'cv_r2_mean': cv_r2_mean,
            'cv_r2_std': cv_r2_std,
            'coefficients': model.coef_
        }

    return seasonal_results


def create_seasonal_comparison_plot(seasonal_results, output_file='investigation_b_comprehensive.svg'):
    """
    Create comprehensive seasonal comparison visualization.

    Args:
        seasonal_results: Dictionary with seasonal model results
        output_file: Output filename
    """
    print("\n" + "=" * 80)
    print("SECTION 12: SEASONAL COMPARISON VISUALIZATION")
    print("=" * 80)

    if not seasonal_results or len(seasonal_results) < 2:
        print("Insufficient seasonal data for comparison plot.")
        return

    seasons = list(seasonal_results.keys())

    # Create comprehensive figure
    fig = plt.figure(figsize=(16, 12))
    gs = fig.add_gridspec(3, 2, hspace=0.3, wspace=0.3)

    # 1. Model Performance Comparison
    ax1 = fig.add_subplot(gs[0, :])

    season_names = []
    cv_r2_values = []
    cv_r2_errors = []
    train_r2_values = []

    for season in seasons:
        season_names.append(f"Season {season}")
        cv_r2_values.append(seasonal_results[season]['cv_r2_mean'])
        cv_r2_errors.append(seasonal_results[season]['cv_r2_std'])
        train_r2_values.append(seasonal_results[season]['train_r2'])

    x = np.arange(len(season_names))
    width = 0.35

    ax1.bar(x - width/2, train_r2_values, width, label='Train R²', alpha=0.8)
    ax1.bar(x + width/2, cv_r2_values, width, label='CV R²', alpha=0.8, yerr=cv_r2_errors)
    ax1.set_xlabel('Season', fontsize=12)
    ax1.set_ylabel('R² Score', fontsize=12)
    ax1.set_title('Seasonal Model Performance Comparison', fontsize=14, weight='bold')
    ax1.set_xticks(x)
    ax1.set_xticklabels(season_names)
    ax1.legend()
    ax1.grid(axis='y', alpha=0.3)

    # 2 & 3. Coefficient Comparison for each season
    for idx, season in enumerate(seasons[:2]):  # Show first two seasons
        ax = fig.add_subplot(gs[1, idx])

        coefs = seasonal_results[season]['coefficients']
        feature_names = seasonal_results[season]['feature_names']

        # Sort by absolute value, show top 15
        sorted_idx = np.argsort(np.abs(coefs))[::-1][:15]
        sorted_coefs = coefs[sorted_idx]
        sorted_names = [feature_names[i] for i in sorted_idx]

        colors = ['red' if c < 0 else 'blue' for c in sorted_coefs]

        y_pos = np.arange(len(sorted_names))
        ax.barh(y_pos, sorted_coefs, color=colors, alpha=0.7)
        ax.set_yticks(y_pos)
        ax.set_yticklabels(sorted_names, fontsize=8)
        ax.set_xlabel('Coefficient', fontsize=10)
        ax.set_title(f'Season {season} - Top 15 Coefficients', fontsize=12, weight='bold')
        ax.axvline(x=0, color='black', linestyle='-', linewidth=0.8)
        ax.grid(axis='x', alpha=0.3)

    # 4 & 5. Diagnostic plots for each season
    for idx, season in enumerate(seasons[:2]):
        ax = fig.add_subplot(gs[2, idx])

        model = seasonal_results[season]['model']
        X = seasonal_results[season]['X']
        y = seasonal_results[season]['y']

        y_pred = model.predict(X)
        residuals = y - y_pred

        ax.scatter(y_pred, residuals, alpha=0.5, s=15)
        ax.axhline(y=0, color='r', linestyle='--', linewidth=2)
        ax.set_xlabel('Fitted Values', fontsize=10)
        ax.set_ylabel('Residuals', fontsize=10)
        ax.set_title(f'Season {season} - Residuals vs Fitted', fontsize=12, weight='bold')
        ax.grid(alpha=0.3)

    plt.savefig(output_file, format='svg', bbox_inches='tight')
    plt.close()

    print(f"Seasonal comparison plot saved: {output_file}")


# ============================================================================
# MAIN EXECUTION PIPELINE
# ============================================================================

def main():
    """
    Main execution pipeline for the complete analysis.
    """
    print("\n" + "="*80)
    print("HIT BAT RAT RESEARCH A3: COMPLETE ANALYSIS PIPELINE")
    print("Egyptian Fruit Bats and Black Rats Interaction Study")
    print("="*80)

    # Step 1: Load and prepare data
    bat_df, site_df = load_and_prepare_data()

    # Step 2: Merge datasets
    merged_df = merge_datasets(bat_df, site_df)

    # Step 3: Engineer features
    enriched_df = engineer_features(merged_df)

    # Step 4: Clean data
    clean_df = clean_data(enriched_df, response_var='bat_landing_to_food')

    # Step 5: Exploratory analysis
    corr_matrix = create_correlation_heatmap(clean_df)
    descriptive_statistics(clean_df)

    # Step 6: Prepare modeling data
    X, y, feature_names, scaler = prepare_modeling_data(clean_df)

    # Step 7: INVESTIGATION A - Compare models
    print("\n" + "="*80)
    print("INVESTIGATION A: PREDATOR VS COMPETITOR ANALYSIS")
    print("="*80)

    model_results, best_model, best_model_name = compare_models(X, y)
    create_model_comparison_plot(model_results)

    # Step 8: Model diagnostics
    create_diagnostic_plots(best_model, X, y)

    # Step 9: Coefficient analysis
    create_coefficient_plot(best_model, feature_names)

    # Step 10: INVESTIGATION B - Seasonal analysis
    print("\n" + "="*80)
    print("INVESTIGATION B: SEASONAL VARIATION ANALYSIS")
    print("="*80)

    seasonal_results = seasonal_analysis(clean_df)

    if seasonal_results:
        create_seasonal_comparison_plot(seasonal_results)

    # Final summary
    print("\n" + "="*80)
    print("ANALYSIS COMPLETE")
    print("="*80)
    print("\nGenerated files:")
    print("  - correlation_heatmap.svg")
    print("  - model_comparison_investigation_a.svg")
    print("  - model_diagnostics_investigation_a.svg")
    print("  - investigation_a_coefficients_improved.svg")
    print("  - investigation_b_comprehensive.svg")
    print("\nAll visualizations have been saved as SVG files.")
    print("="*80)


if __name__ == "__main__":
    main()
