"""
Data Validator Module for ML Trading System

This module provides comprehensive data quality checks, outlier detection,
and data quality report generation.
"""

import logging
from datetime import timedelta
from typing import Dict, List
import pandas as pd
import numpy as np

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='[%(levelname)s] %(message)s'
)
logger = logging.getLogger(__name__)


def check_missing_values(df: pd.DataFrame) -> Dict:
    """
    Check for missing values in the DataFrame.

    Args:
        df: DataFrame to check

    Returns:
        Dictionary with missing value information:
        {
            'total_missing': int,
            'missing_by_column': dict,
            'missing_percentage': float
        }

    Example:
        >>> report = check_missing_values(df)
        >>> print(f"Total missing: {report['total_missing']}")
    """
    total_missing = df.isna().sum().sum()
    missing_by_column = df.isna().sum().to_dict()

    # Filter out columns with no missing values
    missing_by_column = {k: v for k, v in missing_by_column.items() if v > 0}

    total_cells = df.shape[0] * df.shape[1]
    missing_percentage = (total_missing / total_cells * 100) if total_cells > 0 else 0

    report = {
        'total_missing': int(total_missing),
        'missing_by_column': missing_by_column,
        'missing_percentage': round(missing_percentage, 2)
    }

    if total_missing > 0:
        logger.warning(f"Found {total_missing} missing values ({missing_percentage:.2f}%)")
        for col, count in missing_by_column.items():
            logger.warning(f"  - {col}: {count} missing")
    else:
        logger.info("✓ No missing values found")

    return report


def check_date_gaps(df: pd.DataFrame, max_gap_days: int = 5) -> Dict:
    """
    Identify missing trading days (date gaps).

    Since markets are closed on weekends, we allow gaps up to max_gap_days
    (default: 5 days) before flagging as an issue.

    Args:
        df: DataFrame with DatetimeIndex
        max_gap_days: Maximum allowed gap in days (default: 5)

    Returns:
        Dictionary with gap information:
        {
            'gap_count': int,
            'gaps': list of tuples (date1, date2, gap_days)
        }

    Example:
        >>> report = check_date_gaps(df)
        >>> if report['gap_count'] > 0:
        ...     print(f"Found {report['gap_count']} date gaps")
    """
    if not isinstance(df.index, pd.DatetimeIndex):
        logger.warning("Index is not DatetimeIndex, skipping date gap check")
        return {'gap_count': 0, 'gaps': []}

    # Calculate differences between consecutive dates
    date_diffs = df.index.to_series().diff()

    # Find gaps larger than max_gap_days
    large_gaps = date_diffs[date_diffs > timedelta(days=max_gap_days)]

    gaps = []
    for date, gap in large_gaps.items():
        prev_date = df.index[df.index.get_loc(date) - 1]
        gaps.append((prev_date.strftime('%Y-%m-%d'), date.strftime('%Y-%m-%d'), gap.days))

    gap_count = len(gaps)

    if gap_count > 0:
        logger.warning(f"Found {gap_count} date gaps > {max_gap_days} days")
        for prev_date, curr_date, days in gaps[:5]:  # Show first 5
            logger.warning(f"  - Gap between {prev_date} and {curr_date}: {days} days")
    else:
        logger.info(f"✓ No date gaps > {max_gap_days} days")

    return {
        'gap_count': gap_count,
        'gaps': gaps
    }


def check_outliers(df: pd.DataFrame, threshold: float = 5.0) -> Dict:
    """
    Find extreme outliers in numeric columns.

    Outliers are defined as values more than 'threshold' standard deviations
    from the mean.

    Args:
        df: DataFrame to check
        threshold: Number of standard deviations to consider as outlier (default: 5)

    Returns:
        Dictionary with outlier information:
        {
            'total_outliers': int,
            'outliers_by_column': dict
        }

    Example:
        >>> report = check_outliers(df, threshold=5)
        >>> print(f"Total outliers: {report['total_outliers']}")
    """
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    outliers_by_column = {}
    total_outliers = 0

    for col in numeric_cols:
        # Skip columns with all NaN
        if df[col].isna().all():
            continue

        mean = df[col].mean()
        std = df[col].std()

        if std == 0:
            continue

        # Find outliers
        outliers = ((df[col] - mean).abs() > threshold * std).sum()

        if outliers > 0:
            outliers_by_column[col] = int(outliers)
            total_outliers += outliers

    if total_outliers > 0:
        logger.warning(f"Found {total_outliers} outliers (>{threshold} std from mean)")
        for col, count in list(outliers_by_column.items())[:5]:  # Show first 5
            logger.warning(f"  - {col}: {count} outliers")
    else:
        logger.info(f"✓ No outliers found (>{threshold} std)")

    return {
        'total_outliers': total_outliers,
        'outliers_by_column': outliers_by_column
    }


def generate_data_quality_report(df: pd.DataFrame, feature_cols: List[str] = None) -> str:
    """
    Generate a comprehensive data quality report.

    Args:
        df: DataFrame to analyze
        feature_cols: Optional list of feature column names

    Returns:
        Formatted string report

    Example:
        >>> report = generate_data_quality_report(df, feature_cols)
        >>> print(report)
    """
    report_lines = []
    report_lines.append("=" * 60)
    report_lines.append("DATA QUALITY REPORT")
    report_lines.append("=" * 60)

    # Basic info
    report_lines.append("\n📊 DATASET OVERVIEW")
    report_lines.append(f"Rows: {len(df):,}")
    report_lines.append(f"Columns: {len(df.columns)}")

    if isinstance(df.index, pd.DatetimeIndex):
        report_lines.append(f"Date range: {df.index[0].strftime('%Y-%m-%d')} to {df.index[-1].strftime('%Y-%m-%d')}")
        days = (df.index[-1] - df.index[0]).days
        report_lines.append(f"Duration: {days} days")

    # Missing values
    report_lines.append("\n📋 MISSING VALUES")
    missing_report = check_missing_values(df)
    report_lines.append(f"Total missing: {missing_report['total_missing']:,}")
    report_lines.append(f"Percentage: {missing_report['missing_percentage']:.2f}%")

    if missing_report['missing_by_column']:
        report_lines.append("Missing by column:")
        for col, count in list(missing_report['missing_by_column'].items())[:10]:
            pct = count / len(df) * 100
            report_lines.append(f"  - {col}: {count:,} ({pct:.2f}%)")

    # Date gaps
    if isinstance(df.index, pd.DatetimeIndex):
        report_lines.append("\n📅 DATE CONTINUITY")
        gap_report = check_date_gaps(df)
        if gap_report['gap_count'] > 0:
            report_lines.append(f"⚠️  Found {gap_report['gap_count']} date gaps > 5 days")
        else:
            report_lines.append("✓ No significant date gaps")

    # Outliers
    report_lines.append("\n🔍 OUTLIERS (>5 std)")
    outlier_report = check_outliers(df, threshold=5.0)
    if outlier_report['total_outliers'] > 0:
        report_lines.append(f"⚠️  Found {outlier_report['total_outliers']:,} outliers")
        report_lines.append("Outliers by column:")
        for col, count in list(outlier_report['outliers_by_column'].items())[:10]:
            report_lines.append(f"  - {col}: {count:,}")
    else:
        report_lines.append("✓ No extreme outliers detected")

    # Feature statistics (if feature columns provided)
    if feature_cols:
        report_lines.append("\n📈 FEATURE STATISTICS")
        report_lines.append(f"Number of features: {len(feature_cols)}")

        # Check which features exist
        existing_features = [col for col in feature_cols if col in df.columns]
        if existing_features:
            feature_df = df[existing_features]

            report_lines.append(f"\nFeature ranges:")
            for col in existing_features[:10]:  # Show first 10
                if df[col].dtype in [np.float64, np.float32, np.int64, np.int32]:
                    min_val = df[col].min()
                    max_val = df[col].max()
                    mean_val = df[col].mean()
                    report_lines.append(f"  - {col}: [{min_val:.4f}, {max_val:.4f}] (mean: {mean_val:.4f})")

    # Target variable statistics (if exists)
    if 'target' in df.columns:
        report_lines.append("\n🎯 TARGET VARIABLE")
        target = df['target'].dropna()
        report_lines.append(f"Non-null values: {len(target):,}")
        report_lines.append(f"Mean: {target.mean():.6f}")
        report_lines.append(f"Std: {target.std():.6f}")
        report_lines.append(f"Min: {target.min():.6f}")
        report_lines.append(f"Max: {target.max():.6f}")

        # Distribution
        positive_pct = (target > 0).sum() / len(target) * 100
        report_lines.append(f"Positive returns: {positive_pct:.1f}%")

    # Data quality score
    report_lines.append("\n⭐ DATA QUALITY SCORE")
    quality_score = 100

    if missing_report['missing_percentage'] > 5:
        quality_score -= 20
    if gap_report['gap_count'] > 0:
        quality_score -= 10
    if outlier_report['total_outliers'] > 100:
        quality_score -= 10

    report_lines.append(f"Score: {quality_score}/100")

    if quality_score >= 90:
        report_lines.append("Status: ✓ EXCELLENT")
    elif quality_score >= 70:
        report_lines.append("Status: ⚠️  GOOD (minor issues)")
    else:
        report_lines.append("Status: ❌ NEEDS ATTENTION")

    report_lines.append("\n" + "=" * 60)

    report_text = "\n".join(report_lines)
    logger.info("\n" + report_text)

    return report_text


if __name__ == "__main__":
    # Example usage
    from data_collector import DataCollector
    from feature_engineering import FeatureEngineer

    # Load data
    collector = DataCollector('SPY')
    df = collector.download_historical('2020-01-01', '2024-12-31')

    # Create features
    engineer = FeatureEngineer()
    df = engineer.create_basic_features(df)
    df = engineer.create_target(df, horizon=5)

    # Generate report
    report = generate_data_quality_report(df, feature_cols=engineer.get_feature_names())
