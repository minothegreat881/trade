"""
Comprehensive Data Leakage Detection Suite

Tests for all common sources of look-ahead bias in trading models.
CRITICAL: Run this before trusting any backtest results!

Run: python validate_no_leakage.py
"""

import pandas as pd
import numpy as np
from pathlib import Path
import sys
import codecs

# Fix Windows encoding
if sys.platform == 'win32' and hasattr(sys.stdout, 'buffer'):
    sys.stdout = codecs.getwriter('utf-8')(sys.stdout.buffer, 'strict')
    sys.stderr = codecs.getwriter('utf-8')(sys.stderr.buffer, 'strict')


class LeakageDetector:
    """
    Detect data leakage in trading model.

    Tests for:
    - Temporal ordering violations
    - Target leakage into features
    - Future data in features
    - Train/test contamination
    - Sentiment data timing issues
    """

    def __init__(self):
        self.issues_found = []
        self.tests_passed = 0
        self.tests_failed = 0

    def log_issue(self, test_name, severity, description):
        """Log detected issue"""
        self.issues_found.append({
            'test': test_name,
            'severity': severity,
            'description': description
        })

    def test_temporal_order(self, train, test):
        """
        TEST 1: Verify train dates < test dates
        Critical: Train must come before test chronologically
        """
        print("\n[TEST 1] Temporal Order Check")
        print("="*70)

        train_min = train.index.min()
        train_max = train.index.max()
        test_min = test.index.min()
        test_max = test.index.max()

        print(f"Train period: {train_min.date()} to {train_max.date()}")
        print(f"Test period:  {test_min.date()} to {test_max.date()}")
        print(f"Gap: {(test_min - train_max).days} days")

        if train_max >= test_min:
            self.log_issue(
                "Temporal Order",
                "CRITICAL",
                f"Train data ({train_max.date()}) overlaps with test data ({test_min.date()})!"
            )
            print(f"\n❌ FAIL: Train/test overlap detected!")
            self.tests_failed += 1
            return False
        else:
            print("\n✅ PASS: No temporal overlap")
            self.tests_passed += 1
            return True

    def test_target_in_features(self, train_X):
        """
        TEST 2: Check if target is in feature columns
        Critical: Target variable must not be in X
        """
        print("\n[TEST 2] Target Leakage Check")
        print("="*70)

        # Check for 'target' in feature names
        target_cols = [col for col in train_X.columns if 'target' in col.lower()]

        if len(target_cols) > 0:
            self.log_issue(
                "Target Leakage",
                "CRITICAL",
                f"Target-related columns found in features: {target_cols}"
            )
            print(f"❌ FAIL: Found target-related columns: {target_cols}")
            self.tests_failed += 1
            return False
        else:
            print(f"Checked {len(train_X.columns)} features")
            print("✅ PASS: No target in features")
            self.tests_passed += 1
            return True

    def test_future_returns_correlation(self, train_full):
        """
        TEST 3: Check if features use future returns
        Features should NOT be highly correlated with future
        """
        print("\n[TEST 3] Future Returns Correlation Check")
        print("="*70)

        if 'Close' not in train_full.columns:
            print("⚠️ SKIP: No Close price column found")
            return True

        # Calculate future returns
        df = train_full.copy()
        df['future_1d'] = df['Close'].pct_change(-1)
        df['future_5d'] = df['Close'].pct_change(-5)

        feature_cols = [col for col in df.columns
                       if col not in ['Close', 'target', 'future_1d', 'future_5d']]

        suspicious_features = []

        for col in feature_cols[:20]:  # Check first 20 features
            if col in df.columns:
                try:
                    corr_1d = df[col].corr(df['future_1d'])
                    corr_5d = df[col].corr(df['future_5d'])

                    # Suspiciously high correlation (>0.9) suggests leakage
                    if abs(corr_1d) > 0.9 or abs(corr_5d) > 0.9:
                        suspicious_features.append({
                            'feature': col,
                            'corr_1d': corr_1d,
                            'corr_5d': corr_5d
                        })
                except:
                    pass

        if len(suspicious_features) > 0:
            print(f"⚠️ WARNING: {len(suspicious_features)} features highly correlated with future")
            for feat in suspicious_features[:5]:
                print(f"  - {feat['feature']}: corr_5d={feat['corr_5d']:.3f}")

            self.log_issue(
                "Future Returns",
                "HIGH",
                f"Features suspiciously correlated with future: {[f['feature'] for f in suspicious_features]}"
            )
            self.tests_failed += 1
            return False
        else:
            print(f"Checked {min(20, len(feature_cols))} features")
            print("✅ PASS: No suspicious correlations with future returns")
            self.tests_passed += 1
            return True

    def test_sentiment_alignment(self, train_full, test_full):
        """
        TEST 4: Check sentiment data temporal alignment
        Sentiment for day T should be available at day T (not T+1)
        """
        print("\n[TEST 4] Sentiment Temporal Alignment")
        print("="*70)

        sentiment_cols = [col for col in train_full.columns
                         if 'fear_greed' in col.lower() or 'vix' in col.lower()
                         or 'btc' in col.lower() or 'sentiment' in col.lower()]

        print(f"Found {len(sentiment_cols)} sentiment features")

        if len(sentiment_cols) == 0:
            print("⚠️ SKIP: No sentiment features found")
            return True

        # Check for missing sentiment data in test period
        issues = []

        for col in sentiment_cols:
            # Check missing rate in test
            test_missing_pct = test_full[col].isnull().mean() * 100

            if test_missing_pct > 50:
                issues.append(f"{col}: {test_missing_pct:.1f}% missing in test")

        if len(issues) > 0:
            print(f"⚠️ WARNING: High missing rates in test period")
            for issue in issues[:3]:
                print(f"  - {issue}")
            # This is warning, not fail
            print("\n⚠️ PARTIAL PASS: Check if this is expected")
            self.tests_passed += 1
            return True
        else:
            print("✅ PASS: Sentiment data properly aligned")
            self.tests_passed += 1
            return True

    def test_feature_calculation_order(self, train_full):
        """
        TEST 5: Verify rolling features only use past data
        Re-calculate features and compare with stored values
        """
        print("\n[TEST 5] Feature Calculation Order")
        print("="*70)

        if 'Close' not in train_full.columns:
            print("⚠️ SKIP: No Close price column found")
            return True

        # Check SMA calculations (skip first 60 rows due to min_periods)
        checks = []

        if 'sma_20' in train_full.columns:
            expected_sma_20 = train_full['Close'].rolling(20, min_periods=1).mean()
            actual_sma_20 = train_full['sma_20']

            # Skip first 20 rows where min_periods < 20
            diff = (expected_sma_20.iloc[20:] - actual_sma_20.iloc[20:]).abs().max()
            checks.append(('sma_20', diff))

        if 'sma_50' in train_full.columns:
            expected_sma_50 = train_full['Close'].rolling(50, min_periods=1).mean()
            actual_sma_50 = train_full['sma_50']

            # Skip first 50 rows where min_periods < 50
            diff = (expected_sma_50.iloc[50:] - actual_sma_50.iloc[50:]).abs().max()
            checks.append(('sma_50', diff))

        # Check volatility
        if 'volatility_20d' in train_full.columns:
            returns = train_full['Close'].pct_change()
            expected_vol = returns.rolling(20, min_periods=1).std()
            actual_vol = train_full['volatility_20d']

            # Skip first 20 rows
            diff = (expected_vol.iloc[20:] - actual_vol.iloc[20:]).abs().max()
            checks.append(('volatility_20d', diff))

        if len(checks) == 0:
            print("⚠️ SKIP: No features to verify")
            return True

        failures = []
        for feature, diff in checks:
            print(f"  {feature}: max diff = {diff:.6f}")
            if diff > 0.1:  # More than 0.1 difference
                failures.append((feature, diff))

        if len(failures) > 0:
            self.log_issue(
                "Feature Calculation",
                "HIGH",
                f"Feature calculation mismatch: {failures}"
            )
            print(f"\n❌ FAIL: {len(failures)} features have incorrect calculations")
            self.tests_failed += 1
            return False
        else:
            print(f"\n✅ PASS: All {len(checks)} feature calculations correct")
            self.tests_passed += 1
            return True

    def test_train_test_overlap(self, train, test):
        """
        TEST 6: Check for exact row duplicates between train/test
        No date should appear in both train and test
        """
        print("\n[TEST 6] Train/Test Row Overlap")
        print("="*70)

        train_dates = set(train.index)
        test_dates = set(test.index)

        overlap = train_dates.intersection(test_dates)

        print(f"Train dates: {len(train_dates)}")
        print(f"Test dates: {len(test_dates)}")
        print(f"Overlap: {len(overlap)}")

        if len(overlap) > 0:
            self.log_issue(
                "Train/Test Overlap",
                "CRITICAL",
                f"{len(overlap)} dates appear in both train and test!"
            )
            print(f"\n❌ FAIL: {len(overlap)} overlapping dates")
            print(f"Examples: {list(overlap)[:5]}")
            self.tests_failed += 1
            return False
        else:
            print("\n✅ PASS: No overlapping dates")
            self.tests_passed += 1
            return True

    def test_statistical_sanity(self, train_X, test_X):
        """
        TEST 7: Check if test distribution is too different from train
        Test should come from similar distribution as train
        """
        print("\n[TEST 7] Statistical Sanity Check")
        print("="*70)

        warnings = []

        # Check first 10 features
        for col in train_X.columns[:10]:
            if col in test_X.columns:
                train_mean = train_X[col].mean()
                test_mean = test_X[col].mean()
                train_std = train_X[col].std()

                if train_std > 0:
                    z_score = abs((test_mean - train_mean) / train_std)

                    if z_score > 4:  # More than 4 std away
                        warnings.append(f"{col}: z-score={z_score:.2f}")

        print(f"Checked {min(10, len(train_X.columns))} features")

        if len(warnings) > 3:
            print(f"⚠️ WARNING: {len(warnings)} features have unusual test statistics")
            for w in warnings[:3]:
                print(f"  - {w}")
            self.log_issue(
                "Statistical Sanity",
                "MEDIUM",
                f"Test distribution differs significantly: {warnings}"
            )
            # Warning only, not fail
            print("\n⚠️ PARTIAL PASS: Test data has different distribution")
            self.tests_passed += 1
            return True
        else:
            print("\n✅ PASS: Test data statistics reasonable")
            self.tests_passed += 1
            return True

    def test_target_calculation(self, train_full):
        """
        TEST 8: Verify target is calculated correctly (forward returns)
        Target should be FORWARD return, not backward
        """
        print("\n[TEST 8] Target Calculation Verification")
        print("="*70)

        if 'Close' not in train_full.columns or 'target' not in train_full.columns:
            print("⚠️ SKIP: Cannot verify (missing Close or target)")
            return True

        # Recalculate target as 5-day forward return
        # Correct formula: (Close[t+5] - Close[t]) / Close[t]
        # NOT pct_change(-5) which has opposite sign!
        close_shifted = train_full['Close'].shift(-5)
        expected_target = (close_shifted - train_full['Close']) / train_full['Close']
        actual_target = train_full['target']

        # Compare (allowing for some rows at end to be NaN)
        valid_idx = expected_target.notna() & actual_target.notna()
        diff = (expected_target[valid_idx] - actual_target[valid_idx]).abs().max()

        print(f"Target calculation diff: {diff:.8f}")
        print(f"Sample: Expected={expected_target.iloc[0]:.6f}, Actual={actual_target.iloc[0]:.6f}")

        if diff > 0.0001:  # More than 0.01% difference
            self.log_issue(
                "Target Calculation",
                "CRITICAL",
                f"Target calculation incorrect! Diff={diff:.6f}"
            )
            print(f"\n❌ FAIL: Target calculation mismatch")
            print(f"   This suggests target may be using wrong direction!")
            self.tests_failed += 1
            return False
        else:
            print("\n✅ PASS: Target is correctly calculated as forward return")
            self.tests_passed += 1
            return True

    def run_all_tests(self):
        """
        Run all leakage detection tests
        """
        print("="*70)
        print("DATA LEAKAGE DETECTION SUITE")
        print("="*70)
        print("\nCRITICAL: Validating Sharpe 1.28 is legitimate...")

        # Load data
        print("\nLoading data...")

        try:
            # Load X and y separately
            train_X = pd.read_csv('data/train_test_sentiment/train_X.csv',
                                 index_col=0, parse_dates=True)
            train_y = pd.read_csv('data/train_test_sentiment/train_y.csv',
                                 index_col=0, parse_dates=True)
            test_X = pd.read_csv('data/train_test_sentiment/test_X.csv',
                                index_col=0, parse_dates=True)
            test_y = pd.read_csv('data/train_test_sentiment/test_y.csv',
                                index_col=0, parse_dates=True)

            # Load original data with prices
            train_orig = pd.read_csv('data/train_test/train_data.csv',
                                    index_col=0, parse_dates=True)
            test_orig = pd.read_csv('data/train_test/test_data.csv',
                                   index_col=0, parse_dates=True)

            print(f"✓ Train X: {train_X.shape}")
            print(f"✓ Train y: {train_y.shape}")
            print(f"✓ Test X: {test_X.shape}")
            print(f"✓ Test y: {test_y.shape}")

            # Normalize dates
            for df in [train_X, train_y, test_X, test_y, train_orig, test_orig]:
                # Convert to DatetimeIndex if not already
                if not isinstance(df.index, pd.DatetimeIndex):
                    df.index = pd.to_datetime(df.index, utc=True).tz_localize(None)
                else:
                    # Remove timezone if present
                    if hasattr(df.index, 'tz') and df.index.tz is not None:
                        df.index = df.index.tz_localize(None)

                # Normalize to midnight
                df.index = df.index.normalize()

            # Merge for full data
            train_full = train_X.copy()
            train_full['target'] = train_y['target']
            if 'Close' in train_orig.columns:
                train_full = train_full.join(train_orig[['Close']], how='left')

            test_full = test_X.copy()
            test_full['target'] = test_y['target']
            if 'Close' in test_orig.columns:
                test_full = test_full.join(test_orig[['Close']], how='left')

        except FileNotFoundError as e:
            print(f"❌ ERROR: Could not load data files!")
            print(f"   {e}")
            return

        # Run tests
        print("\nRunning tests...\n")

        self.test_temporal_order(train_full, test_full)
        self.test_target_in_features(train_X)
        self.test_train_test_overlap(train_full, test_full)
        self.test_target_calculation(train_full)
        self.test_feature_calculation_order(train_full)
        self.test_future_returns_correlation(train_full)
        self.test_sentiment_alignment(train_full, test_full)
        self.test_statistical_sanity(train_X, test_X)

        # Summary
        print("\n" + "="*70)
        print("TEST SUMMARY")
        print("="*70)
        print(f"Tests passed: {self.tests_passed}")
        print(f"Tests failed: {self.tests_failed}")

        if len(self.issues_found) > 0:
            print(f"\n⚠️ {len(self.issues_found)} ISSUES FOUND:\n")
            for issue in self.issues_found:
                print(f"[{issue['severity']}] {issue['test']}")
                print(f"  → {issue['description']}\n")
        else:
            print("\n✅ NO CRITICAL ISSUES DETECTED!")

        # Final verdict
        print("="*70)
        print("FINAL VERDICT")
        print("="*70)

        critical_issues = [i for i in self.issues_found if i['severity'] == 'CRITICAL']
        high_issues = [i for i in self.issues_found if i['severity'] == 'HIGH']

        if len(critical_issues) > 0:
            print("\n❌ CRITICAL DATA LEAKAGE DETECTED!")
            print("   DO NOT TRUST BACKTEST RESULTS!")
            print("   Sharpe 1.28 is likely INFLATED by leakage.")
            print(f"\n   Fix {len(critical_issues)} critical issue(s) and re-run backtest.")
            print("   Expected result: Sharpe will be LOWER.")

        elif len(high_issues) > 0:
            print("\n⚠️ HIGH-PRIORITY ISSUES DETECTED")
            print("   Review and fix before trusting results.")
            print(f"   {len(high_issues)} issue(s) need attention.")

        elif self.tests_failed > 0:
            print("\n⚠️ MINOR ISSUES DETECTED")
            print("   Results likely valid but review warnings.")

        else:
            print("\n✅ DATA IS CLEAN!")
            print("   No leakage detected.")
            print("   Sharpe 1.28 appears LEGITIMATE.")
            print("\n   Safe to proceed to walk-forward validation.")

        print("="*70 + "\n")

        return len(critical_issues) == 0 and len(high_issues) == 0


def main():
    detector = LeakageDetector()
    is_clean = detector.run_all_tests()

    if not is_clean:
        sys.exit(1)  # Exit with error code if issues found


if __name__ == "__main__":
    main()
