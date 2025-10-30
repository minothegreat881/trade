"""
Validation script for pipeline output
Run this after pipeline.py completes
"""
import pandas as pd
import numpy as np
import os
import sys

# Fix Windows encoding
if sys.platform == 'win32':
    import codecs
    sys.stdout = codecs.getwriter('utf-8')(sys.stdout.buffer, 'strict')
    sys.stderr = codecs.getwriter('utf-8')(sys.stderr.buffer, 'strict')


def validate():
    print("=" * 60)
    print("PIPELINE VALIDATION")
    print("=" * 60)

    # Check if files exist
    if not os.path.exists('data/train_test/train_data.csv'):
        print("\n❌ ERROR: train_data.csv not found!")
        print("   Run pipeline.py first")
        return False

    if not os.path.exists('data/train_test/test_data.csv'):
        print("\n❌ ERROR: test_data.csv not found!")
        print("   Run pipeline.py first")
        return False

    # Load data
    train = pd.read_csv('data/train_test/train_data.csv',
                        index_col=0, parse_dates=True)
    test = pd.read_csv('data/train_test/test_data.csv',
                       index_col=0, parse_dates=True)

    issues = []

    # Test 1: Temporal order
    print("\n[TEST 1] Temporal order...")
    if not train.index.is_monotonic_increasing:
        issues.append("Train data not in temporal order!")
    if not test.index.is_monotonic_increasing:
        issues.append("Test data not in temporal order!")
    if train.index.max() >= test.index.min():
        issues.append("Train/test overlap detected!")
    if len(issues) == 0:
        print("  ✓ PASS")
    else:
        for issue in issues:
            print(f"  ✗ FAIL: {issue}")

    # Test 2: No missing values
    print("\n[TEST 2] Missing values...")
    train_missing = train.isna().sum().sum()
    test_missing = test.isna().sum().sum()
    if train_missing > 0 or test_missing > 0:
        issues.append(
            f"Missing values: train={train_missing}, test={test_missing}")
        print(f"  ✗ FAIL: {issues[-1]}")
    else:
        print("  ✓ PASS")

    # Test 3: Target range
    print("\n[TEST 3] Target range...")
    if 'target' not in train.columns:
        issues.append("Target column not found!")
        print(f"  ✗ FAIL: {issues[-1]}")
    else:
        train_target_min = train['target'].min()
        train_target_max = train['target'].max()
        if train_target_min < -0.5 or train_target_max > 0.5:
            warning = f"Target out of typical range: [{train_target_min:.2f}, {train_target_max:.2f}]"
            print(f"  ⚠ WARNING: {warning}")
            print("    (This might be OK during market crashes)")
        else:
            print(
                f"  ✓ PASS (range: [{train_target_min:.4f}, {train_target_max:.4f}])")

    # Test 4: Required features
    print("\n[TEST 4] Required features...")
    required = ['return_1d', 'return_5d',
                'volatility_20d', 'trend', 'target']
    missing_features = [f for f in required if f not in train.columns]
    if len(missing_features) > 0:
        issues.append(f"Missing features: {missing_features}")
        print(f"  ✗ FAIL: {issues[-1]}")
    else:
        print(f"  ✓ PASS (found {len(train.columns)} features)")

    # Test 5: Data size
    print("\n[TEST 5] Data size...")
    size_issues = []
    if len(train) < 500:
        size_issues.append(f"Train set too small: {len(train)} rows")
    if len(test) < 100:
        size_issues.append(f"Test set too small: {len(test)} rows")

    if len(size_issues) > 0:
        for si in size_issues:
            issues.append(si)
            print(f"  ✗ FAIL: {si}")
    else:
        print(f"  ✓ PASS (train={len(train)}, test={len(test)})")

    # Test 6: No look-ahead bias
    print("\n[TEST 6] Look-ahead bias check...")
    if 'target' in train.columns and 'return_1d' in train.columns:
        # Target should NOT correlate perfectly with current returns
        # If it does, we have look-ahead bias
        corr = train['target'].corr(train['return_1d'])
        if abs(corr) > 0.95:
            issues.append(
                f"Possible look-ahead bias! Target correlates {corr:.4f} with return_1d")
            print(f"  ✗ FAIL: {issues[-1]}")
        else:
            print(f"  ✓ PASS (correlation: {corr:.4f}, expected < 0.95)")

    # Test 7: Feature ranges
    print("\n[TEST 7] Feature ranges...")
    suspicious = []
    for col in train.columns:
        if train[col].dtype in [np.float64, np.int64]:
            if train[col].std() == 0:
                suspicious.append(f"{col} has zero variance")

    if len(suspicious) > 0:
        print(f"  ⚠ WARNING: {len(suspicious)} features with zero variance")
        for s in suspicious[:3]:  # Show first 3
            print(f"    - {s}")
    else:
        print(f"  ✓ PASS (all features have variance)")

    # Summary
    print("\n" + "=" * 60)
    if len(issues) == 0:
        print("🎉 ALL TESTS PASSED!")
        print("✓ Pipeline output is valid")
        print("✓ Ready for modeling")
        print("\n📊 Summary:")
        print(f"  - Train samples: {len(train)}")
        print(f"  - Test samples: {len(test)}")
        print(f"  - Features: {len(train.columns) - 1}")  # -1 for target
        print(f"  - Target mean (train): {train['target'].mean():.6f}")
        print(f"  - Target std (train): {train['target'].std():.6f}")
        return True
    else:
        print("⚠️  VALIDATION FAILED")
        print(f"Found {len(issues)} issue(s):")
        for i, issue in enumerate(issues, 1):
            print(f"  {i}. {issue}")
        print("\n→ Fix these issues before proceeding to modeling")
        return False


if __name__ == "__main__":
    success = validate()
    exit(0 if success else 1)
