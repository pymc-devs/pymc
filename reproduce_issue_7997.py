#!/usr/bin/env python3
"""
Reproduction script for issue #7997: CategoricalGibbsMetropolis doesn't respect the tune parameter
https://github.com/pymc-devs/pymc/issues/7997
"""

import pymc as pm
import numpy as np

print("Testing CategoricalGibbsMetropolis tune parameter behavior...")
print("=" * 70)

# Original reproduction code from the issue
with pm.Model():
    pm.Categorical("cat", [0.02, 0.08, 0.9])
    idata = pm.sample(draws=20, tune=20, random_seed=42, discard_tuned_samples=False)

print(f"\nRequested: tune=20, draws=20")
print(f"Expected output: 20 tuning samples + 20 posterior samples")

# Check the actual number of samples
n_warmup = len(idata.warmup_posterior.draw) if hasattr(idata, 'warmup_posterior') else 0
n_posterior = len(idata.posterior.draw)

print(f"\nActual results:")
print(f"  Warmup samples: {n_warmup}")
print(f"  Posterior samples: {n_posterior}")

# Check the tune stats
if hasattr(idata, 'warmup_sample_stats'):
    warmup_tune_stats = idata.warmup_sample_stats["tune"].values
    print(f"\n  Warmup 'tune' stats: {np.unique(warmup_tune_stats)} (should be [True])")
    
if hasattr(idata, 'sample_stats'):
    posterior_tune_stats = idata.sample_stats["tune"].values
    print(f"  Posterior 'tune' stats: {np.unique(posterior_tune_stats)} (should be [False])")

# Verify the fix
if n_warmup == 20 and n_posterior == 20:
    print("\n✅ SUCCESS: The tune parameter is now respected!")
    print("   Tuning samples and posterior samples are correctly separated.")
else:
    print(f"\n❌ FAILURE: Expected 20 warmup + 20 posterior, got {n_warmup} + {n_posterior}")

# Also test with BinaryGibbsMetropolis for comparison
print("\n" + "=" * 70)
print("Comparing with BinaryGibbsMetropolis (should work correctly)...")
print("=" * 70)

with pm.Model():
    pm.Bernoulli("binary", p=0.3)
    idata_binary = pm.sample(draws=20, tune=20, random_seed=42, discard_tuned_samples=False)

n_warmup_binary = len(idata_binary.warmup_posterior.draw) if hasattr(idata_binary, 'warmup_posterior') else 0
n_posterior_binary = len(idata_binary.posterior.draw)

print(f"\nBinaryGibbsMetropolis results:")
print(f"  Warmup samples: {n_warmup_binary}")
print(f"  Posterior samples: {n_posterior_binary}")

if n_warmup_binary == 20 and n_posterior_binary == 20:
    print("\n✅ BinaryGibbsMetropolis works correctly (as expected)")
else:
    print(f"\n⚠️  BinaryGibbsMetropolis also has issues: {n_warmup_binary} + {n_posterior_binary}")
