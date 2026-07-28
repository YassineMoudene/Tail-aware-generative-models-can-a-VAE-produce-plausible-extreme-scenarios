# Tail-aware generative models: can a VAE produce plausible extreme scenarios?

A controlled experiment on the tail behaviour of deep generative models, and anextreme-value-theory (EVT) corrective.

**Notebook:** [`tail_aware_generative_models.ipynb`](tail_aware_generative_models.ipynb)

## The problem: tail scarcity

Synthetic scenario generation with deep generative models (VAEs, GANs) is an
increasingly proposed answer to a fundamental obstacle in climate-risk and
infrastructure-resilience modelling: **the events that matter most are the ones we
have almost no data for.** But a generator is only useful for crisis planning if the
scenarios it produces are *plausible in the tail*, and standard generative-model
validation (histogram overlap, moment matching) is dominated by the bulk of the
distribution and blind to tail errors.

This repository asks the question directly, in a setting where the ground truth is
known exactly.

## The experiment

1. **Simulate** a heavy-tailed hazard intensity with known tail index α = 3
   (half-Student-t, a regime documented for extreme wind speeds and precipitation).
2. **Train** a standard Gaussian VAE (learned decoder variance, latent dim 2) on
   20 000 samples.
3. **Diagnose** the generated distribution with EVT-native tools: log-log survival
   functions, upper-quantile QQ plots, the Hill estimator, exceedance probabilities.
4. **Correct** the failure with a spliced generator: VAE for the bulk,
   peaks-over-threshold Generalized Pareto Distribution for the tail
   (Pickands–Balkema–de Haan).

## Key result

| Statistic (200k generated samples) | Real data | VAE alone | VAE + GPD hybrid |
|---|---:|---:|---:|
| Hill tail index (true = 3) | 2.82 | 5.31 | **2.99** |
| P(X > q99 of real data) | 0.0100 | 0.0128 | **0.0103** |
| P(X > q99.9 of real data) | 0.0010 | 0.0005 | **0.0011** |
| Largest generated event | 1302 (observed) | 242 | **1425** |

The vanilla VAE fails in two opposite directions at once: it **over-produces
moderate exceedances** (the Gaussian decoder inflates its variance to cover
heavy-tailed data, smearing mass into the near-tail) while **censoring
catastrophes**  across 200 000 samples, ten times the training set, it never
generates an event even a quarter the size of the largest *observed* one. For
resilience planning this is the worst combination: nuisance events are
over-sampled, and the scenario that already happened is treated as impossible.

The failure is structural, not a tuning issue: a Gaussian latent pushed through a
locally-Lipschitz network cannot manufacture polynomial tails (formalised for
normalizing flows by Jaini et al., ICML 2020; the same mechanism constrains VAE
decoders). The corrective is therefore structural too: EVT specifies the exact
asymptotic form of the tail, and the spliced hybrid restores the tail index, the
extreme exceedance probabilities, and the ability to generate
worse-than-observed — yet still plausible — events.

## Why this matters beyond the toy

Nothing in the splice construction is specific to one dimension. In realistic
settings the deep-generative component earns its keep where EVT alone cannot go
covariates, multivariate dependence, and non-stationarity while the EVT component
guarantees asymptotic tail correctness. The two are complementary: **EVT provides
both the validation battery and the corrective for generative scenario models
trained on scarce extreme events.**

**Natural extensions:** conditional generation with tail-spliced validation,
bivariate hazards and tail-dependence (χ) diagnostics, heavy-tailed latent priors
as an internal alternative to external splicing, non-stationary GPD parameters
ξ(t), β(t) connecting the corrective to climate-change attribution.

## Reproducing

```bash
pip install numpy scipy matplotlib torch jupyter
jupyter nbconvert --to notebook --execute tail_aware_generative_models.ipynb
```

Runs in a few minutes on CPU. Seeds are fixed (`torch.manual_seed(0)`,
`np.random.seed(0)`); exact tail digits vary slightly across CPU/threading
configurations, so the committed notebook is the reference for the numbers quoted
above.

## Related work in this portfolio

This experiment complements two applied climate-attribution studies:
non-stationary GEV attribution of the 2003 European heatwave (ERA5, GMST
covariate), and Western Mediterranean precipitation extremes (CHIRPS / ERA5 /
CMIP6, probability-ratio gradient across quantiles). Together they span the two
halves of the problem: *inferring* tail behaviour from scarce observations, and
*generating* synthetic data that respects it.

## References

Coles (2001), *An Introduction to Statistical Modeling of Extreme Values*;
Jaini et al. (2020), *Tails of Lipschitz Triangular Flows*, ICML.
