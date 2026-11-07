# Linear Regression — cheat sheet

> Models a continuous target as a linear combination of features:
> **ŷ = β₀ + β₁x₁ + … + βₚxₚ**, fit by minimizing squared error.

## 1. Intuition
Draw the straight line (or hyperplane) through the cloud of points that is, on
average, closest to all of them. "Closest" = smallest sum of squared vertical
distances (residuals).

## 2. When to use / when NOT
| Use when | Avoid when |
|---|---|
| Target is continuous, relationship ~linear | Target is categorical (use logistic) |
| You want an interpretable, fast baseline | Strong non-linearities / interactions (use trees/GBM) |
| Few, low-collinearity features | Heavy multicollinearity, many outliers |

## 3. Key formulas
- Model: `ŷ = Xβ`
- Loss (MSE / OLS): `J(β) = (1/n) Σ (yᵢ − ŷᵢ)²`
- Closed-form (Normal Equation): `β = (XᵀX)⁻¹ Xᵀy`
- Gradient (for gradient descent): `∇J = −(2/n) Xᵀ(y − Xβ)`

## 4. Assumptions (the 5 — classic interview question)
1. **Linearity** — relationship between X and Y is linear.
2. **Independence** of errors (no autocorrelation).
3. **Homoscedasticity** — constant variance of residuals across X.
4. **Normality of residuals** (for valid inference / p-values, CIs).
5. **No (perfect) multicollinearity** — check **VIF** (>5–10 is a problem).
(Bonus: no severe outliers / high-leverage points.)

## 5. How it's trained
- **Normal Equation** (exact, O(p³) — fine for small p) or
- **Gradient Descent** (scales to large data / many features).
- OLS is also the **MLE** solution under Gaussian errors.

## 6. Metrics
| Metric | Meaning |
|---|---|
| **R²** | fraction of variance explained = `1 − SS_res/SS_tot`. Never decreases when adding features. |
| **Adjusted R²** | `1 − (1−R²)(n−1)/(n−p−1)` — penalizes useless features; use to compare models with different p. |
| **RMSE / MAE** | average error in target units (RMSE punishes big errors more). |

## 7. Regularized variants
- **Ridge (L2):** adds `λΣβ²` → shrinks coefficients, handles multicollinearity.
- **Lasso (L1):** adds `λΣ|β|` → drives some coefficients to 0 (feature selection).
- **ElasticNet:** mix of L1 + L2.

## 8. Pros / cons
| Pros | Cons |
|---|---|
| Simple, fast, interpretable coefficients | Assumes linearity |
| Closed-form solution exists | Sensitive to outliers & multicollinearity |
| Great baseline | Underfits complex data |

## 9. Gotchas / traps
- **R² always ↑ with more features** → use adjusted R² to compare.
- **Correlation ≠ causation**; coefficients are associational.
- Scale features before gradient descent / regularization.
- A high R² can still hide bad residual patterns → always plot residuals.

## 10. Rapid-fire Q&A
- **Q: R² vs Adjusted R²?** R² = variance explained but never decreases with more
  features; adjusted R² penalizes added predictors, so it can decrease — use it to
  compare models of different size.
- **Q: Assumptions?** Linearity, independent errors, homoscedasticity, normal
  residuals, no multicollinearity.
- **Q: Why squared error?** Differentiable, convex (single global min), and equals
  MLE under Gaussian noise.
- **Q: Normal equation vs gradient descent?** Normal eq is exact but O(p³) and
  needs `XᵀX` invertible; GD scales better and works when p is huge.
- **Q: How to detect multicollinearity?** VIF; also unstable/large-variance coeffs.

## 11. Next
Regularization → Logistic Regression → Generalized Linear Models.
See `linear_regression.ipynb` for the from-scratch derivation + sklearn on real data.
