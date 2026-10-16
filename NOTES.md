# Kalman filter: notes

## 0. The big picture (read this first)

**The problem.** We want x, the true state (e.g. position and velocity of a car). We never see it.
We have two imperfect sources of information:
1. **Physics:** "the car was here, moving this fast, so now it should be there." Wrong by w (cov Q).
2. **Sensor:** "GPS says the car is here." Wrong by v (cov R).

**The loop.** Each time step:
- **Predict:** move the estimate forward with physics. Uncertainty grows (+Q).
- **Update:** a measurement arrives. Combine it with the prediction. Uncertainty shrinks.

**The key fact that makes the KF possible.** If everything is Gaussian and linear, then the
estimate stays Gaussian forever. A Gaussian is fully described by two things: mean and covariance.
So the whole filter only needs to carry **x̂ (mean) and P (covariance)**. Nothing else.

**Two routes to the same K.** There are two ways to derive the update. Both give the same K.
| Route | Question it asks | Section |
|---|---|---|
| Bayes (multiply Gaussians) | "Given both opinions, what is the most probable x?" | §2 |
| MMSE (minimize trace of P⁺) | "Which K makes the error as small as possible on average?" | §4 |

**Why we start in 1D.** In 1D every matrix is a number. Formulas are short and we can check them
with real numbers by hand. Then §3 gives a translation table from 1D to matrices. Every matrix formula
is a 1D formula, translated word by word. Nothing new happens in the matrix version except that
unmeasured states can be corrected (§7).

---

## Dict
The state: what we want to know
┌───────────┬──────┬───────────────────────────────────────────────────────────────────────────────────────────────────────────────┐
│   Name    │ Size │                                                    Meaning                                                    │
├───────────┼──────┼───────────────────────────────────────────────────────────────────────────────────────────────────────────────┤
│ x         │ N×1  │ True state. We never see it.                                                                                  │
├───────────┼──────┼───────────────────────────────────────────────────────────────────────────────────────────────────────────────┤
│ x̂         │ N×1  │ Our estimate of x. In code, this is state().                                                                  │
├───────────┼──────┼───────────────────────────────────────────────────────────────────────────────────────────────────────────────┤
│ e = x − x̂ │ N×1  │ Estimation error. We can't compute it, because x is unknown, but we can reason about it.                      │
├───────────┼──────┼───────────────────────────────────────────────────────────────────────────────────────────────────────────────┤
│ P         │ N×N  │ Covariance of e: how uncertain x̂ is. Diagonal = variances, off-diagonal = correlations. This is covariance(). │
└───────────┴──────┴───────────────────────────────────────────────────────────────────────────────────────────────────────────────┘
P = how much I distrust my current estimate (past doubt + physics doubt, see below).

Predict: the physics model
┌──────┬──────┬─────────────────────────────────────────────────────────────────────────────────────────────────────┐
│ Name │ Size │                                               Meaning                                               │
├──────┼──────┼─────────────────────────────────────────────────────────────────────────────────────────────────────┤
│ F    │ N×N  │ Transition matrix. It moves the state one step forward in time, e.g. position += velocity·dt.       │
├──────┼──────┼─────────────────────────────────────────────────────────────────────────────────────────────────────┤
│ w    │ N×1  │ Process noise. This is what the physics model gets wrong: wind, bumps, acceleration we don't model. │
├──────┼──────┼─────────────────────────────────────────────────────────────────────────────────────────────────────┤
│ Q    │ N×N  │ Covariance of w: how much we distrust the model.                                                    │
└──────┴──────┴─────────────────────────────────────────────────────────────────────────────────────────────────────┘
Q = how much I distrust my physics.

Model: x_next = F x + w.
Predict step: x̂ ← F x̂, P ← F P Fᵀ + Q (§1).

Update: the sensor model
┌──────┬──────┬─────────────────────────────────────────────────────────────────────────────────────────────────────────┐
│ Name │ Size │                                                 Meaning                                                 │
├──────┼──────┼─────────────────────────────────────────────────────────────────────────────────────────────────────────┤
│ z    │ M×1  │ Measurement. The number the sensor gives us.                                                            │
├──────┼──────┼─────────────────────────────────────────────────────────────────────────────────────────────────────────┤
│ H    │ M×N  │ Measurement matrix. It says which parts of the state the sensor sees. In the CV test, it picks x and y. │
├──────┼──────┼─────────────────────────────────────────────────────────────────────────────────────────────────────────┤
│ v    │ M×1  │ Measurement noise.                                                                                      │
├──────┼──────┼─────────────────────────────────────────────────────────────────────────────────────────────────────────┤
│ R    │ M×M  │ Covariance of v: how much we distrust the sensor. It comes from the datasheet or a calibration.         │
└──────┴──────┴─────────────────────────────────────────────────────────────────────────────────────────────────────────┘
R = how much I distrust my sensor.

Model: z = H x + v.

Update: the quantities we compute
┌────────────────┬──────┬────────────────────────────────────────────────────────────────────────────────────────────────────────┐
│      Name      │ Size │                                                Meaning                                                 │
├────────────────┼──────┼────────────────────────────────────────────────────────────────────────────────────────────────────────┤
│ y = z − H x̂    │ M×1  │ Innovation (or residual): the difference between what the sensor says and what we expected it to say.  │
├────────────────┼──────┼────────────────────────────────────────────────────────────────────────────────────────────────────────┤
│ S = H P Hᵀ + R │ M×M  │ Covariance of y: the expected size of the surprise, combining prediction uncertainty and sensor noise. │
├────────────────┼──────┼────────────────────────────────────────────────────────────────────────────────────────────────────────┤
│ K = P Hᵀ S⁻¹   │ N×M  │ Kalman gain. How much of the surprise y to add to x̂.                                                   │
└────────────────┴──────┴────────────────────────────────────────────────────────────────────────────────────────────────────────┘

Update step:
x̂ ← x̂ + K y,
P ← (I − KH) P (I − KH)ᵀ + K R Kᵀ  (Joseph form).

Superscripts
- ⁻ (prior): after predict, before update.
- ⁺ (posterior): after update.

**Where Q goes.** Q does not appear in K directly. It enters P during predict:
```
P⁻ = F P Fᵀ + Q
     └─────┘   └┘
   old doubt,  new doubt from physics
   moved forward
```
So the chain of trust is:
```
Q (physics doubt) ──┐
                    ├─► P⁻ (prediction doubt) ──┐
old P (past doubt) ─┘                           ├─► K (trust slider)
                        R (sensor doubt) ───────┘
```

**Where the numbers come from in practice.** We never estimate P from samples at runtime.
- P₀: a guess (large = "I know nothing at the start").
- R: datasheet or offline calibration.
- Q: a modelling choice, tuned.
- P after that: carried forward by the formulas.

---

## 1. Linear map of a Gaussian (gives the predict step)
If x ~ N(μ, P) and y = F x + w with w ~ N(0, Q) independent, then y ~ N(Fμ, FPFᵀ + Q).

cov(y) = cov(Fx + w) = cov(Fx) + cov(w) = F cov(x) Fᵀ + Q = FPFᵀ + Q
(cov(Fx + w) splits into two parts because w is independent of x.)

Rule to remember: **cov(A e) = A cov(e) Aᵀ.** In 1D: var(a·e) = a²·var(e).
This one rule is used everywhere: predict (A = F), S (A = H), Joseph form (A = I − KH).

cov of a vector: cov(e) = E[(e − μ)(e − μ)ᵀ]. It is a table of all pairwise covariances
cov(eᵢ, eⱼ). The diagonal holds the variances.

---

## 2. Product of two Gaussians (the Bayes route, 1D)

**Why multiply?** Bayes' rule:
```
p(x | z)  ∝  p(z | x) · p(x)
posterior    likelihood  prior
             (sensor)    (prediction)
```
Both factors are opinions about **the same unknown x** (the true state now):
1. Prediction: "x is about μ₁ = x̂⁻, give or take σ₁² = P⁻."
2. Sensor: "x is about μ₂ = z, give or take σ₂² = R."  (1D, H = 1)

Multiplying means: a value of x is plausible only if **both** opinions find it plausible.
The product of two Gaussians is again a Gaussian. This is why the filter can loop forever
with only a mean and a variance.

**Result.** N(μ₁, σ₁²) · N(μ₂, σ₂²) ∝ N(μ, σ²) with
- 1/σ² = 1/σ₁² + 1/σ₂²  (precisions add)
- μ = (σ₂²·μ₁ + σ₁²·μ₂) / (σ₁² + σ₂²)  (precision-weighted average)

Rewrite the mean as "prediction + correction":
- μ = μ₁ + K(μ₂ − μ₁), with **K = σ₁² / (σ₁² + σ₂²)**
- σ² = (1 − K)·σ₁²

Translate to KF names (1D, H = 1):
- x̂⁺ = x̂⁻ + K(z − x̂⁻)
- K = P⁻ / (P⁻ + R)
- P⁺ = (1 − K) P⁻

**K is a trust slider.** It compares uncertainties (variances), never values (means).
- K → 1 when the prediction is vague (P⁻ ≫ R): follow the sensor.
- K → 0 when the sensor is noisy (R ≫ P⁻): keep the prediction.

**Worked example (GPS, step 3).** Old P = 2.19, Q = 1, R = 4, x̂⁻ = 5.55, z = 5.5.
- P⁻ = 2.19 + 1 = 3.19  (Q enters here)
- K = 3.19 / (3.19 + 4) = 0.44. This is below 0.5 because P⁻ < R: the prediction is now more certain than the sensor.
- x̂⁺ = 5.55 + 0.44·(5.5 − 5.55) = 5.53
- P⁺ = (1 − 0.44)·3.19 = 1.79

**In short:** prior × likelihood = posterior. Multiplying = both must agree. Precisions add.
The mean is a precision-weighted average. K = trust in the measurement.

---

## 3. From 1D to matrices: the translation table

The matrix formulas are the 1D formulas, translated with these rules:
| 1D                             | Matrix         | Why |
|-----------                     |----------------|------|
| number a                       | matrix A       |
| a·b = b·a                      | AB ≠ BA        | order matters; keep the order from the derivation |
| a²·P (variance of a·e)         | A P Aᵀ         | rule cov(Ae) = APAᵀ from §1 |
| a·b / c                        | A B C⁻¹        | no division for matrices; multiply by the inverse |
| P + R                          | S = H P Hᵀ + R | prediction doubt, first converted to sensor units by H |
| h = 1 (sensor sees x directly) | H (M×N)        | sensor may see only some states, or in other units |

Apply it to every formula:
| 1D (H = 1)      | 1D, scalar h        | Matrix |
|-----------------|---------------------|--------|
| P⁻ = f²P + Q    | same                | P⁻ = F P Fᵀ + Q |
| y = z − x̂       | y = z − h x̂         | y = z − H x̂ |
| S = P + R       | S = h²P + R         | S = H P Hᵀ + R |
| K = P / (P + R) | K = P h / (h²P + R) | K = P Hᵀ S⁻¹ |
| P⁺ = (1 − K) P  | P⁺ = (1 − K h) P    | P⁺ = (I − K H) P |

Meaning of the H pieces:
- **H P Hᵀ:** my state doubt, expressed in sensor units.
- **P Hᵀ:** "which states are linked to what the sensor sees", and converts the correction back to state units.
- **K H:** go from state to sensor, then back to state.

Check: put N = M = 1 and H = 1 into K = P Hᵀ S⁻¹. Then S = P + R, P Hᵀ = P, S⁻¹ = 1/(P + R),
so K = P / (P + R). Same formula.

---

## 4. Kalman gain (the MMSE route)

**Idea.** Pick K so that the posterior error is as small as possible on average.

**Step 1: error after the update.** With z = Hx + v:
```
e⁺ = x − x̂⁺ = x − x̂ − K(Hx + v − Hx̂) = (I − KH) e − K v
```
e and v are independent, so by cov(Ae) = APAᵀ:
```
P⁺ = (I − KH) P (I − KH)ᵀ + K R Kᵀ        (valid for ANY K)
```

**Step 2: what does "minimize P⁺" mean?**
We minimize tr(P⁺) = sum of the variances = E‖x − x̂⁺‖² (mean squared error).


**Step 3: derivative of a number with respect to a matrix.** ∂tr/∂K is a table the same shape as K.
Entry (i, j) is the ordinary partial derivative ∂(tr)/∂Kᵢⱼ. It is just the gradient.
Two rules, both 1D rules in disguise:

| 1D               | Matrix 
|------------------|------------------------------- 
| d(k·a)/dk = a    | ∂tr(KA)/∂K = Aᵀ 
| d(k²·b)/dk = 2kb | ∂tr(KBKᵀ)/∂K = 2KB  (B symmetric) |

For a term with Kᵀ: transpose it first. tr(Xᵀ) = tr(X).

**Step 4: the derivation, side by side.**

|                | 1D (H = 1)                     | Matrix |
|----------------|------------                    |--------|
| cost           | P⁺ = (1 − K)²P + K²R           | tr(P⁺) = tr(P) − tr(KHP) − tr(PHᵀKᵀ) + tr(KHPHᵀKᵀ) + tr(KRKᵀ) |
| combine        | = P − 2KP + K²(P + R)          | = tr(P) − 2 tr(KHP) + tr(K S Kᵀ) |
| derivative = 0 | −2P + 2K(P + R) = 0            | −2 PHᵀ + 2 K S = 0 |
| solve          | K = P / (P + R)                | K = P Hᵀ S⁻¹ |
| minimum?       | second derivative 2(P + R) > 0 | S is positive definite |

Notes on the "combine" row:
- The two middle terms are equal: PHᵀKᵀ = (KHP)ᵀ because P = Pᵀ, and transposing doesn't change the trace. This gives the factor 2.
- The last two terms share the shape K(…)Kᵀ, so they combine: K(HPHᵀ + R)Kᵀ = K S Kᵀ.

**Short form.** Put the optimal K back into P⁺:
- 1D: P⁺ = P R / (P + R) = (1 − K) P
- Matrix: P⁺ = (I − KH) P

This short form is only true for the optimal K (see §5).

**Why both routes give the same K.** With Gaussian noise, the most probable x (Bayes) and the
x with the smallest mean squared error (MMSE) are the same point. Without Gaussian noise, the
MMSE route still gives the best *linear* estimator.

Interview answer: "K minimizes the expected squared error, tr(P⁺). It's the 1D derivation:
the trace makes the cost a number, P + R becomes S, and division becomes multiplying by S⁻¹."

---

## 5. Joseph form
Why is P⁺ = (I − KH) P (I − KH)ᵀ + K R Kᵀ better numerically than P⁺ = (I − KH) P? When are the two equal?
(Your words. Hints: which formula is valid for any K? Which one is visibly symmetric, as a sum of A P Aᵀ terms?
What happens to a tiny rounding error in K in each?)

---

## 6. Intuition
(Your words. Use K = P⁻ / (P⁻ + R) and P⁻ = P + Q.)
- R → 0 :
- R → ∞ :
- Q too small :
- Q too large :

---

## 7. Unmeasured states are corrected through P

This is the one thing the matrix version adds. The 1D formula can't do it.

Example: state [x, v], the sensor sees only x, so H = [1 0].
```
P = [ Pxx  Pxv ]      P Hᵀ = [ Pxx ]      S = Pxx + R  (a number)
    [ Pxv  Pvv ]             [ Pxv ]

K = P Hᵀ / S = [ Pxx / (Pxx + R) ]   ← the familiar P / (P + R), for position
               [ Pxv / (Pxx + R) ]   ← velocity gets corrected too, through Pxv
```
Nobody measures v. But P knows that errors in x and v are correlated (predict creates this
correlation: x += v·dt). So a surprise in x also corrects v.

Exercise: P = [[4, 2], [2, 3]], R = 1, H = [1 0], x̂⁻ = [0, 0], z = 5. Compute K and x̂⁺.

---

## Interview questions I should be able to answer
- What are the assumptions of the KF? What breaks when they don't hold?
- KF vs EKF vs UKF vs particle filter?
- What is the innovation, and how do you check that the filter is consistent (NIS/NEES)?
- How would you handle measurements arriving out of order, or at different rates?
