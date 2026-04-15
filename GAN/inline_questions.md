# GAN Inline Questions — Answers

---

## Inline Question 4 (1 point)

### Part 1: Analytical evaluation of $\min_x \max_y f(x, y) = xy$

Consider the three cases for $x$:

| Case | $\max_y xy$ | Reasoning |
|------|------------|-----------|
| $x > 0$ | $+\infty$ | Take $y \to +\infty$ |
| $x = 0$ | $0$ | $f = 0$ for all $y$ |
| $x < 0$ | $+\infty$ | Take $y \to -\infty$ |

The outer minimization over $x$ selects the case that gives the smallest maximum:

$$\min_x \max_y \, xy = \boxed{0}, \quad \text{achieved at } x = 0$$

---

### Part 2: Numerical alternating gradient updates

Starting at $(x_0, y_0) = (1, 1)$ with step size $\eta = 1$.

Gradients: $\dfrac{\partial f}{\partial y} = x$, $\quad \dfrac{\partial f}{\partial x} = y$

Update rules per step $i$:
1. $y_i = y_{i-1} + \eta \cdot x_{i-1}$
2. $x_i = x_{i-1} - \eta \cdot y_i$

**Step-by-step computation:**

| Step | $y_i = y_{i-1} + x_{i-1}$ | $x_i = x_{i-1} - y_i$ |
|------|--------------------------|----------------------|
| $i=1$ | $y_1 = 1 + 1 = 2$ | $x_1 = 1 - 2 = -1$ |
| $i=2$ | $y_2 = 2 + (-1) = 1$ | $x_2 = -1 - 1 = -2$ |
| $i=3$ | $y_3 = 1 + (-2) = -1$ | $x_3 = -2 - (-1) = -1$ |
| $i=4$ | $y_4 = -1 + (-1) = -2$ | $x_4 = -1 - (-2) = 1$ |
| $i=5$ | $y_5 = -2 + 1 = -1$ | $x_5 = 1 - (-1) = 2$ |
| $i=6$ | $y_6 = -1 + 2 = 1$ | $x_6 = 2 - 1 = 1$ |

**Summary table of $(x_t, y_t)$ pairs:**

| $t$ | 0 | 1 | 2 | 3 | 4 | 5 | 6 |
|-----|---|---|---|---|---|---|---|
| $y_t$ | 1 | 2 | 1 | -1 | -2 | -1 | 1 |
| $x_t$ | 1 | -1 | -2 | -1 | 1 | 2 | 1 |

The updates cycle back to $(x_6, y_6) = (1, 1) = (x_0, y_0)$, forming a **period-6 orbit** around the saddle point $(0, 0)$.

---

## Inline Question 5 (1 point)

**No**, we will never reach the optimal value using alternating gradient updates.

As shown above, the iterates form a **closed periodic orbit** of period 6 around the saddle point $(x^*, y^*) = (0, 0)$. The updates perpetually cycle through the same 6 points and never converge.

This reveals a fundamental instability of alternating gradient ascent/descent on minimax objectives with saddle points. The saddle point of $f(x,y) = xy$ is a **center** in the dynamical system induced by these updates — not an attractor. Gradient methods converge to minima (or maxima), but saddle points are neither, so the iterates orbit rather than converge. This is precisely why training GANs is difficult in practice: the minimax game has a saddle point as its solution, and naive alternating gradient updates tend to oscillate rather than converge.

---

## Inline Question 6 (1 point)

**No**, this is not a good sign.

If the generator loss steadily decreases while the discriminator loss remains stuck at a high constant value from the very beginning, it indicates the **discriminator has completely failed to learn**. A non-learning discriminator provides no meaningful gradient signal — its output is essentially random — so the generator's apparent "improvement" is illusory: it is not actually learning to produce realistic images, it is merely exploiting an uninformative discriminator.

In a healthy GAN training run, both losses should be dynamic and competitive:
- The discriminator should initially improve (loss decreasing) as it learns to distinguish real from fake.
- The generator should then adapt in response to an improving discriminator.

The scenario described — generator loss down, discriminator loss high and flat — is a symptom of **training imbalance** and often precedes or coincides with **mode collapse**, where the generator maps many noise inputs to the same output(s) that happen to fool the broken discriminator. The generated images in this case would likely be low-quality or degenerate.
