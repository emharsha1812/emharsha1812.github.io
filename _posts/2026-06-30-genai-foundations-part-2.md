---
layout: post
title: "Generative AI Foundations — Part 2: Variational Divergence Minimization and GANs"
date: 2026-06-30
series: genai-foundations
series_part: 2
series_title: "Generative AI Foundations"
series_description: "A ground-up course covering the math and modern generative AI."
description: "f-divergences, the Fenchel conjugate, and the classical GAN derived from first principles."
tags: generative-ai, mathematics
categories: courses
---

> **A note on attribution.** These notes are directly inspired by [Prof. Prathosh A P's course *Mathematical Foundations of Generative AI* (BSDA5002)](https://study.iitm.ac.in/ds/course_pages/BSDA5002.html), offered through IIT Madras. Prof. Prathosh is an Assistant Professor in the Division of EECS at IISc Bangalore, and the course is one of the clearest treatments of deep generative models I have come across — covering probabilistic foundations, VAEs, GANs, diffusion models, and LLMs from the ground up.
>
> Much of the structure and topic progression here follows his course. Where I have departed from it is intentional: I have rewritten explanations with more mathematical detail and intermediate steps, filling in gaps that I found myself having to work through on paper. The goal is to make the ideas easier to grasp for someone encountering them for the first time, without sacrificing rigour. All credit for the original framing and pedagogy belongs to Prof. Prathosh — any errors in this retelling are mine alone.

# Chapter 2: Variational Divergence Minimization and Generative Adversarial Networks

## Where We Left Off

Part 1 reduced generative modeling to a single optimization problem:

$$
\theta^\ast = \arg\min_{\theta} D(P_X \Vert P_\theta).
$$

The data consists of $\mathrm{i.i.d.}$ samples from an unknown distribution $P_X$ on $\mathbb{R}^d$. The model distribution $P_\theta$ is defined implicitly through a generator network $g_\theta$, which transforms noise $z \sim \mathcal{N}(0, I)$ into samples in data space. The divergence $D$ measures how far apart $P_\theta$ is from $P_X$, and the training procedure adjusts $\theta$ to minimize this divergence.

This formulation raised four questions:

1. How is $D(P_X \Vert P_\theta)$ computed when neither density is available in closed form?
2. Which divergence is the right one to use?
3. How is the generator $g_\theta$ parameterized?
4. How is the resulting optimization problem solved in practice?

The present chapter develops the answer to the first two questions and provides the structural answers for the third and fourth. The method is called **Variational Divergence Minimization** ($\mathrm{VDM}$), and its most famous instance is the **Generative Adversarial Network** ($\mathrm{GAN}$).

> [!important] Structure of the chapter
> The first half introduces the family of $f$-divergences, identifies the obstacle to computing them from samples, and resolves the obstacle using the Fenchel conjugate. This produces a saddle-point optimization problem that is solvable with neural networks. The second half specializes this construction to one specific divergence and recovers the classical GAN as a corollary.

---

# Part I. The Family of $f$-Divergences

## Definition

Let $P_X$ and $P_\theta$ be probability distributions on $\mathcal{X} \subseteq \mathbb{R}^d$ with densities $p_X(x)$ and $p_\theta(x)$. Let

$$
f : \mathbb{R}_+ \to \mathbb{R}
$$

be a convex, lower semi-continuous function with $f(1) = 0$. The **$f$-divergence** between $P_X$ and $P_\theta$ is defined as

$$
D_f(P_X \Vert P_\theta) = \int_{\mathcal{X}} p_\theta(x) \, f\!\left( \frac{p_X(x)}{p_\theta(x)} \right) dx.
$$

The three conditions on $f$ play distinct roles. Convexity ensures non-negativity of $D_f$ through Jensen's inequality. The condition $f(1) = 0$ ensures that $D_f$ vanishes when the two distributions coincide. Lower semi-continuity is a technical condition needed for the Fenchel duality machinery introduced later in the chapter.

The quantity $p_X(x) / p_\theta(x)$ is the likelihood ratio at $x$. The integrand evaluates $f$ at this ratio and weights the result by $p_\theta(x)$. Integration produces an averaged measure of how far the ratio deviates from $1$.

## Properties

The $f$-divergence has two properties that justify its use as a measure of dissimilarity.

**Non-negativity.** For every choice of admissible $f$,

$$
D_f(P_X \Vert P_\theta) \geq 0.
$$

This follows from Jensen's inequality applied to the convex function $f$. The likelihood ratio is a non-negative random variable whose expectation under $P_\theta$ equals $1$:

$$
\mathbb{E}_{x \sim P_\theta}\!\left[\frac{p_X(x)}{p_\theta(x)}\right] = \int p_\theta(x) \cdot \frac{p_X(x)}{p_\theta(x)} \, dx = \int p_X(x) \, dx = 1.
$$

By Jensen's inequality,

$$
\mathbb{E}_{x \sim P_\theta}\!\left[f\!\left( \frac{p_X(x)}{p_\theta(x)} \right)\right] \geq f\!\left(\mathbb{E}_{x \sim P_\theta}\!\left[\frac{p_X(x)}{p_\theta(x)}\right]\right) = f(1) = 0.
$$

**Identity of indiscernibles.** Under the additional assumption that $f$ is strictly convex at $1$,

$$
D_f(P_X \Vert P_\theta) = 0 \quad \text{if and only if} \quad P_X = P_\theta \text{ almost everywhere.}
$$

These two properties make $D_f$ a divergence. The $f$-divergence is not, in general, a distance: it is typically asymmetric, and it does not satisfy the triangle inequality.

> [!note] Divergence and distance
> A distance is required to be symmetric and to satisfy the triangle inequality. A divergence is required only to be non-negative and to vanish exactly on equal distributions. Most $f$-divergences are not distances, and the asymmetry $D_f(P_X \Vert P_\theta) \neq D_f(P_\theta \Vert P_X)$ has practical consequences for the resulting generative models.

## Convex Functions

The definition of $f$-divergence depends on convexity of $f$. A function $f : \mathbb{R} \to \mathbb{R}$ is **convex** if, for any two points $u_1, u_2$ in its domain and any non-negative weights $\alpha_1, \alpha_2$ with $\alpha_1 + \alpha_2 = 1$,

$$
\alpha_1 f(u_1) + \alpha_2 f(u_2) \geq f(\alpha_1 u_1 + \alpha_2 u_2).
$$

Geometrically, the chord connecting any two points on the graph of $f$ lies on or above the graph. Equivalently, the region above the graph (the epigraph) is a convex set.

If $f$ is twice differentiable, a simpler characterization applies: $f$ is convex on an interval if and only if $f''(u) \geq 0$ everywhere on that interval.

> [!example] Convexity check for $f(u) = u \log u$
> The function $f(u) = u \log u$ is the generator of the forward KL divergence. Its first and second derivatives are
> 
> $$
> f'(u) = \log u + 1, \qquad f''(u) = \frac{1}{u}.
> $$
> 
> For $u > 0$, the second derivative is positive. The function is therefore strictly convex on $\mathbb{R}_+$. The value at $u = 1$ is $f(1) = 1 \cdot \log 1 = 0$, satisfying the second condition for an $f$-divergence generator.

## Common Choices of $f$

The flexibility of the $f$-divergence framework comes from the fact that several standard divergences are obtained as special cases by choosing different generator functions.

### Forward Kullback-Leibler Divergence

Setting $f(u) = u \log u$ gives

$$
D_{\mathrm{KL}}(P_X \Vert P_\theta) = \int p_\theta(x) \cdot \frac{p_X(x)}{p_\theta(x)} \log \frac{p_X(x)}{p_\theta(x)} \, dx = \int p_X(x) \log \frac{p_X(x)}{p_\theta(x)} \, dx.
$$

This is the standard Kullback-Leibler divergence. It is the divergence implicitly minimized by maximum likelihood estimation. Forward KL places strong penalties on regions where $P_X$ has mass but $P_\theta$ does not, because the integrand becomes large when $p_X$ is large and $p_\theta$ is small.

> [!example] Numerical computation of forward KL
> Let $P_X$ be the Bernoulli distribution with parameter $p = 0.7$ and $P_\theta$ be Bernoulli with parameter $q = 0.5$.
> 
> $$
> D_{\mathrm{KL}}(P_X \Vert P_\theta) = 0.7 \log \frac{0.7}{0.5} + 0.3 \log \frac{0.3}{0.5} \approx 0.082.
> $$
> 
> Swapping the arguments gives
> 
> $$
> D_{\mathrm{KL}}(P_\theta \Vert P_X) = 0.5 \log \frac{0.5}{0.7} + 0.5 \log \frac{0.5}{0.3} \approx 0.087.
> $$
> 
> The two values differ, which demonstrates that KL is asymmetric.

### Reverse Kullback-Leibler Divergence

Setting $f(u) = -\log u$ gives the reverse Kullback-Leibler divergence:

$$
D_{\mathrm{KL}}(P_\theta \Vert P_X) = \int p_\theta(x) \log \frac{p_\theta(x)}{p_X(x)} \, dx.
$$

The integrand is large when $P_\theta$ places mass on regions where $P_X$ has little mass. Reverse KL therefore penalizes the model for placing probability where the data has none.

The two KL divergences induce different optimization behavior. Consider a model family that is too restrictive to fit a multimodal $P_X$ — for instance, a single Gaussian fitted to a mixture of two well-separated Gaussians. Forward KL forces the model to place mass on all modes of $P_X$, even at the cost of placing significant mass in regions where $P_X$ has none. The resulting model is broad and covers all modes. Reverse KL forces the model to concentrate on regions where $P_X$ has high mass, even at the cost of ignoring some modes entirely. The resulting model is narrow and may concentrate on a single mode. These two behaviors are referred to as mode-covering and mode-seeking, respectively.

### Jensen-Shannon Divergence

The function

$$
f(u) = \tfrac{1}{2}\left(u \log u - (u + 1) \log \tfrac{u + 1}{2}\right)
$$

generates the Jensen-Shannon divergence:

$$
D_{\mathrm{JS}}(P_X \Vert P_\theta) = \tfrac{1}{2} D_{\mathrm{KL}}(P_X \Vert M) + \tfrac{1}{2} D_{\mathrm{KL}}(P_\theta \Vert M), \qquad M = \tfrac{1}{2}(P_X + P_\theta).
$$

The Jensen-Shannon divergence is symmetric: $D_{\mathrm{JS}}(P_X \Vert P_\theta) = D_{\mathrm{JS}}(P_\theta \Vert P_X)$. It is also bounded above by $\log 2$. These properties make it numerically more tractable than KL. The classical GAN, as derived later in this chapter, minimizes a quantity closely related to the Jensen-Shannon divergence.

### Total Variation Distance

The function $f(u) = \tfrac{1}{2}|u - 1|$ generates the total variation distance:

$$
D_{\mathrm{TV}}(P_X, P_\theta) = \tfrac{1}{2} \int |p_X(x) - p_\theta(x)| \, dx.
$$

The total variation distance is a true metric: symmetric and satisfying the triangle inequality. It also has an operational interpretation as the maximum difference in probability that $P_X$ and $P_\theta$ assign to any measurable event.

> [!note] Different divergences induce different geometries
> Each choice of $f$ defines a different geometry on the space of probability distributions. The gradients available for optimization, the sensitivity to outliers, and the behavior when the two distributions have disjoint support all depend on $f$. The single optimization recipe developed in the rest of this chapter applies to all such choices, so the divergence becomes a hyperparameter of the method.

---

# Part II. The Sample-Based Computation Problem

## The Generator Setup

The training data consists of

$$
\mathcal{D} = \{x_1, x_2, \ldots, x_n\}, \qquad x_i \overset{\mathrm{i.i.d.}}{\sim} P_X, \qquad x_i \in \mathbb{R}^d.
$$

The generator network defines an implicit distribution by

$$
z \sim \mathcal{N}(0, I), \qquad \hat{x} = g_\theta(z), \qquad \hat{x} \sim P_\theta.
$$

Training proceeds by minimizing an $f$-divergence between $P_\theta$ and $P_X$:

$$
\theta^\ast = \arg\min_\theta D_f(P_X \Vert P_\theta).
$$

If $\theta^\ast$ makes $P_\theta = P_X$, then the generator becomes a sampler for the true data distribution.

## The Obstacle

Direct computation of $D_f$ requires evaluating the integral

$$
D_f(P_X \Vert P_\theta) = \int p_\theta(x) f\!\left( \frac{p_X(x)}{p_\theta(x)} \right) dx.
$$

Two terms in this expression are inaccessible. The density $p_X(x)$ is unknown: only samples from $P_X$ are available. The density $p_\theta(x)$ is implicitly defined by the generator network and admits no closed form for general $g_\theta$. Samples from $P_\theta$ can be produced cheaply by running the generator, but pointwise evaluation of $p_\theta$ is not possible.

Direct computation of $D_f$ is therefore infeasible. The next two sections develop the tools required to estimate $D_f$ using only samples from $P_X$ and $P_\theta$.

## Estimating Integrals from Samples

The standard tool for converting integrals into computable quantities is the Law of Large Numbers. Let $h : \mathbb{R}^d \to \mathbb{R}$ be a function and consider the integral

$$
I = \int h(x) p_X(x) \, dx = \mathbb{E}_{x \sim P_X}[h(x)].
$$

If $x_1, \ldots, x_n$ are independent samples from $P_X$, the Law of Large Numbers states that

$$
\frac{1}{n} \sum_{i=1}^n h(x_i) \xrightarrow{n \to \infty} \mathbb{E}_{x \sim P_X}[h(x)].
$$

The integral can therefore be estimated by an empirical average over samples, without any reference to the density $p_X(x)$. The same construction applies to expectations under $P_\theta$, using samples generated by the network.

> [!example] Estimating an expectation from samples
> Let $X$ be a real-valued random variable with unknown distribution. Suppose $\mathbb{E}[X^2]$ is required. Drawing $n = 1000$ samples $x_1, \ldots, x_{1000}$ and computing
> 
> $$
> \frac{1}{1000} \sum_{i=1}^{1000} x_i^2
> $$
> 
> yields an estimate of $\mathbb{E}[X^2]$. No information about the density is used, and accuracy of the estimate is controlled by the sample size.

The path forward is therefore to rewrite $D_f$ in a form that depends on $P_X$ and $P_\theta$ only through expectations. The tool that achieves this rewriting is the Fenchel conjugate of $f$.

---

# Part III. The Fenchel Conjugate

## Definition

Let $f : \mathbb{R} \to \mathbb{R}$ be a convex function. The **Fenchel conjugate** of $f$, denoted $f^\ast$, is defined by

$$
f^\ast(t) = \sup_{u \in \mathrm{dom}\, f} \{tu - f(u)\}.
$$

The conjugate is itself a function of a new variable $t$. Its domain $\mathrm{dom}\, f^\ast$ is the set of $t$ for which the supremum is finite.

The expression $tu - f(u)$ is, for fixed $t$, an affine function of $u$ minus $f(u)$. The supremum identifies the largest gap between a linear function of slope $t$ and the graph of $f$. The conjugate $f^\ast(t)$ records this gap as a function of the slope.

## Properties

Two properties of $f^\ast$ are needed for the derivation that follows.

**Convexity of the conjugate.** The function $f^\ast$ is convex on its domain. This holds because $f^\ast$ is the supremum of affine functions of $t$ (one for each value of $u$), and a supremum of affine functions is always convex.

**Biconjugation.** If $f$ is convex and lower semi-continuous, then applying the conjugate operation twice recovers $f$:

$$
f^{**} = f.
$$

Written out, this means

$$
f(u) = \sup_{t \in \mathrm{dom}\, f^\ast} \{tu - f^\ast(t)\}.
$$

This identity is the technical statement of the Fenchel-Moreau theorem. Its importance for the present construction is that it expresses $f(u)$ as a supremum, over a parameter $t$, of an expression that is affine in $u$. Affine expressions interact cleanly with integrals and expectations, while the original nonlinear function $f$ does not.

> [!example] Conjugate of $f(u) = \tfrac{1}{2} u^2$
> Setting up the supremum,
> 
> $$
> f^\ast(t) = \sup_u \{tu - \tfrac{1}{2} u^2\}.
> $$
> 
> The derivative of the bracket with respect to $u$ is $t - u$, which vanishes at $u = t$. Substituting,
> 
> $$
> f^\ast(t) = t \cdot t - \tfrac{1}{2} t^2 = \tfrac{1}{2} t^2.
> $$
> 
> The function $\tfrac{1}{2} u^2$ is therefore self-conjugate, in the sense that $f^\ast = f$.

> [!example] Conjugate of $f(u) = u \log u$
> Setting up the supremum on $u > 0$,
> 
> $$
> f^\ast(t) = \sup_{u > 0} \{tu - u \log u\}.
> $$
> 
> The derivative is $t - \log u - 1$, which vanishes at $u = e^{t - 1}$. Substituting,
> 
> $$
> f^\ast(t) = t \cdot e^{t - 1} - e^{t - 1}(t - 1) = e^{t - 1}.
> $$
> 
> The conjugate of $u \log u$ is therefore the shifted exponential $e^{t - 1}$, with domain $\mathbb{R}$. This conjugate appears in the VDM realization of forward KL divergence.

The biconjugation identity is the form that will be substituted into the $f$-divergence integral. The result of this substitution is the variational lower bound developed in the next section.

---

# Part IV. The Variational Representation of $D_f$

## Substituting the Conjugate Identity

The biconjugation identity gives, for every $u$,

$$
f(u) = \sup_{t \in \mathrm{dom}\, f^\ast} \{tu - f^\ast(t)\}.
$$

Substituting $u = p_X(x) / p_\theta(x)$ into the $f$-divergence integral produces

$$
D_f(P_X \Vert P_\theta) = \int p_\theta(x) \sup_{t \in \mathrm{dom}\, f^\ast}\!\left\{ t \cdot \frac{p_X(x)}{p_\theta(x)} - f^\ast(t) \right\} dx.
$$

The supremum inside the integral is taken pointwise: for each $x$, an optimal $t$ is selected. Because the optimal $t$ depends on $x$, it can be viewed as the value, at $x$, of a function $T(x)$. The function $T$ maps from data space into the domain of $f^\ast$:

$$
T : \mathcal{X} \to \mathrm{dom}\, f^\ast.
$$

The pointwise supremum can then be rewritten as a supremum over all such functions:

$$
D_f(P_X \Vert P_\theta) = \sup_{T : \mathcal{X} \to \mathrm{dom}\, f^\ast} \int p_\theta(x) \!\left\{ T(x) \cdot \frac{p_X(x)}{p_\theta(x)} - f^\ast(T(x)) \right\} dx.
$$

The factor of $p_\theta(x) \cdot \frac{p_X(x)}{p_\theta(x)} = p_X(x)$ simplifies the first term, giving

$$
D_f(P_X \Vert P_\theta) = \sup_{T} \!\left\{ \int p_X(x) T(x) \, dx - \int p_\theta(x) f^\ast(T(x)) \, dx \right\}.
$$

Both integrals are now expectations:

$$
\boxed{ \; D_f(P_X \Vert P_\theta) = \sup_{T : \mathcal{X} \to \mathrm{dom}\, f^\ast} \Big\{ \mathbb{E}_{x \sim P_X}[T(x)] - \mathbb{E}_{x \sim P_\theta}[f^\ast(T(x))] \Big\}. \;}
$$

This identity is the **variational representation** of $D_f$. The divergence has been rewritten as a supremum of a difference of expectations. The densities $p_X$ and $p_\theta$ no longer appear explicitly. The expectations can be estimated by sample averages.

The supremum is now over an auxiliary function $T$. The optimal $T^\ast$ is characterized by the pointwise first-order condition obtained from the original supremum, and depends on the ratio $p_X(x) / p_\theta(x)$.

## Restricting to a Function Class

The supremum in the variational representation ranges over all measurable functions $T : \mathcal{X} \to \mathrm{dom}\, f^\ast$. In practice, the optimization is conducted over a restricted class $\mathcal{T}$ of functions, typically those representable by a neural network with parameters $w$:

$$
\mathcal{T} = \{T_w : w \in W\}.
$$

If $\mathcal{T}$ does not contain the optimal $T^\ast$, the supremum over $\mathcal{T}$ is strictly smaller than the supremum over all functions:

$$
\sup_{T \in \mathcal{T}} \Big\{ \mathbb{E}_{P_X}[T(x)] - \mathbb{E}_{P_\theta}[f^\ast(T(x))] \Big\} \;\leq\; D_f(P_X \Vert P_\theta).
$$

The right-hand side of the variational representation, when restricted to a function class $\mathcal{T}$, is therefore a **lower bound** on the true divergence. The bound becomes tight as $\mathcal{T}$ grows toward the full space of functions.

This lower bound is the central object of variational divergence minimization. In the next section, $\mathcal{T}$ is taken to be the function class realized by a neural network, and the lower bound becomes the basis for an adversarial training procedure.

> [!note] Significance of the bound being a lower bound
> The optimization problem treats $T$ as an inner maximization and $\theta$ as an outer minimization. The inner maximization tightens the lower bound on $D_f$. The outer minimization drives this tightened lower bound — and therefore $D_f$ itself — toward zero. The min-max structure aligns the directions of optimization in a way that is consistent with minimizing the true divergence, despite the bound being one-sided.

---

# Part V. Variational Divergence Minimization

## The Saddle-Point Objective

The generator network defines $P_\theta$ implicitly through $z \sim \mathcal{N}(0, I)$ and $\hat{x} = g_\theta(z)$. Expectations under $P_\theta$ can therefore be rewritten as expectations over $z$:

$$
\mathbb{E}_{x \sim P_\theta}[\phi(x)] = \mathbb{E}_{z \sim \mathcal{N}(0, I)}[\phi(g_\theta(z))]
$$

for any measurable function $\phi$. Substituting this into the variational representation and parameterizing $T$ by a neural network $T_w$ yields the **VDM objective**:

$$
J(\theta, w) = \mathbb{E}_{x \sim P_X}[T_w(x)] - \mathbb{E}_{z \sim \mathcal{N}(0, I)}[f^\ast(T_w(g_\theta(z)))].
$$

The training problem is a saddle-point problem:

$$
\theta^\ast, w^\ast = \arg\min_\theta \max_w J(\theta, w).
$$

The two networks have opposing roles. The network $T_w$ maximizes $J$, which corresponds to identifying the function that best exposes the discrepancy between $P_X$ and $P_\theta$. The generator $g_\theta$ minimizes $J$, which corresponds to producing samples that reduce the apparent discrepancy. The min-max structure is the defining feature of adversarial training.

## Parameterization of $T_w$

The function $T_w$ must take values in $\mathrm{dom}\, f^\ast$. For many choices of $f$, this domain is a proper subset of $\mathbb{R}$, such as a half-line. A neural network with a linear output layer produces outputs in all of $\mathbb{R}$. An activation function is therefore applied to the network output to constrain its range.

The standard parameterization is

$$
T_w(x) = \sigma_f(V_w(x)),
$$

where $V_w : \mathcal{X} \to \mathbb{R}$ is a neural network with parameters $w$ and $\sigma_f : \mathbb{R} \to \mathrm{dom}\, f^\ast$ is an activation function determined by the choice of $f$. The composition $\sigma_f \circ V_w$ produces values in the appropriate range.

The training of a VDM model therefore involves two neural networks: a generator $g_\theta$ that maps noise to data, and a witness network $V_w$ whose output (after the activation $\sigma_f$) serves as $T_w$.

## Training Algorithm at a High Level

The saddle-point objective is optimized by alternating gradient steps. One iteration consists of:

A minibatch $\{x_1, \ldots, x_B\}$ is drawn from $\mathcal{D}$, and a minibatch of noise vectors $\{z_1, \ldots, z_B\}$ is drawn from $\mathcal{N}(0, I)$. The empirical version of the objective is computed by replacing expectations with sample averages:

$$
\hat{J}(\theta, w) = \frac{1}{B} \sum_{i=1}^B T_w(x_i) - \frac{1}{B} \sum_{i=1}^B f^\ast(T_w(g_\theta(z_i))).
$$

The parameters $w$ are updated by a gradient ascent step on $\hat{J}$. The parameters $\theta$ are then updated by a gradient descent step on $\hat{J}$. The two updates are repeated.

> [!note] Practical considerations for saddle-point optimization
> The min-max problem is not a standard minimization. Convergence properties are weaker, oscillation can occur, and the two networks can fall out of balance. Many practical refinements have been developed to stabilize this training procedure, including spectral normalization, gradient penalties, and asymmetric learning rates. These refinements do not change the mathematical structure but improve the empirical behavior of training.

---

# Part VI. Generative Adversarial Networks as a Special Case

## The GAN Choice of $f$

The Generative Adversarial Network corresponds to the choice

$$
f(u) = u \log u - (u + 1) \log(u + 1).
$$

This function is related to the generator of the Jensen-Shannon divergence by an additive constant and a factor of two. The classical GAN therefore minimizes a quantity closely related to $D_{\mathrm{JS}}(P_X \Vert P_\theta)$.

## Computing the Conjugate

The conjugate $f^\ast$ is required to instantiate the VDM objective. Setting up the supremum,

$$
f^\ast(t) = \sup_{u > 0} \{tu - u \log u + (u + 1) \log(u + 1)\}.
$$

Differentiating the bracket with respect to $u$ and setting the result equal to zero:

$$
t - \log u - 1 + \log(u + 1) + 1 = 0,
$$

which simplifies to

$$
\log \frac{u + 1}{u} = -t.
$$

Solving for $u$:

$$
\frac{u + 1}{u} = e^{-t} \quad \Longrightarrow \quad u = \frac{1}{e^{-t} - 1}.
$$

For $u$ to be positive, $e^{-t} > 1$ is required, which means $t < 0$. The domain of the conjugate is therefore

$$
\mathrm{dom}\, f^\ast = (-\infty, 0).
$$

Substituting the optimal $u$ back into the bracket and simplifying gives

$$
f^\ast(t) = -\log(1 - e^t), \qquad t < 0.
$$

## The Activation Function for the GAN

The activation function $\sigma_f : \mathbb{R} \to (-\infty, 0)$ must map any real number into the negative real axis. The choice used in the classical GAN is

$$
\sigma_f(v) = -\log(1 + e^{-v}).
$$

This function takes values in $(-\infty, 0)$ for all $v \in \mathbb{R}$: as $v \to +\infty$ it approaches $0$ from below, and as $v \to -\infty$ it tends to $-\infty$.

The activation function admits a useful rewriting. The standard sigmoid function is $\sigma(v) = 1 / (1 + e^{-v})$. Taking the logarithm,

$$
\log \sigma(v) = -\log(1 + e^{-v}) = \sigma_f(v).
$$

Defining the **discriminator** network as the sigmoid of the witness output,

$$
D_w(x) = \sigma(V_w(x)) = \frac{1}{1 + e^{-V_w(x)}},
$$

allows the witness $T_w$ to be written compactly as

$$
T_w(x) = \log D_w(x).
$$

The discriminator $D_w(x)$ takes values in $(0, 1)$ and has the form of a binary classifier output.

## The GAN Objective

The two terms in the VDM objective are now expressed in terms of $D_w$. The first term is

$$
\mathbb{E}_{x \sim P_X}[T_w(x)] = \mathbb{E}_{x \sim P_X}[\log D_w(x)].
$$

For the second term, applying $f^\ast$ to $T_w(x) = \log D_w(x)$ gives

$$
f^\ast(T_w(x)) = -\log(1 - e^{\log D_w(x)}) = -\log(1 - D_w(x)).
$$

The objective becomes

$$
J_{\mathrm{GAN}}(\theta, w) = \mathbb{E}_{x \sim P_X}[\log D_w(x)] + \mathbb{E}_{z \sim \mathcal{N}(0, I)}[\log(1 - D_w(g_\theta(z)))].
$$

The training problem is

$$
\theta^\ast, w^\ast = \arg\min_\theta \max_w J_{\mathrm{GAN}}(\theta, w).
$$

This expression is the GAN objective in its original form. It has been derived as the specialization of variational divergence minimization to one particular $f$. Different choices of $f$ produce different objectives within the same general framework — the family of $f$-GANs.

> [!note] Role of the discriminator output
> The function $D_w(x) \in (0, 1)$ is interpreted as the discriminator's estimate of the probability that $x$ is drawn from $P_X$ rather than from $P_\theta$. The first term in $J_{\mathrm{GAN}}$ is maximized when $D_w(x) \to 1$ on real samples. The second term is maximized when $D_w(x) \to 0$ on generated samples. The discriminator therefore behaves as a binary classifier distinguishing real from generated data.

---

# Part VII. Training the GAN

## Architecture

A GAN consists of two networks.

The **generator** $g_\theta$ takes a latent vector $z \in \mathbb{R}^k$ drawn from $\mathcal{N}(0, I)$ and produces a candidate sample $\hat{x} = g_\theta(z) \in \mathbb{R}^d$. For image generation, $g_\theta$ is typically a convolutional network with upsampling layers. For lower-dimensional data, an MLP suffices.

The **discriminator** $D_w$ takes a candidate sample $x \in \mathbb{R}^d$ and produces a scalar in $(0, 1)$. Architecturally, $D_w$ is a binary classifier consisting of $V_w$ followed by a sigmoid. For images, $V_w$ is typically a CNN; for tabular data, an MLP.

## Updating the Discriminator

With $\theta$ held fixed, the discriminator parameters $w$ are updated by gradient ascent on $J_{\mathrm{GAN}}$. The update rule is

$$
w \leftarrow w + \eta_w \nabla_w J_{\mathrm{GAN}}(\theta, w),
$$

where $\eta_w$ is a step size. In practice, the expectations are estimated from minibatches:

A minibatch of real data $\{x_1, \ldots, x_B\}$ is drawn from $\mathcal{D}$, and a minibatch of noise vectors $\{z_1, \ldots, z_B\}$ is drawn from $\mathcal{N}(0, I)$. The generator (with $\theta$ fixed) produces fake samples $\hat{x}_i = g_\theta(z_i)$. The empirical objective is

$$
\hat{J}_{\mathrm{GAN}}(\theta, w) = \frac{1}{B} \sum_{i=1}^B \log D_w(x_i) + \frac{1}{B} \sum_{i=1}^B \log(1 - D_w(\hat{x}_i)).
$$

Backpropagation through $D_w$ provides the gradient $\nabla_w \hat{J}_{\mathrm{GAN}}$, which is used to update $w$.

## Updating the Generator

With $w$ held fixed, the generator parameters $\theta$ are updated by gradient descent on $J_{\mathrm{GAN}}$. The first term of the objective does not depend on $\theta$, so only the second term contributes to the gradient:

$$
\theta \leftarrow \theta - \eta_\theta \nabla_\theta \big\{ \mathbb{E}_{z \sim \mathcal{N}(0, I)}[\log(1 - D_w(g_\theta(z)))] \big\}.
$$

In practice, a minibatch of noise $\{z_1, \ldots, z_B\}$ is sampled, the generator produces $\hat{x}_i = g_\theta(z_i)$, and the empirical generator loss is

$$
\frac{1}{B} \sum_{i=1}^B \log(1 - D_w(g_\theta(z_i))).
$$

Backpropagation through $D_w$ and then through $g_\theta$ provides the gradient with respect to $\theta$, which is used to update the generator.

## Alternation and Convergence

One training iteration consists of one discriminator step followed by one generator step. Some variants perform multiple discriminator updates per generator update, to keep the discriminator close to optimal. The two networks improve in alternation.

At convergence in the idealized setting where $\mathcal{T}$ is rich enough to contain the optimal witness, the generator distribution equals the data distribution, $P_\theta = P_X$, and the discriminator output is the constant $\tfrac{1}{2}$ everywhere: the discriminator can do no better than random guessing.

> [!note] The non-saturating generator loss
> The original generator loss $\log(1 - D_w(g_\theta(z)))$ has vanishingly small gradients with respect to $\theta$ when $D_w(g_\theta(z))$ is close to zero — that is, when the discriminator confidently identifies generated samples as fake. Early in training, this is the typical regime, and the generator receives little signal. A standard practical modification is to replace the generator loss with $-\log D_w(g_\theta(z))$, which has the same fixed points but provides larger gradients in the regime where the discriminator is winning. This modification is not part of the mathematical derivation but is part of the standard training recipe.

---

# Part VIII. The Classifier-Guided Sampler Interpretation

The GAN objective can also be derived from a different starting point that does not invoke $f$-divergences. The two derivations yield the same objective, which provides an alternative perspective on what the training procedure accomplishes.

## Binary Classification Between Real and Generated Samples

Consider a trained generator $g_\theta$ that defines an implicit distribution $P_\theta$. To assess how close $P_\theta$ is to $P_X$, a binary classifier $D_w(x)$ is trained to distinguish real samples (drawn from $P_X$) from generated samples (drawn from $P_\theta$). Real samples are labeled with $y = 1$, generated samples with $y = 0$, and $D_w(x)$ is interpreted as the predicted probability that $x$ is real.

The standard log-likelihood objective for binary classification is

$$
\max_w \; \mathbb{E}_{x \sim P_X}[\log D_w(x)] + \mathbb{E}_{x \sim P_\theta}[\log(1 - D_w(x))].
$$

The two terms maximize the log-probability of the correct label under the classifier. This is the standard cross-entropy loss for binary classification, written here as a maximization rather than a minimization (the sign is flipped).

The expression coincides with the inner maximization in $J_{\mathrm{GAN}}$. The GAN discriminator is therefore a binary classifier trained to distinguish $P_X$ from $P_\theta$, with the maximization performed over the network parameters $w$.

## Adversarial Generator Update

If the classifier $D_w$ succeeds in distinguishing real from generated samples, the two distributions are statistically distinguishable. If $D_w$ fails, this is taken as evidence (subject to qualifications discussed below) that $P_\theta$ has moved closer to $P_X$. The generator parameters $\theta$ are therefore updated to reduce the classifier's ability to distinguish real from generated samples — that is, to make $D_w$ fail.

The combined objective is

$$
\min_\theta \max_w \; \big\{ \mathbb{E}_{x \sim P_X}[\log D_w(x)] + \mathbb{E}_{z \sim \mathcal{N}(0, I)}[\log(1 - D_w(g_\theta(z)))] \big\}.
$$

This is identical to $J_{\mathrm{GAN}}$ as derived from the $f$-divergence framework. The two derivations agree on the form of the objective.

> [!note] Equivalence of the two derivations
> The classifier-guided derivation begins with a binary classification problem and adds an adversarial generator update. The variational derivation begins with an $f$-divergence and applies the Fenchel conjugate. The two paths arrive at the same training objective. The Fenchel duality machinery makes precise the sense in which a binary classifier between $P_X$ and $P_\theta$ is a witness of the divergence between the two distributions.

---

# Part IX. Limitations of the Classifier-Guided View

The classifier-guided interpretation suggests that classifier failure on samples from $P_X$ and $P_\theta$ implies $P_X = P_\theta$. This implication holds only when the classifier is drawn from a sufficiently rich function class. With a restricted classifier, failure to distinguish is not sufficient to conclude that the two distributions agree.

## Counter-Example with a Restricted Classifier

Consider two distributions $P_X$ and $P_\theta$ on $\mathbb{R}^2$ constructed as follows. The distribution $P_X$ is concentrated along one diagonal of a square region, and $P_\theta$ is concentrated along the other diagonal. A linear classifier on $\mathbb{R}^2$ consists of a single hyperplane, which corresponds to a straight line in the plane. No linear classifier can separate the two diagonals: any line either intersects both diagonals symmetrically or misses both, and neither configuration produces a useful classification rule.

A more expressive classifier — for instance, a neural network with non-linear activations — can capture the "X"-shaped arrangement and distinguish the two distributions. The failure of the linear classifier reflects its restricted capacity, not the equality of the two distributions.

## Consequence for Training

This observation has a direct consequence for GAN training. In the variational framework, the function class $\mathcal{T}$ over which the witness $T$ is optimized must contain — or be close to containing — the optimal witness $T^\ast$. If $\mathcal{T}$ is too restrictive, the supremum over $\mathcal{T}$ is a strictly loose lower bound on $D_f$, and minimizing the bound does not drive the actual divergence to zero.

This is also the reason the generator and discriminator must be trained **simultaneously**, rather than in two separate phases. If the discriminator is trained to convergence first and then frozen while the generator is updated, the generator can learn to fool that specific discriminator without making $P_\theta$ close to $P_X$ in any meaningful sense. The generator and discriminator must co-evolve so that, as $P_\theta$ improves, the discriminator remains capable of identifying the residual differences between the two distributions.

In practice, the balance between the two networks must be maintained throughout training. A discriminator that is too weak produces unhelpful gradients for the generator. A discriminator that is too strong saturates the generator's loss and similarly suppresses useful gradients. The training procedure must keep both networks roughly matched in capability.

---

# Part X. Variants of GANs

The general GAN framework admits many specializations. Two important variants address the architecture of the generator and the conditional setting.

## Deep Convolutional GAN

In a basic GAN applied to image data, the generator is sometimes implemented as an MLP that produces a flat output vector, which is then reshaped into an image. This design ignores the spatial structure of images and is inefficient for high-resolution generation.

The Deep Convolutional GAN, or DC-GAN, replaces the MLP generator with a fully convolutional network. The construction proceeds in stages. The latent vector $z \in \mathbb{R}^k$ is first reshaped into a low-resolution feature map with many channels. A sequence of **transposed convolutional layers** then progressively increases the spatial resolution while reducing the number of channels, until the final layer produces an output tensor of the target image dimensions:

$$
z \in \mathbb{R}^k \;\longrightarrow\; \text{reshape} \;\longrightarrow\; \text{transposed-convolution layers} \;\longrightarrow\; \hat{x} \in \mathbb{R}^d.
$$

The transposed convolution is the spatial upsampling counterpart of the ordinary convolution. It increases spatial resolution while applying a learned filter at each location.

The discriminator in a DC-GAN mirrors this design. It is a standard convolutional classifier that downsamples its input through ordinary convolutional layers and produces a sigmoid output. The DC-GAN architecture was the first GAN variant that produced visually plausible images at modest resolution, and it established the convolutional architectural recipe used by many subsequent generative models.

## Conditional GAN

The standard GAN samples from the unconditional data distribution $P_X$. Many applications require sampling from a conditional distribution $P_{X \mid Y}$, where $Y$ is a label, attribute, or auxiliary input.

The training data in this setting consists of pairs:

$$
\mathcal{D} = \{(x_1, y_1), (x_2, y_2), \ldots, (x_n, y_n)\}, \qquad (x_i, y_i) \overset{\mathrm{i.i.d.}}{\sim} P_{X, Y}.
$$

The generator and discriminator both receive the condition $y$ as an additional input:

$$
\hat{x} = g_\theta(z, y), \qquad D_w(x, y) \in (0, 1).
$$

The training objective becomes

$$
J_{\mathrm{cGAN}}(\theta, w) = \mathbb{E}_{(x, y) \sim P_{X, Y}}[\log D_w(x, y)] + \mathbb{E}_{z, y}[\log(1 - D_w(g_\theta(z, y), y))],
$$

where $z \sim \mathcal{N}(0, I)$ and $y$ is drawn from the marginal $P_Y$ or directly from the dataset.

The form of the condition $y$ depends on the application. For class labels, $y$ is typically a one-hot vector in $\{0, 1\}^K$. For text-conditioned image generation, $y$ is an embedding produced by a pretrained text encoder. For image-to-image translation tasks, $y$ is itself an image.

The conditional GAN samples from $P_{X \mid Y}$ rather than $P_X$. At inference time, the generated sample reflects both the random noise $z$ and the specified condition $y$.

---

# Part XI. Inference with a Trained Model

The training procedure is computationally intensive, but generating samples from a trained model is a single forward pass through the generator network.

For an unconditional model with trained parameters $\theta^\ast$:

A latent vector $z_{\text{test}}$ is drawn from $\mathcal{N}(0, I)$. The generator computes $x_{\text{test}} = g_{\theta^\ast}(z_{\text{test}})$. The output $x_{\text{test}}$ is a sample from $P_{\theta^\ast}$, which, if training has succeeded, is close to $P_X$. New samples are produced by drawing additional $z$ values.

For a conditional model:

A condition $y$ is specified. A latent vector $z_{\text{test}}$ is drawn from $\mathcal{N}(0, I)$. The generator computes $x_{\text{test}} = g_{\theta^\ast}(z_{\text{test}}, y)$. The output is a sample from $P_{\theta^\ast}(\cdot \mid y) \approx P_{X \mid Y}(\cdot \mid y)$.

The discriminator $D_{w^\ast}$ is not used at inference time. Its sole purpose during training is to provide a gradient signal for the generator. Once the generator is trained, the discriminator is discarded.

---

# Summary of the Chapter

The chapter has developed the variational divergence minimization framework and identified the classical GAN as a particular instance.

The starting point was the family of $f$-divergences, parameterized by a convex function $f$ with $f(1) = 0$. Different choices of $f$ recover standard divergences including forward KL, reverse KL, Jensen-Shannon, and total variation. Each $f$-divergence is defined by an integral over both densities $p_X$ and $p_\theta$, neither of which is available in the generative modeling setting.

The Fenchel conjugate $f^\ast(t) = \sup_u \{tu - f(u)\}$ and the biconjugation identity $f(u) = \sup_t \{tu - f^\ast(t)\}$ provide the technical tool for rewriting $D_f$ in a sample-friendly form. The result is the variational representation

$$
D_f(P_X \Vert P_\theta) = \sup_T \big\{ \mathbb{E}_{P_X}[T(x)] - \mathbb{E}_{P_\theta}[f^\ast(T(x))] \big\}.
$$

Restricting the supremum to a function class $\mathcal{T}$ realized by a neural network gives a lower bound on $D_f$. Minimizing this lower bound over the generator parameters $\theta$, while maximizing it over the witness parameters $w$, defines the saddle-point objective of variational divergence minimization.

The specific choice $f(u) = u \log u - (u + 1) \log(u + 1)$ produces the classical GAN objective

$$
J_{\mathrm{GAN}}(\theta, w) = \mathbb{E}_{x \sim P_X}[\log D_w(x)] + \mathbb{E}_{z \sim \mathcal{N}(0, I)}[\log(1 - D_w(g_\theta(z)))],
$$

where $D_w(x) = \sigma(V_w(x))$ is the sigmoid of a neural network output. The same objective is obtained by training a binary classifier to distinguish real from generated samples and updating the generator to fool the classifier. The two derivations are equivalent.

The success of the variational framework depends on the function class $\mathcal{T}$ being rich enough to approximate the optimal witness. A counter-example with a linear classifier on $\mathbb{R}^2$ illustrates that classifier failure with a restricted classifier does not imply equality of distributions. This is the formal reason that the generator and discriminator must be trained simultaneously.

Two important variants extend the basic GAN. The Deep Convolutional GAN replaces the MLP generator with a fully convolutional architecture based on transposed convolutions. The Conditional GAN extends the framework to model conditional distributions $P_{X \mid Y}$ by providing both networks with a label or attribute input.

The chapter has answered the first two of the four open questions from Part 1: divergence estimation from samples is achieved through the variational representation, and the choice of divergence is encoded by the choice of $f$. The architectural choice of generator and discriminator, and the practical details of stabilizing the saddle-point optimization, are partially addressed but remain the subject of much continuing research. Subsequent chapters develop alternative families of generative models — variational autoencoders, normalizing flows, diffusion models, autoregressive models — each of which provides a different set of answers to the same four questions.

---

# Key Notation Reference (Additions for this Chapter)

| Symbol | Meaning |
|---|---|
| $D_f(P \Vert Q)$ | $f$-divergence between distributions $P$ and $Q$ |
| $f$ | Convex generator function with $f(1) = 0$ |
| $f^\ast$ | Fenchel conjugate of $f$, defined by $f^\ast(t) = \sup_u \{tu - f(u)\}$ |
| $\mathrm{dom}\, f^\ast$ | Domain of the conjugate (where the supremum is finite) |
| $D_{\mathrm{KL}}(P \Vert Q)$ | Kullback-Leibler divergence; obtained from $f(u) = u \log u$ |
| $D_{\mathrm{JS}}(P \Vert Q)$ | Jensen-Shannon divergence; symmetric and bounded |
| $D_{\mathrm{TV}}(P, Q)$ | Total variation distance; obtained from $f(u) = \tfrac{1}{2}\lvert u - 1 \rvert$ |
| $T(x)$ | Variational witness function, $T : \mathcal{X} \to \mathrm{dom}\, f^\ast$ |
| $\mathcal{T}$ | Function class over which $T$ is optimized |
| $T_w$ | Witness function realized by a neural network with parameters $w$ |
| $V_w(x)$ | Raw neural network output before activation, $V_w : \mathcal{X} \to \mathbb{R}$ |
| $\sigma_f$ | Activation function mapping $\mathbb{R}$ into $\mathrm{dom}\, f^\ast$ |
| $D_w(x)$ | GAN discriminator output, $D_w(x) \in (0, 1)$ |
| $w^\ast$ | Optimal discriminator parameters |
| $J(\theta, w)$ | The VDM objective |
| $J_{\mathrm{GAN}}(\theta, w)$ | The classical GAN objective, a special case of $J(\theta, w)$ |
| $P_{X, Y}$ | Joint distribution of data and labels in the conditional setting |
| $P_{X \mid Y}$ | Conditional data distribution given label $y$ |
| $g_\theta(z, y)$ | Conditional generator taking noise and label as input |
| $D_w(x, y)$ | Conditional discriminator taking sample and label as input |
