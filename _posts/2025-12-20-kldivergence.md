---
layout: post
title: KL Divergence
date: 2025-12-20 00:12:00
description: Mathematics of KL Divergence
tags: mathematics, machine learning
categories: mathematics, ml
tabs: true
---


We first define the relative entropy (KL-divergence), cross entropy, and entropy in the context of discrete probability distributions. We then provide a definition of the KL-divergence for continuous random variables.


## The KL-divergence for Discrete Distributions

Assume two probability distributions $p(\cdot)$ and $q(\cdot)$ over elements in some discrete sets $\mathcal{X}\_p$ and $\mathcal{X}\_q$ respectively. That is, $p(x)$ or $q(x)$ denote the respective probabilities, which are strictly positive unless $x \not\in \mathcal{X}\_p$ for which $p(x) = 0$ (or similarly $x \not\in \mathcal{X}\_q$ for which $q(x) = 0$).

A key measure for the proximity between the distributions $p(\cdot)$ and $q(\cdot)$ is the *Kullback–Leibler divergence*, also shortened as *KL-divergence*, and also known as the *relative entropy*. It is denoted $D\_{\mathrm{KL}}(p\parallel q)$ and as long as $\mathcal{X}\_p \subseteq \mathcal{X}\_q$ it is the expected value of $\log p(X)/q(X)$ where $X$ is a random variable following the probability law $p(\cdot)$. Namely,

$$
D_{\mathrm{KL}}(p\parallel q) = \sum_{x\in\mathcal{X}_p} p(x) \log\frac{p(x)}{q(x)}.
\qquad (B.1)
$$

Further if $\mathcal{X}\_p \not\subseteq \mathcal{X}\_q$, that is if there is some element in $\mathcal{X}\_p$ that is not in $\mathcal{X}\_q$, then by definition $D\_{\mathrm{KL}}(p\parallel q)=+\infty$. This definition as infinity is natural since we would otherwise divide by $0$ for some $q(x)$.

Observe that the expression for $D_{\mathrm{KL}}(p\parallel q)$ from (B.1) can be decomposed into the difference of $H(p)$ from $H(p,q)$ via,

$$
D_{\mathrm{KL}}(p\parallel q)
=
\sum_{x\in\mathcal{X}} p(x)\log\frac{1}{q(x)}
-
\sum_{x\in\mathcal{X}} p(x)\log\frac{1}{p(x)}
.
$$

where the first term is $H(p,q)$ and the second term is $H(p)$.

Here,

$$
H(p,q) = -\sum_{x\in\mathcal{X}} p(x)\log q(x)
\qquad (B.2)
$$

is called the *cross entropy* of $p$ and $q$ and

$$
H(p) = -\sum_{x\in\mathcal{X}} p(x)\log p(x)
\qquad (B.3)
$$

is called the *entropy* of $p$. Hence in words, the KL-divergence or relative entropy of $p$ and $q$ is the cross entropy of $p$ and $q$ with the entropy of $p$ subtracted. Note that in case where there are only two values in $\mathcal{X}$, say $0$ and $1$, where we denote $p(1)=p\_1$ and $q(1)=q\_1$, we have

$$
H(p) = -\bigl(p_1\log p_1 + (1-p_1)\log(1-p_1)\bigr),
\qquad (B.4)
$$

$$
H(p,q) = -\bigl(p_1\log q_1 + (1-p_1)\log(1-q_1)\bigr).
\qquad (B.5)
$$

Some observations are in order. First observe that $D\_{\mathrm{KL}}(p\parallel q) \ge 0$. Further note that in general $D\_{\mathrm{KL}}(p\parallel q) \ne D\_{\mathrm{KL}}(q\parallel p)$ and similarly $H(p,q) \ne H(q,p)$. Hence as a “distance measure” the KL-divergence is not a true metric since it is not symmetric over its arguments. Nevertheless, when $p=q$ the KL-divergence is 0 and similarly the cross entropy equals the entropy. In addition, it can be shown that $D\_{\mathrm{KL}}(p\parallel q) = 0$ only when $p = q$. Hence the KL-divergence may play a role similar to a distance metric in certain applications. In fact, one may consider a sequence $q^{(1)},q^{(2)},\dots$ which has decreasing $D\_{\mathrm{KL}}(p\parallel q^{(t)})$ approaching $0$ as $t\to\infty$. For such a sequence the probability distributions $q^{(t)}$ approach$^1$ the target distribution $p$ since the KL-divergence convergences to $0$.

## The KL-divergence for Continuous Distributions

The KL-divergence in (B.1) naturally extends to arbitrary probability distributions that are not necessarily discrete. In our case let us consider continuous multi-dimensional distributions. In this case $p(\cdot)$ and $q(\cdot)$ are probability densities, and the sets $\mathcal{X}\_p$ and $\mathcal{X}\_q$ are their respective supports. Now very similarly to (B.1), as long as $\mathcal{X}\_p \subseteq \mathcal{X}\_q$ we define,

$$
D_{\mathrm{KL}}(p\parallel q) = \int_{x\in\mathcal{X}_p} p(x) \log\frac{p(x)}{q(x)} dx.
\qquad (B.6)
$$
