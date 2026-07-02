---
layout: post
title: "Generative AI Foundations — Part 1"
date: 2026-06-30
series: genai-foundations
series_part: 1
series_title: "Generative AI Foundations"
series_description: "A ground-up course covering the math and modern generative AI."
description: "Part 1"
tags: generative-ai, mathematics
categories: courses
---

> **A note on attribution.** These notes are directly inspired by [Prof. Prathosh A P's course *Mathematical Foundations of Generative AI* (BSDA5002)](https://study.iitm.ac.in/ds/course_pages/BSDA5002.html), offered through IIT Madras. Prof. Prathosh is an Assistant Professor in the Division of EECS at IISc Bangalore, and the course is one of the clearest treatments of deep generative models I have come across — covering probabilistic foundations, VAEs, GANs, diffusion models, and LLMs from the ground up.
>
> Much of the structure and topic progression here follows his course. Where I have departed from it is intentional: I have rewritten explanations with more mathematical detail and intermediate steps, filling in gaps that I found myself having to work through on paper. The goal is to make the ideas easier to grasp for someone encountering them for the first time, without sacrificing rigour. All credit for the original framing and pedagogy belongs to Prof. Prathosh — any errors in this retelling are mine alone.

# Deep Generative Models $\mathrm{DGM}$

## Objective

The objective of this course is to develop a mathematical foundation for Generative AI.

A generative model is a model that learns how data is distributed. After learning, the model should be able to create new samples that look as if they came from the same source as the training data.

In mathematical terms, the course studies the following basic problem:

> We observe data samples from an unknown probability distribution. We want to learn a model distribution that is close to the true data distribution. Then we want to generate new samples from the learned distribution.

The true data distribution is usually unknown. We only have a finite dataset. The main task is to use this finite dataset to build a useful approximation of the unknown distribution.

---

# Families of Deep Generative Models

The course will cover the main families of deep generative models.

## 1. Generative Adversarial Networks $\mathrm{GANs}$

A Generative Adversarial Network learns a generator by training it against a discriminator.

The generator tries to produce samples that look real. The discriminator tries to separate real samples from generated samples.

The training process is adversarial because the two networks have opposite goals.

A basic GAN has two components:

- a generator $G_\theta$, which maps a simple random variable into a generated sample;
- a discriminator $D_\phi$, which tries to distinguish real data from generated data.

A typical generator has the form

$$
z \sim p_z, \qquad \hat{x} = G_\theta(z).
$$

Here $z$ is sampled from a simple known distribution, such as a standard Gaussian distribution, and $\hat{x}$ is the generated sample.

![Training GANs](assets/img/training_gans.png)

## 2. Variational Autoencoders $\mathrm{VAEs}$

A Variational Autoencoder uses latent variables to model the data distribution.

A latent variable is an unobserved variable that explains hidden structure in the data. In a VAE, the model assumes that each observed data point $x$ is generated from a latent variable $z$.

The basic generative direction is

$$
z \sim p(z), \qquad x \sim p_\theta(x \mid z).
$$

A VAE also learns an approximate inference model

$$
q_\phi(z \mid x),
$$

which maps data points back to distributions over latent variables.

![Variational Autoencoders](assets/img/variational_autoencoders.png)

## 3. Denoising Diffusion Probabilistic Models $\mathrm{DDPMs}$ and Score-Based Models

Diffusion models generate data by reversing a noising process.

The forward process gradually adds noise to data. After many steps, the data becomes almost pure noise.

The reverse process learns to remove noise step by step. Starting from random noise, the model gradually produces a clean sample.

A high-level view is

$$
x_0 \rightarrow x_1 \rightarrow x_2 \rightarrow \cdots \rightarrow x_T,
$$

where $x_0$ is the clean data and $x_T$ is almost pure noise.

Generation runs in the reverse direction:

$$
x_T \rightarrow x_{T-1} \rightarrow \cdots \rightarrow x_1 \rightarrow x_0.
$$

![Denoising Diffusion Probabilistic Models](assets/img/ddpms.png)

Score-based models are closely related. They learn the score function

$$
\nabla_x \log p(x),
$$

which gives the direction in which the log-density increases most rapidly.

## 4. Auto-Regressive Models $\mathrm{AR}$ and Large Language Models $\mathrm{LLMs}$

Auto-regressive models generate a sequence one element at a time.

For a sequence

$$
x = (x_1, x_2, \ldots, x_T),
$$

an auto-regressive model factorizes the joint distribution as

$$
p(x_1, x_2, \ldots, x_T)
= \prod_{t=1}^{T} p(x_t \mid x_1, x_2, \ldots, x_{t-1}).
$$

Large Language Models are usually trained using this idea. Given previous tokens, the model predicts the next token.

For text generation, the model repeatedly samples

$$
x_t \sim p_\theta(x_t \mid x_1, \ldots, x_{t-1}).
$$

This turns next-token prediction into full sequence generation.

## 5. State-Space Models $\mathrm{SSMs}$

State-space models represent sequence processing using hidden states.

A general state-space form is

$$
h_t = A h_{t-1} + B x_t,
$$

$$
y_t = C h_t + D x_t.
$$

Here:

- $x_t$ is the input at time $t$;
- $h_t$ is the hidden state;
- $y_t$ is the output;
- $A, B, C, D$ are parameters or parameterized transformations.

Modern examples include S4 and Mamba. These models are important for long-sequence modeling.

## 6. Preference-Based Alignment for $\mathrm{LLMs}$

Preference-based alignment methods train language models to better match human preferences.

Important examples include:

- Reinforcement Learning from Human Feedback $\mathrm{RLHF}$, which uses Proximal Policy Optimization $\mathrm{PPO}$ as its RL optimizer;
- Direct Preference Optimization $\mathrm{DPO}$, a direct alternative to RLHF that avoids reinforcement learning and optimizes the model on preference data via supervised learning.

The goal is not only to generate likely text, but to generate useful, safe, and preference-aligned text.

With that overview in place, we can now set up the common mathematical framework used throughout the course.

---

# Generative Models

## What is a Generative Model?

A generative model is any model that, after training on examples, can produce new examples of the same kind. 

Examples of modern generative models include:

| System | Input | Output |
|---|---|---|
| ChatGPT, Gemini, Claude | Text prompt | New text, code |
| DALL·E, Stable Diffusion | Text prompt | New image |
| Speech generators (TTS) | Text | New audio (`.wav` file) |



All three rows have an input that conditions the output, so these are conditional generators, i.e. they produce something given an input. The input is usually called the condition or prompt.
There are also unconditional generators, for example, a model trained on faces that just outputs a fresh random face each time you press a button, with no prompt at all. We will start with the unconditional case because it is simpler, then build up to the conditional case.

These systems generate different types of data.

## Conditional Text Generators

Models such as ChatGPT, Gemini, and Claude are conditional text generators.

They generate text conditioned on an input prompt.

If the prompt is denoted by $c$, and the output text is denoted by $x$, then the model is trying to sample from a conditional distribution

$$
p_\theta(x \mid c).
$$

The condition $c$ may be a natural language instruction, a question, a code prompt, a document, or a conversation history.

The output $x$ may be natural language, code, structured text, or a mixture of these.

## Conditional Image Generators

DALL-E and Stable Diffusion are conditional image generators.

They generate an image conditioned on a text prompt.

If the text prompt is $c$, and the generated image is $x$, then the model samples from

$$
p_\theta(x \mid c).
$$

The same mathematical idea appears again: the model learns a distribution over outputs conditioned on inputs.

## Speech Generators

A speech generator maps text to speech.

At a high level, the mapping is

$$
\text{text} \longrightarrow \text{waveform or speech file}.
$$

If the text input is $c$, and the generated speech signal is $x$, then the model learns

$$
p_\theta(x \mid c).
$$

The output may be stored as a waveform, such as a WAV file, or represented internally as an audio signal.

---

# Starting Point: Data
Everything we do begins with data. Without data, there is nothing to learn. Thus our starting point is data.

We are given a dataset

$$
\mathcal{D} = \{x_1, x_2, \ldots, x_n\}.
$$


Each $x_i$ is a data point. Each $x_i$ is one sample (one image, one sentence, one audio clip). We assume:


The dataset is assumed to consist of independent and identically distributed samples from an unknown data distribution $P_X$. This is written as

$$
x_1, x_2, \ldots, x_n \overset{\mathrm{i.i.d.}}{\sim} P_X.
$$

This means two things.

First, the samples are independent:

$$
x_i \perp x_j \qquad \text{for } i \neq j.
$$

Second, all samples come from the same distribution:

$$
x_i \sim P_X \qquad \text{for all } i.
$$

#### What does "iid" mean?
- **Independent**: knowing one sample tells you nothing about the others.
- **Identically distributed**: all of them come from the same underlying probability distribution $P_X$.

> [!example] Coin-flip analogy
> Flipping a fair coin 100 times gives 100 iid samples. Each flip is independent of the others (the coin has no memory), and each flip has the same probability of heads or tails (the distribution does not change between flips).
>
> Now imagine the "samples" are entire images of cats instead of coin flips. The assumption is the same: each cat photo was drawn from some big abstract "distribution of cat photos", independently of the others.


The distribution $P_X$ is unknown. We do not have a formula for it. We only observe samples from it.

## Data as Vectors

Each data point is represented as a vector:

$$
x_i \in \mathbb{R}^d.
$$

Here $d$ is the dimensionality of the data. Simply put, it means how many numbers it takes to specify one sample.

The value of $d$ depends on how the data is represented.

For a tabular dataset, $d$ may be the number of features.

For an image, $d$ depends on height, width, and number of channels.

For example, suppose each image has height $400$, width $400$, and $3$ color channels. Then the number of scalar values in one image is

$$
d = 400 \times 400 \times 3 = 480000.
$$

So each image can be represented as a vector in

$$
\mathbb{R}^{480000}.
$$

The vector form does not mean that the image is visually one-dimensional. It only means that all pixel values are collected into one long vector so that mathematical operations can be applied. So one image is a single point in $\mathbb{R}^{480{,}000}$. The whole dataset is a cloud of $n$ points sitting somewhere in this enormous space.

This is one of the central difficulties of generative modeling: the space is **astronomically large**, and we usually have far fewer than 480,000 data points to learn from. The number of *possible* 400×400 color images in $\mathbb{R}^{480{,}000}$ vastly exceeds the number of cat photos that have ever been taken.

## Dataset Example

Suppose

$$
\mathcal{D} = \{x_1, x_2, \ldots, x_n\}, \qquad n = 1000.
$$

Then the dataset contains $1000$ observed samples.

The modeling assumption is

$$
x_i \sim P_X,
$$

and for distinct samples,

$$
x_i \perp x_j \qquad \text{when } i \neq j.
$$

The symbol $X$ denotes a random variable. The observed data points $x_1, \ldots, x_n$ are instances or realizations of that random variable.

Thus:

$$
X \sim P_X,
$$

and each sample $x_i$ is one observed value of $X$.

### What is $P_X$?

$X$ is a **random variable** that takes values in $\mathbb{R}^d$. The notation $X \sim P_X$ means "$X$ is drawn from the distribution $P_X$".

$P_X$ is a probability distribution over $\mathbb{R}^d$. You can imagine it as a function that assigns a "density" to every point in the space:
- Points that look like real cat photos get **high density**.
- Points that look like pure static noise get essentially **zero density**.
- Points in between (a photo of a dog, say) get some density depending on how plausibly cat-like they could be.

![Probability distribution](assets/img/probability_distribution.png)

> [!warning] The key catch
> We **do not know** $P_X$. We only have $n$ samples from it. The whole job of generative modeling is to figure out $P_X$ from these samples, well enough to be able to draw new samples from it.



In the notes, we see $n = 1000$ as a small example. Real datasets are vastly larger:
- ImageNet: roughly 14 million images
- LAION-5B (used for training image models): ~5.85 billion image–text pairs
- LLM training data: trillions of tokens

But $n$ is always finite, while $\mathbb{R}^d$ is infinite. Generative modeling is the art of making sensible guesses about the infinite based on the finite.

> [!note] Notation gotcha
> The notes write $x_i \neq x_j$ to remind us that the samples in the dataset are **distinct points**, but they are all drawn from the **same distribution** $P_X$. Distinct samples, same source.
---

# Diagram: Image as a High-Dimensional Vector

![Image as a high-dimensional vector](assets/img/image_high_dim_vector.png)


---

# Generative Modeling

We can now restate the setup in compact form.

## Given

We are given data

$$
\mathcal{D} = \{x_1, x_2, x_3, \ldots, x_n\},
$$

where

$$
x_i \overset{\mathrm{i.i.d.}}{\sim} P_X.
$$

The distribution $P_X$ is unknown.

## Goal

The goal of generative modeling is:

$$
\text{Estimate } P_X \text{ and learn to sample from it.}
$$

This goal has two parts.

First, the model should learn a distribution close to the real data distribution.

Second, the model should allow us to generate new samples.

If the learned model distribution is denoted by $P_\theta$, then we want

$$
P_\theta \approx P_X.
$$

Once this is achieved, sampling from $P_\theta$ should produce samples that look like samples from $P_X$.

Even if you knew the exact formula for $P_X$, that does not automatically mean you can produce a fresh sample from it. Sampling from an arbitrary high-dimensional distribution is its own hard problem.

For instance, you might write down a probability density function on $\mathbb{R}^{480{,}000}$ that perfectly describes cat photos — but how would you actually *generate* a new cat photo from that formula? You cannot enumerate every possible point in the space and pick one weighted by its density. The space is too big.

So generative modeling cares about **both** the description and the production mechanism.


---

# General Principle of Generative Models

The general principle has three steps.

## Step 1: Assume a Parametric Family

Assume a parametric family of distributions for the data.

This family is denoted by

$$
\{P_\theta : \theta \in \Theta\}.
$$

Here:

- $\theta$ denotes the model parameters;
- $\Theta$ denotes the parameter space;
- $P_\theta$ is the model distribution for a particular value of $\theta$.

In deep generative models, $P_\theta$ is represented using a deep neural network.

We assume there is a family of candidate distributions $\{P_\theta\}$, indexed by parameters $\theta$:

$$
P_\theta : \text{a candidate distribution, parameterized by } \theta
$$

In deep generative models, $\theta$ is the set of weights of a **deep neural network**. Different choices of $\theta$ give different distributions $P_\theta$.

The neural network does not merely output a number. It represents a distribution, a sampling process, or a transformation that induces a distribution. 

> [!example] Intuition for the parametric family
> Imagine a giant catalogue of probability distributions, each tagged with an index $\theta$. We are betting that **somewhere in this catalogue** lives a distribution that is very close to the true $P_X$. Our job is to flip through the catalogue and find it.
>
> The "catalogue" here is the set of all possible weight settings of a neural network. There are inconceivably many — large enough to contain very flexible distributions, hopefully one of which is a good match for the data.

## Step 2: Define a Divergence Between $P_X$ and $P_\theta$

We need a way to measure how different $P_\theta$ is from $P_X$. We need a way to measure **how far apart** two distributions are. Such a measure is called a **divergence** and is denoted $D(P \,\|\, Q)$.

This is done using a divergence or distance-like quantity:

$$
D(P_X \Vert P_\theta).
$$

The notation $D(P_X \Vert P_\theta)$ means that the divergence is measured between the true data distribution $P_X$ and the model distribution $P_\theta$.

A divergence should satisfy

$$
D(P_X \Vert P_\theta) \geq 0.
$$

Usually, we also want

$$
D(P_X \Vert P_\theta) = 0
\quad \text{if and only if} \quad
P_X = P_\theta.
$$ 

(zero only when the two distributions are identical)


The divergence tells us how far the current model is from the target distribution.

> [!note] Divergence is not the same as distance
> A proper **distance** is symmetric: $d(P, Q) = d(Q, P)$. Many divergences are not symmetric: in general, $D(P \,\|\, Q) \neq D(Q \,\|\, P)$. That asymmetry actually matters, and we will see later why it changes how the model behaves.


## Step 3: Solve an Optimization Problem

Pick the $\theta$ that makes $P_\theta$ as close as possible to $P_X$.

The model parameters are chosen by minimizing the divergence:

$$
\theta^* = \arg\min_{\theta} D(P_X \Vert P_\theta).
$$

Here $\theta^*$ is the best parameter value according to the chosen divergence.

After optimization, the learned model distribution is

$$
P_{\theta^*}.
$$

The aim is that

$$
P_{\theta^*} \approx P_X.
$$

Then samples from $P_{\theta^*}$ should resemble samples from the true data distribution.

This is the **central optimization problem of generative modeling**. Different choices of divergence give different families of models. Different parameterizations of $P_\theta$ also give different families.

> [!tip] What changes across model families
> The families we will study differ in their choices of:
> - **Parameterization** (Step 1): How is $P_\theta$ defined? Is it implicit (GAN) or explicit (VAE, diffusion)?
> - **Divergence** (Step 2): Are we minimizing KL divergence? Jensen–Shannon? Wasserstein? Something else?
> - **Optimization** (Step 3): How do we actually minimize the divergence when we only have samples — not the full densities?


---

# Latent-Variable Generator Example

Let us walk through one concrete way to set up a generative model. This is the **latent variable** approach, which sits underneath VAEs, GANs, and diffusion models.

## Latent Random Variable

Let

$$
z \in \mathbb{R}^k
$$

be a random variable with a known distribution that lives in a low-dimensional space.

A common choice is the standard multivariate Gaussian distribution:

$$
z \sim \mathcal{N}(0, I).
$$

Here $z$ has $k$ dimensions (with $k$ usually much smaller than $d$), and it is drawn from a **standard multivariate Gaussian** — mean zero, identity covariance. This distribution is **known** and **easy to sample from**: you just draw $k$ independent standard normal random numbers.

The distribution of $z$ is chosen to be simple because generation begins by sampling $z$.

> [!example] Why start from noise?
> We need a starting point we **can sample from**. The Gaussian is the simplest non-trivial distribution we know how to sample. Computers can draw from it cheaply. We will then **transform** this simple noise into something complex.

## Generator Function

Now we define a function:

$$
g_\theta : \mathbb{R}^k \to \mathbb{R}^d, \quad \hat{x} = g_\theta(z)
$$

In words: $g_\theta$ is a function that takes a $k$-dimensional noise vector and produces a $d$-dimensional sample (for example, an image with $d = 480{,}000$).

Define

$$
\hat{x} = g_\theta(z).
$$

Then $\hat{x}$ is a generated sample.

The generated sample belongs to the same space as the data:

$$
\hat{x} \in \mathbb{R}^d.
$$

The random variable $z$ has a known distribution, but $\hat{x}$ has a different distribution. The distribution of $\hat{x}$ is induced by the transformation $g_\theta$.

This induced distribution is denoted by

$$
\hat{x} \sim P_\theta.
$$

More explicitly,

$$
z \sim \mathcal{N}(0, I),
\qquad
\hat{x} = g_\theta(z),
\qquad
\hat{x} \sim P_\theta.
$$

## Generator as a Neural Network

In deep generative modeling, $g_\theta$ is usually a neural network.

The parameters $\theta$ are the weights and biases of the network.

The role of $g_\theta$ is to transform simple noise into structured data.

For example:

- simple Gaussian noise becomes an image;
- simple Gaussian noise becomes an audio signal;
- a latent vector becomes a realistic sample in data space.

## Diagram: Latent Variable to Generated Sample

![Latent variable to generated sample](assets/img/latent_variable_to_generated_sample.png)

The same idea can be written as a single mathematical pipeline:

$$
z \sim \mathcal{N}(0,I)
\quad \longrightarrow \quad
\hat{x} = g_\theta(z)
\quad \longrightarrow \quad
\hat{x} \sim P_\theta.
$$

---

# Density Induced by the Generator

The notes denote the density of the generated sample by

$$
p_\theta(\hat{x}).
$$

This means the generated random variable $\hat{x}$ follows the model distribution $P_\theta$.

When $z$ is random, then $\hat{x} = g_\theta(z)$ is also random. The neural network does not change that - it just **reshapes** the distribution.

More carefully, one usually writes

$$
\hat{x} \sim P_\theta,
$$

where $P_\theta$ is the distribution, and $p_\theta$ is its density if a density exists.

We call the distribution of $\hat{x} = g_\theta(z)$ the **pushforward** distribution and denote it $P_\theta$:

$$
z \sim \mathcal{N}(0, I), \qquad \hat{x} = g_\theta(z), \qquad \hat{x} \sim P_\theta.
$$

So $P_\theta$ is the distribution you get when you draw from $\mathcal{N}(0, I)$ and pass through the network $g_\theta$. The shape of $P_\theta$ depends entirely on $\theta$.

The distinction is useful:

- $P_\theta$ denotes the probability distribution;
- $p_\theta(x)$ denotes the density or probability mass function evaluated at $x$.

For continuous data, $p_\theta(x)$ is a probability density.

For discrete data, $p_\theta(x)$ is a probability mass function.

> [!tip] Different $\theta$ → different $P_\theta$
> Change $\theta$ and the output distribution changes. The whole "catalogue" of distributions from Step 1 is literally generated this way: every neural network weight setting $\theta$ gives one distribution $P_\theta$.

---

# Optimization Problem

The training problem is to find parameters $\theta$ such that the generated distribution $P_\theta$ is close to the true data distribution $P_X$.

This is written as

$$
\theta^* = \arg\min_{\theta} D(P_X \Vert P_\theta).
$$

Here:

- $P_X$ is the true data distribution;
- $P_\theta$ is the model distribution;
- $D(P_X \Vert P_\theta)$ is the divergence;
- $\theta^*$ is the optimal parameter value.

The divergence satisfies

$$
D(P_X \Vert P_\theta) \geq 0,
$$

and ideally

$$
D(P_X \Vert P_\theta) = 0
\iff
P_X = P_\theta.
$$

After solving the optimization problem, the model distribution becomes

$$
P_{\theta^*}.
$$

If training is successful, then

$$
P_{\theta^*} \approx P_X.
$$

---

# Sampling After Training

After training, generation is simple.

First sample from the known latent distribution:

$$
z \sim \mathcal{N}(0,I).
$$

Then pass this sample through the learned generator:

$$
\hat{x} = g_{\theta^*}(z).
$$

The output satisfies

$$
\hat{x} \sim P_{\theta^*}.
$$

Since training aims to make $P_{\theta^*}$ close to $P_X$, we treat $\hat{x}$ as an approximate sample from the real data distribution.

The generation process is therefore

$$
z \sim \mathcal{N}(0,I)
\quad \longrightarrow \quad
\hat{x} = g_{\theta^*}(z)
\quad \longrightarrow \quad
\hat{x} \sim P_{\theta^*} \approx P_X.
$$

The same process can be read as a simple three-step procedure:

1. Draw $z \sim \mathcal{N}(0, I)$. *(Easy - just use a random number generator.)*
2. Compute $\hat{x} = g_{\theta^*}(z)$. *(Easy - just a forward pass through the trained network.)*
3. The output $\hat{x}$ is a sample from $P_{\theta^*}$, which is close to $P_X$.

$$
\underbrace{z \sim \mathcal{N}(0, I)}_{\text{easy to sample}} \;\;\longrightarrow\;\; \underbrace{\hat{x} = g_{\theta^*}(z) \sim P_{\theta^*}}_{\text{close to } P_X}
$$

That is the whole magic, in skeleton form. Every model we will study fills in this skeleton with different details.

> [!example] Putting numbers on it
> Suppose $k = 128$ (latent dimension) and $d = 480{,}000$ (image dimension).
> - Draw a 128-number noise vector from a Gaussian. Trivial.
> - Feed it through a neural network with millions of weights.
> - Out comes a 480,000-number vector. Reshape it to $400 \times 400 \times 3$ and it looks like a coherent photograph.
>
> The network has effectively learned to **expand 128 numbers of noise into a high-resolution image** that lies in the same "cloud" as the training data.

## Diagram: Sampling from the Learned Generator



---

# Why This Works

The latent distribution $\mathcal{N}(0,I)$ is simple. By itself, it does not look like the data distribution.

The generator $g_\theta$ changes the distribution.

If $z$ has distribution $P_Z$, and

$$
\hat{x} = g_\theta(z),
$$

then the distribution of $\hat{x}$ is the pushforward of $P_Z$ through $g_\theta$.

This is written as

$$
P_\theta = (g_\theta)_\# P_Z.
$$

The symbol $(g_\theta)_\# P_Z$ means: take samples from $P_Z$, pass them through $g_\theta$, and look at the resulting distribution in data space.

The whole problem is to choose $g_\theta$ so that this induced distribution is close to $P_X$:

$$
(g_\theta)_\# P_Z \approx P_X.
$$

This gives a compact view of many generative models.

---

# Important Questions Raised by the Notes

The first part ends with four key questions.

## Question 1: How do we compute the divergence without knowing $P_X$ and $P_\theta$?

$$
D(P_X \,\|\, P_\theta) = \;?
$$

We do **not** know $P_X$ (we only have samples), and $P_\theta$ is defined implicitly through the neural network (we usually cannot write down a closed-form density for it either). So how do we compute — let alone minimize — the divergence between two distributions when neither is available in closed form?

In many models, the generated distribution $P_\theta$ is also not available as an explicit density. We can sample from it, but we may not be able to evaluate $p_\theta(x)$ exactly.

So the first problem is:

$$
\text{How can we estimate } D(P_X \Vert P_\theta)
\text{ using samples?}
$$

This question leads to different model families.

GANs estimate distributional difference using a discriminator.

VAEs optimize a tractable lower bound.

Diffusion models use denoising objectives and score matching ideas.

Auto-regressive models use likelihood factorization.

## Question 2: What should be the choice of divergence metric?

There are many divergences to choose from: KL divergence, Jensen–Shannon divergence, Wasserstein distance, $f$-divergences, integral probability metrics, and so on. Each gives different behavior — different gradients, different failure modes, different stability properties. Which one should we use, and why? Different divergences lead to different generative models.

Common choices include:

Forward KL divergence:

$$
D_{\mathrm{KL}}(P_X \Vert P_\theta),
$$

Reverse KL divergence:

$$
D_{\mathrm{KL}}(P_\theta \Vert P_X),
$$

Jensen-Shannon divergence:

$$
D_{\mathrm{JS}}(P_X \Vert P_\theta),
$$

Wasserstein distance:

$$
W(P_X, P_\theta).
$$

The choice of divergence affects:

- training stability;
- sample quality;
- mode coverage;
- whether likelihoods are tractable;
- whether samples can be generated efficiently.

## Question 3: How do we choose $g_\theta(z)$, and therefore $P_\theta$?

A neural network has many possible architectures: convolutional, transformer-based, U-Net, and others. How do we design the network so that it can flexibly produce the kinds of samples we want? Are there architectural choices that make optimization easier?

The generator architecture determines which distributions can be represented.

The choice of $g_\theta$ includes:

- the neural network architecture;
- the latent dimension $k$;
- the output dimension $d$;
- the activation functions;
- the inductive biases built into the model.

For images, convolutional or attention-based architectures may be useful. For text, transformer-based architectures are common. For sequences, recurrent, attention-based, or state-space architectures may be used. The model distribution $P_\theta$ is determined by both the latent distribution and the generator:

$$
P_\theta = (g_\theta)_\# P_Z.
$$

## Question 4: How do we solve the optimization problem?

The ideal optimization problem is

$$
\theta^* = \arg\min_{\theta} D(P_X \Vert P_\theta).
$$

In practice, we solve an empirical optimization problem using finite data.

Assuming we have picked a divergence and a parameterization, how do we actually compute $\theta^*$? With millions of parameters and a highly non-convex objective, this is not a straightforward calculus problem. Stochastic gradient descent? With what tricks? Where do the gradients even come from?

> [!important] The course roadmap
> Every subsequent part of this course will return to these four questions and answer them, one model family at a time. By the end, you will see the same four questions answered in many different ways — and the right answer will depend on what you are trying to build.

---

# Summary of the First Part

## Recap

- We are given iid samples from an **unknown** distribution $P_X$ over $\mathbb{R}^d$.
- We pick a parametric family $\{P_\theta\}$, defined implicitly by passing noise $z \sim \mathcal{N}(0, I)$ through a neural network $g_\theta$.
- We measure the gap between $P_X$ and $P_\theta$ using a divergence $D$.
- We minimize this divergence over $\theta$ to get $\theta^*$.
- After training, sampling is easy: draw $z$, push it through $g_{\theta^*}$, and you have a new sample that looks like the data.
- Making this work requires answering four big questions — and the rest of the course is the answer.

The main open questions for the next part are:

1. how to compute or estimate divergence when $P_X$ and $P_\theta$ are unknown or implicit;
2. how to choose the divergence;
3. how to choose the generator $g_\theta$;
4. how to solve the resulting optimization problem.

---

## Key Notation Reference

| Symbol | Meaning |
|---|---|
| $\mathcal{D}$ | The training dataset $\{x_1, \ldots, x_n\}$ |
| $n$ | Number of training samples |
| $d$ | Dimensionality of each sample (e.g., 480,000 for a 400×400 RGB image) |
| $x_i$ | The $i$-th observed sample, $x_i \in \mathbb{R}^d$ |
| $X$ | A random variable with distribution $P_X$ |
| $P_X$ | The true (unknown) data distribution over $\mathbb{R}^d$ |
| $P_Z$ | The latent distribution before applying $g_\theta$ |
| $P_\theta$ | The model's distribution, indexed by parameters $\theta$ |
| $p_\theta(x)$ | Density or mass function of $P_\theta$, when it exists |
| $\theta$ | Parameters of the model (weights of a neural network) |
| $\theta^*$ | The optimal parameters found by minimizing the divergence |
| $z$ | The latent variable, $z \in \mathbb{R}^k$ |
| $k$ | Dimensionality of the latent space (typically $k \ll d$) |
| $\mathcal{N}(0, I)$ | Standard multivariate Gaussian — mean 0, identity covariance |
| $g_\theta$ | The generator network, $g_\theta : \mathbb{R}^k \to \mathbb{R}^d$ |
| $\hat{x}$ | A generated sample, $\hat{x} = g_\theta(z)$ |
| $D(P \,\|\, Q)$ | A divergence between distributions $P$ and $Q$ |
| $W(P, Q)$ | Wasserstein distance between distributions $P$ and $Q$ |
| iid | Independent and identically distributed |
