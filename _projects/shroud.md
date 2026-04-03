---
layout: page
title: Shroud
description: Agent Identity API — disposable digital identities for AI agents
img: assets/img/shroud.png
importance: 2
category: work
---

[Shroud](https://shroud.co.in) is an Agent Identity API that provides autonomous AI systems with disposable digital identities — ephemeral emails, temporary phone numbers, and virtual cards — so agents can operate independently from their creator's personal credentials.

## The Problem

When building autonomous AI agents (LangChain, AutoGen, CrewAI), agents often need to sign up for services, receive OTPs, or make payments. Using personal email addresses or phone numbers creates privacy risks, audit gaps, and operational coupling.

## What Shroud Does

- **Ephemeral Email Addresses** — agent-scoped inboxes (e.g., `agent_77@shroud.ai`) with a built-in Mail Extraction layer that parses OTPs and magic links, achieving 98.2% context efficiency for downstream LLM consumption
- **Temporary Phone Numbers** — carrier-grade numbers without physical SIMs, for SMS-based verification flows
- **Virtual Cards** — configurable spending limits ($50 default) with real-time anomaly detection on spending patterns
- **Full Audit Trails** — immutable logs for every action taken under an agent identity, designed for compliance

## Technical Stack

- **Frontend**: Next.js + React + TypeScript
- **SDKs**: TypeScript SDK with first-class integrations for LangChain, AutoGen, and CrewAI
- **Security**: AES-256-GCM encryption, SOC 2 Type II certified, KYC via Stripe Identity

## Key Design Decisions

Agent identities are scoped to tasks and automatically cleaned up after completion, keeping no persistent state. This makes Shroud suitable for multi-agent pipelines where each sub-agent needs isolated credentials without manual provisioning.

## Try It Out

Shroud is currently in public beta. Get started at [shroud.co.in](https://shroud.co.in) — the Starter tier offers 1,000 monthly profile creates for free.
