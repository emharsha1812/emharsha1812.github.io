---
layout: page
title: fun
permalink: /fun/
description: Movies, songs, books, and bucket-list adventures
nav: false
nav_order: 7
---

<div class="fun-page">
  <style>
    .fun-page {
      --fun-bg-1: #fff8ef;
      --fun-bg-2: #f3f7ff;
      --fun-card: #ffffff;
      --fun-text: #222831;
      --fun-accent: #ff6b35;
      --fun-accent-2: #006d77;
      color: var(--fun-text);
    }

    .fun-hero {
      background: linear-gradient(130deg, var(--fun-bg-1), var(--fun-bg-2));
      border: 2px solid #eceff3;
      border-radius: 20px;
      padding: 1.2rem 1.4rem;
      box-shadow: 0 10px 24px rgba(26, 40, 58, 0.08);
      margin-bottom: 1rem;
    }

    .fun-kicker {
      margin: 0 0 0.4rem;
      font-size: 0.85rem;
      text-transform: uppercase;
      letter-spacing: 0.08em;
      color: var(--fun-accent-2);
      font-weight: 700;
    }

    .fun-hero p {
      margin: 0;
    }

    .fun-grid {
      display: grid;
      gap: 1rem;
      grid-template-columns: repeat(auto-fit, minmax(220px, 1fr));
      margin-top: 1rem;
    }

    .fun-card {
      background: var(--fun-card);
      border: 1px solid #e8edf3;
      border-radius: 16px;
      padding: 1rem 1rem 0.85rem;
      box-shadow: 0 8px 20px rgba(26, 40, 58, 0.06);
      transition: transform 0.2s ease, box-shadow 0.2s ease;
    }

    .fun-card:hover {
      transform: translateY(-3px);
      box-shadow: 0 14px 28px rgba(26, 40, 58, 0.1);
    }

    .fun-card h2 {
      margin-top: 0;
      font-size: 1.05rem;
      color: var(--fun-accent-2);
    }

    .fun-list {
      margin: 0;
      padding-left: 1.1rem;
    }

    .fun-list li {
      margin-bottom: 0.45rem;
    }

    .bucket-list {
      list-style: none;
      padding-left: 0;
      margin: 0;
    }

    .bucket-list li {
      margin-bottom: 0.5rem;
      padding-left: 0.2rem;
    }

    .tick {
      color: var(--fun-accent);
      font-weight: 700;
      margin-right: 0.4rem;
    }

    .fun-note {
      margin-top: 1rem;
      border-left: 4px solid var(--fun-accent);
      background: #fff;
      padding: 0.75rem 0.9rem;
      border-radius: 10px;
    }
  </style>

  <div class="fun-hero">
    <p class="fun-kicker">Off-screen mode</p>
    <p>This is my non-technical corner with a few personal picks.</p>
  </div>

  <div class="fun-grid">
    <section class="fun-card">
      <h2>Movies</h2>
      <ul class="fun-list">
        <li><strong>Bramayugam</strong></li>
        <li><strong>Tumbbad</strong></li>
        <li><strong>The Witch</strong></li>
        <li><strong>Interstellar</strong></li>
        <li><strong>The Pursuit of Happyness</strong></li>
      </ul>
    </section>

    <section class="fun-card">
      <h2>Books (Non-Technical)</h2>
      <ul class="fun-list">
        <li><strong>The Alchemist</strong> by Paulo Coelho</li>
        <li><strong>Atomic Habits</strong> by James Clear</li>
        <li><strong>Man's Search for Meaning</strong> by Viktor E. Frankl</li>
        <li><strong>The Psychology of Money</strong> by Morgan Housel</li>
        <li><strong>The Kite Runner</strong> by Khaled Hosseini</li>
      </ul>
    </section>

    <section class="fun-card">
      <h2>Bucket List</h2>
      <ul class="bucket-list">
        <li><span class="tick">✓</span>Watch the Northern Lights once.</li>
        <li><span class="tick">✓</span>Do a solo trip to a new country.</li>
        <li><span class="tick">✓</span>Learn to play one complete song on guitar.</li>
        <li><span class="tick">✓</span>Run a half marathon.</li>
        <li><span class="tick">✓</span>Build a small personal library of favorite books.</li>
        <li><span class="tick">✓</span>Watch a film at an international film festival.</li>
      </ul>
    </section>
  </div>

  <p class="fun-note"><strong>Open to recommendations:</strong> If you have a movie or book suggestion, I am always happy to explore it.</p>
</div>
