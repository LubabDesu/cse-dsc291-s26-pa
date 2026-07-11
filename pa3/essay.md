# The Inference Turn: Why Post-Training and Inference-Time Compute Will Dominate Frontier Lab Spend by Q4 2027

For most of the deep learning era, progress in large language models followed a single playbook: scale pretraining. More tokens, more parameters, more compute directed at next-token prediction on internet-scale corpora. That playbook is however, no longer the primary one. By Q4 2027, I believe that at least 80% of annual compute spend at frontier AI labs combined will be allocated to post-training (reinforcement learning, RLVR, SFT) and inference-time scaling (reasoning chains, search, agent loops), rather than to pretraining. The structural forces driving this shift are already visible in current financial disclosures, and they are compounding.

The baseline makes the trajectory concrete. Epoch AI's analysis of OpenAI's 2024 cloud compute expenses found inference at roughly $2 billion against $5 billion in R&D compute — approximately 29% of total spend (Epoch AI, October 2025). By 2025, inference costs had grown fourfold to $8.4 billion, compressing adjusted gross margin from 40% to 33% (Reuters / The Information, 2025; Sacra, 2026). R&D compute grew over the same period, but not at that rate — inference is now the faster-growing line by a wide margin.

Two structural forces explain why this accelerates toward 80% by late 2027. The first is reasoning models, which have made inference compute scale with task difficulty rather than remaining fixed per query. DeepSeek-R1-0528 nearly doubled per-query token usage relative to its predecessor, from roughly 12,000 to 23,000 tokens per AIME question (BentoML, 2025). GPT-5.5, released in April 2026, allocates reasoning compute dynamically per agentic task (OpenAI, 2026), and Gemini 3.5 Flash ships with dynamic thinking enabled by default — inference-time compute is no longer an opt-in mode but the baseline (Google DeepMind, 2026). Each successive model generation expands the per-query compute envelope.

The second force is how frontier labs allocate R&D compute. Epoch AI's October 2025 analysis found that the majority of OpenAI's $5 billion 2024 R&D compute went to experiments rather than final training runs. Their March 2026 analysis of MiniMax and Z.ai IPO filings revealed the same pattern: the majority of compute consumed by post-training pipelines and RL iterations, not pretraining (Epoch AI, March 2026). Pretraining is a periodic capital event; marginal gains now come from post-training.

The serious counterargument is that 80% is an aggressive threshold. Epoch AI's 2024 equilibrium analysis argues labs should optimally spend roughly comparably on training and inference — a roughly 50/50 split. Under this model, a lab that underinvests in pretraining leaves capability gains on the table, suggesting an extreme inference-heavy ratio is inefficient and unlikely to persist.

This objection misidentifies who controls the spending ratio. The equilibrium model treats training and inference allocations as variables labs freely optimize for capability. In practice, inference demand is exogenous: driven by continuous end-user queries and agentic task loops, not lab decisions. As agentic deployments scale, inference becomes an operating cost that grows with revenue regardless of what is theoretically optimal. Consequently, the spending ratio is set by user demand, and that demand is structurally inference-heavy.

By Q4 2027, public financial disclosures or credible third-party analyses will show pretraining below 20% of combined annual compute spend at frontier labs, with post-training and inference-time scaling exceeding 80%. The structural incentives of the agent economy ensure it does not reverse.

---

**Sources:**

- Epoch AI, "Most of OpenAI's 2024 compute went to experiments," Oct 2025. https://epoch.ai/data-insights/openai-compute-spend
- Epoch AI, "Final training runs account for a minority of R&D compute spending," Mar 2026. https://epoch.ai/gradient-updates/r-and-d-vs-training-compute
- Epoch AI, "Optimally allocating compute between inference and training," Mar 2024. https://epoch.ai/blog/optimally-allocating-compute-between-inference-and-training
- Reuters / The Information, "OpenAI inference costs increased fourfold in 2025." https://finance.yahoo.com/news/openai-sees-compute-spend-around-223950561.html
- Sacra, "OpenAI," Apr 2026. https://sacra.com/research/openai/
- DeepSeek-AI, "DeepSeek-R1: Incentivizing Reasoning Capability in LLMs via RL," Jan 2025.
- BentoML, "Complete Guide to DeepSeek Models," 2025. https://www.bentoml.com/blog/the-complete-guide-to-deepseek-models-from-v3-to-r1-and-beyond
- OpenAI, "Introducing GPT-5.5," Apr 2026. https://openai.com/index/introducing-gpt-5-5/
- Google DeepMind, "Gemini 3.5: frontier intelligence with action," May 2026. https://blog.google/innovation-and-ai/models-and-research/gemini-models/gemini-3-5/
