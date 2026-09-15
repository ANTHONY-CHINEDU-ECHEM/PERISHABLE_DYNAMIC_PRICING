# Extending this reference implementation

1. **PPO with multi-objective reward.** Replace `agents/q_pricing_agent.py`
   with a PPO agent that can learn a smoother waste/margin tradeoff curve
   than tabular Q-learning's coarse discretization allows.
2. **Doubly-robust off-policy evaluation.** `evaluation/offline_policy_eval.py`
   currently does a direct Monte-Carlo comparison in the simulator; a real
   deployment needs a proper importance-sampling / doubly-robust estimator
   against **logged historical data**, not just simulator replay.
3. **Real elasticity data.** Replace the log-log model in
   `elasticity/demand_model.py` with one fit on real POS transaction data,
   and add competitor-price and weather signals as covariates.
4. **Live competitor pricing feed.** Add a scraped/API competitor-price
   signal as an additional state feature.
