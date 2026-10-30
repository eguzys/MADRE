# Evaluate a MANET Agent

The `evaluate_MANETAgents.py` script provides functionality for evaluating and selecting trained MANET agents. It supports grouped evaluation, baseline comparisons, and filtering of best-performing models or agents.

__Features__

1. **Pattern-Based Agent Selection**:
    - Accepts glob patterns (e.g., `"out/models/*/"` or `"out/models/*/rl_model_*_steps.zip"`) to locate agent models.
    - If a folder is matched, the model with the highest number of steps is selected automatically.

2. **Evaluation Modes**:
    - The evaluation is controlled using the `eval_type` parameter:
        - `grouped_eval`: Standard evaluation. Agents are grouped by number of nodes, users, and jammers. If baselines are provided, they are evaluated alongside each group.
        - `filter_models`: For each agent folder, selects the best model checkpoint by evaluating all available models. The selected model is copied to a `_filtered/` directory.
        - `filter_best_agent`: Evaluates multiple agents (each with their best or last checkpoint) and selects the highest-performing agent overall. The selected agent is copied to the specified `log_dir`.

3. **Configuration Parameters**:
    - Evaluation is configured via `eval_params` (either inline or via a YAML config file). Key parameters include:
    - `n_eval_eps`: Number of episodes
    - `ep_length`: Episode length
    - `obs_type`: Observation type (`"full"`, `"random"`, `"real"`)
    - `state_loss_rate`: Loss rate for observations
    - `metric_param`: Weight for pre/post-jammer throughput when filtering
    - `filterval`: Step divisor used when filtering models

4. **Example Usage**:
    - Evaluate from Python:
    ```python
    from evaluate_MANETAgents import evaluate
    import MANET.Environments as envs

    evaluate(
        env_constr=envs.Static_MANETEnv,
        agent_patterns=["path/to/models/*"],
        eval_params={...},
        eval_type="grouped_eval",
        log_dir="out/evaluations",
        log_results=True,
        log_eval=True,
        log_state=True,
    )
    ```

::: evaluate_MANETAgents
    options:
      show_root_heading: false  # Avoid showing the module name as a header
