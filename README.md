## EnterpriseOps Gym 2

EnterpriseOps Gym 2 is a framework for evaluating the performance of agents in enterprise operations tasks in realistic multi-turn settings. It provides a set of domains, each of which represents a specific area of enterprise operations.

Each domain specifies:

- A policy that the agent must follow
- A set of tools that the agent can use
- A set of tasks to evaluate the agent's performance



## Domains

1. ITSM: IT Service Management

## Running evals

### Setup

Requires Python 3.10+ (the code uses 3.10 syntax). Create a venv and install:

```bash
python3.13 -m venv .venv
.venv/bin/python -m pip install -e ".[dev]"
```

Put your provider credentials in a `.env` file at the repo root (it is gitignored):

```bash
OPENAI_API_KEY=sk-...
# optional, for other providers
AWS_BEARER_TOKEN_BEDROCK=...
AWS_REGION=us-east-1
FIREWORKS_AI_API_KEY=...
LITELLM_DROP_PARAMS=true
```

Models are resolved by [litellm](https://docs.litellm.ai/), so any provider string works
(`gpt-4o`, `bedrock/anthropic.claude-3-5-sonnet-20240620-v1:0`, `fireworks_ai/...`, etc.).

### The `eops` CLI

Installing the package puts an `eops` command on your path (tau2-style):

```bash
set -a && . ./.env && set +a            # load credentials into the environment

eops run    --domain itsm               # run the eval over the domain's tasks
eops tasks  --domain itsm               # list the tasks in a domain
eops domain itsm                        # print the domain policy
```

`eops run` ties together a litellm tool-calling agent, the user simulator, the environment,
and the evaluator (DB-based match + NL-assertion judge). Useful flags:

| Flag | Default | Description |
|------|---------|-------------|
| `--domain`, `-d` | `itsm` | Domain to evaluate |
| `--agent-llm` | `gpt-4o` | LLM for the agent |
| `--user-llm` | `gpt-4o-mini` | LLM for the user simulator |
| `--judge-llm` | `gpt-4o-mini` | LLM for the NL-assertion judge |
| `--task-ids` | all | Run only these task ids |
| `--num-tasks` | all | Run at most N tasks |
| `--max-steps` | `12` | Max conversation steps per task |
| `--save-to` | — | Write full results (trajectories + rewards) to a JSON file |
| `--verbose`, `-v` | off | Print each task's conversation |

Example — run one task, show the conversation, save the trajectories:

```bash
eops run --domain itsm --num-tasks 1 --verbose --save-to results.json
```

```
=== itsm_workload_rebalance_001 ===
  ...conversation...
  DB match : True
  NL [PASS] The agent confirmed the incident's priority was changed to moderate.
  NL [PASS] The agent reassigned the incident to Priya Sharma.
  reward   : 1.0  (stopped=True, tool_calls=3)

=== Summary ===
domain=itsm  agent=gpt-4o  user=gpt-4o-mini  judge=gpt-4o-mini
tasks=1  avg_reward=1.000
```

The total reward is multiplicative over the criteria a task defines, so a run passes only if
every criterion passes (DB-match catches wrong identifiers that the NL judge may overlook).

### Programmatic API

`examples/run_eval.py` does the same thing from Python via `eops_gym.run.run_task`:

```bash
set -a && . ./.env && set +a
python examples/run_eval.py             # override models with AGENT_MODEL / USER_MODEL / JUDGE_MODEL
```

### Tasks

Tasks live in `data/itsm/tasks.json`. Each task specifies a user scenario (persona +
task description), an `initial_state_delta` applied over the seed `data/itsm/db.json`
(`set` / `create` / `delete`), and evaluation criteria (gold `actions` for DB-match and
`nl_assertions` for the LLM judge). Load them in code with:

```python
from eops_gym.domains.itsm.environment import get_tasks, get_environment
task = get_tasks()[0]
env = get_environment(db_delta=task.initial_state_delta)
```

### Unit tests

The component tests run without any API key (LLM calls are mocked):

```bash
.venv/bin/python -m pytest -q
```
