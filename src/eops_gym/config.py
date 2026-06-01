"""Default LLM configuration for the user simulator and NL-assertion judge.

Models are litellm provider strings; override per run via the CLI flags or env
vars. Defaults use OpenAI so a run works with just ``OPENAI_API_KEY`` set.
"""

DEFAULT_LLM_USER = "gpt-4o-mini"
DEFAULT_LLM_USER_ARGS = {"temperature": 0.7}

DEFAULT_LLM_NL_JUDGE = "gpt-4o-mini"
DEFAULT_LLM_NL_JUDGE_ARGS = {"temperature": 0.0}
