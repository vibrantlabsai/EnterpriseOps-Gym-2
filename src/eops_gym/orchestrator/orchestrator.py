"""Minimal run loop: agent <-> user simulator <-> environment.

Trimmed mirror of tau2's orchestrator. Enough to drive one task end-to-end and
produce a trajectory (plus the agent's tool calls) for the evaluator. The agent
is any object exposing ``generate_next_message(message, state) -> (AssistantMessage, state)``
and ``get_init_state()`` — a real litellm agent is item 1's concern.
"""

from typing import Protocol

from eops_gym.data_model.message import AssistantMessage, Message, ToolCall
from eops_gym.environment.environment import Environment
from eops_gym.user.user_simulator import UserSimulator


class Agent(Protocol):
    def get_init_state(self): ...
    def generate_next_message(self, message, state) -> tuple[AssistantMessage, object]: ...


class RunResult:
    def __init__(self, trajectory: list[Message], agent_tool_calls: list[ToolCall], stopped: bool):
        self.trajectory = trajectory
        self.agent_tool_calls = agent_tool_calls
        self.stopped = stopped


class Orchestrator:
    def __init__(
        self,
        agent: Agent,
        user: UserSimulator,
        environment: Environment,
        max_steps: int = 20,
    ):
        self.agent = agent
        self.user = user
        self.environment = environment
        self.max_steps = max_steps

    def run(self) -> RunResult:
        trajectory: list[Message] = []
        agent_tool_calls: list[ToolCall] = []

        agent_state = self.agent.get_init_state()
        user_state = self.user.get_init_state()

        # The user opens the conversation.
        user_msg, user_state = self.user.generate_next_message(None, user_state)
        trajectory.append(user_msg)

        stopped = self.user.is_stop(user_msg)
        last_to_agent: Message = user_msg
        steps = 0

        while not stopped and steps < self.max_steps:
            steps += 1
            agent_msg, agent_state = self.agent.generate_next_message(last_to_agent, agent_state)
            trajectory.append(agent_msg)

            if agent_msg.is_tool_call():
                for tc in agent_msg.tool_calls or []:
                    agent_tool_calls.append(tc)
                    tool_msg = self.environment.get_response(tc)
                    trajectory.append(tool_msg)
                last_to_agent = trajectory[-1]
                continue  # let the agent observe tool results before the user replies

            user_msg, user_state = self.user.generate_next_message(agent_msg, user_state)
            trajectory.append(user_msg)
            last_to_agent = user_msg
            stopped = self.user.is_stop(user_msg)

        return RunResult(trajectory=trajectory, agent_tool_calls=agent_tool_calls, stopped=stopped)
