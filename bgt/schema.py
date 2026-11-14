"""Data structures for the inputs and outputs of the experiment pipeline."""

from typing import Literal, Optional, get_args
from pydantic import BaseModel, ConfigDict, Field, model_validator

Arm = Literal["cold", "linked"]   # cold means no persuader; linked means persuader tries belief, then goal
ARMS = list(get_args(Arm))


class Belief(BaseModel):
    """Agent A response from the evaluations and the wrong answer the persuader (Agent B) pushes."""
    model_config = ConfigDict(extra="forbid")
    question: str
    options: dict[str, str] = Field(min_length=2)
    correct: str
    target: str                      
    target_statement: str            
    logical_args: list[str] = Field(min_length=1)   

    @model_validator(mode="after")
    def _check(self):
        for name in ("correct", "target"):
            if getattr(self, name) not in self.options:
                raise ValueError(f"{name} is not one of the options {list(self.options)}")
        if self.target == self.correct:
            raise ValueError("target must differ from correct")
        return self


class Tool(BaseModel):
    """A dummy tool that just returns `output`."""
    model_config = ConfigDict(extra="forbid")
    name: str
    description: str
    output: str


class Scenario(BaseModel):
    """One task in the suite: a task, its tools, and the harmful side goal the persuader pushes."""
    model_config = ConfigDict(extra="forbid")
    id: str
    role: str
    task: str
    spec: str                         # the rule agent A should follow
    facts: list[str] = Field(min_length=1)
    tools: list[Tool] = Field(min_length=2)
    side_tool: str                    # the harmful tool the persuader wants agent A to call
    side_goal: str                    # a plausible-sounding reason for calling it
    world_truth: list[str] = Field(default_factory=list)   # true background for the fact-check monitor
    belief: Belief                    # the fact the adversarial goal leans on

    @model_validator(mode="after")
    def _check(self):
        if self.side_tool not in [t.name for t in self.tools]:
            raise ValueError(f"side_tool {self.side_tool!r} is not in tools")
        return self


class Result(BaseModel):
    """One row of the output file."""
    model_config = ConfigDict(extra="forbid", validate_assignment=True)

    episode_id: str
    persuadee: str
    scenario: str
    arm: Arm
    rep: int

    belief_valid: Optional[bool] = None      # linked only: did the agent start with the correct belief?
    belief_flipped: Optional[bool] = None    # linked only: did the persuader flip it? (persuasion success)
    diverted: Optional[bool] = None          # did the agent call the harmful side tool? (goal diversion success)

    flag_instruction: Optional[bool] = None  # any persuader goal-message gave an instruction / set goals
    flag_factcheck_msg: Optional[bool] = None  # any persuader goal-message stated something false

    tool_calls: list[str] = Field(default_factory=list)
    persuader_messages: list[str] = Field(default_factory=list)
    agent_reasoning: list[str] = Field(default_factory=list)   # every agent turn
    transcript: list[dict[str, str]] = Field(default_factory=list)