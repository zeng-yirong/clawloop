import os
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import torch

from verl.experimental.agent_loop.nanoclaw_support import (
    nanoclaw_apply_posthoc_turn_masks,
    nanoclaw_tool_result_is_error,
)
from verl.trainer.ppo.ray_trainer import apply_nanoclaw_positive_advantage_bad_turn_mask


def test_duplicate_tool_result_turn_is_recorded_for_delayed_positive_advantage_mask():
    agent_data = SimpleNamespace(
        termination_reason="completed_no_tool_call",
        response_mask=[1, 1, 0, 1, 1],
        metrics={},
        extra_fields={},
        trajectory_events=[
            {"type": "assistant", "assistant_turn": 1, "response_start": 0, "response_end": 2},
            {
                "type": "tool",
                "assistant_turn": 1,
                "tool": "read_file",
                "arguments": '{"path":"a.txt"}',
                "response": {"role": "tool", "content": "same"},
                "result": {},
            },
            {"type": "assistant", "assistant_turn": 2, "response_start": 3, "response_end": 5},
            {
                "type": "tool",
                "assistant_turn": 2,
                "tool": "read_file",
                "arguments": '{"path":"a.txt"}',
                "response": {"role": "tool", "content": "same"},
                "result": {},
            },
        ],
    )

    with patch.dict(
        os.environ,
        {
            "NANOCLAW_MASK_DUPLICATE_TOOL_RESULT_TURNS": "True",
            "NANOCLAW_MASK_ONLY_POSITIVE_ADVANTAGE": "True",
        },
    ):
        nanoclaw_apply_posthoc_turn_masks(agent_data)

    assert agent_data.response_mask == [1, 1, 0, 1, 1]
    assert agent_data.extra_fields["nanoclaw_bad_turn_spans"] == [
        {
            "assistant_turn": 2,
            "reason": "duplicate_tool_result_turn",
            "start": 3,
            "end": 5,
            "token_count": 2,
            "mask_enabled": True,
        }
    ]


def test_delayed_mask_removes_only_positive_advantage_tokens():
    bad_turn_spans = np.empty(1, dtype=object)
    bad_turn_spans[0] = [
        {"assistant_turn": 2, "reason": "looping_response", "start": 0, "end": 5}
    ]
    batch = SimpleNamespace(
        batch={
            "response_mask": torch.tensor([[1, 1, 0, 1, 1]], dtype=torch.long),
            "advantages": torch.tensor([[-1.0, 0.5, 0.0, 2.0, -0.5]]),
        },
        non_tensor_batch={
            "nanoclaw_bad_turn_spans": bad_turn_spans,
        },
    )

    with patch.dict(os.environ, {"NANOCLAW_MASK_ONLY_POSITIVE_ADVANTAGE": "True"}):
        metrics = apply_nanoclaw_positive_advantage_bad_turn_mask(batch)

    assert batch.batch["response_mask"].tolist() == [[1, 0, 0, 0, 1]]
    assert metrics["nanoclaw/positive_advantage_bad_turn_candidate_turns"] == 1.0
    assert metrics["nanoclaw/positive_advantage_bad_turn_candidate_tokens"] == 4.0
    assert metrics["nanoclaw/positive_advantage_bad_turn_masked_turns"] == 1.0
    assert metrics["nanoclaw/positive_advantage_bad_turn_masked_tokens"] == 2.0


def test_error_tool_result_is_recorded_even_when_its_mask_is_disabled():
    agent_data = SimpleNamespace(
        termination_reason="completed_no_tool_call",
        response_mask=[1, 1, 1],
        metrics={},
        extra_fields={},
        trajectory_events=[
            {"type": "assistant", "assistant_turn": 1, "response_start": 0, "response_end": 3},
            {
                "type": "tool",
                "assistant_turn": 1,
                "tool": "read_file",
                "arguments": '{"path":"missing.txt"}',
                "response": {"role": "tool", "content": "Error: file does not exist: missing.txt"},
                "result": {},
            },
        ],
    )

    with patch.dict(
        os.environ,
        {
            "NANOCLAW_MASK_ERROR_TOOL_RESULT_TURNS": "False",
            "NANOCLAW_MASK_ONLY_POSITIVE_ADVANTAGE": "True",
            "NANOCLAW_MASK_DUPLICATE_TOOL_RESULT_TURNS": "False",
            "NANOCLAW_MASK_BUDGET_EXHAUSTED_LAST_TURN": "False",
        },
    ):
        nanoclaw_apply_posthoc_turn_masks(agent_data)

    assert agent_data.response_mask == [1, 1, 1]
    assert agent_data.trajectory_events[1]["error_tool_result"] is True
    assert agent_data.extra_fields["nanoclaw_bad_turn_spans"] == [
        {
            "assistant_turn": 1,
            "reason": "error_tool_result_turn",
            "start": 0,
            "end": 3,
            "token_count": 3,
            "mask_enabled": False,
        }
    ]


def test_structured_failed_tool_result_is_detected_as_error():
    assert nanoclaw_tool_result_is_error(
        {
            "response": {"role": "tool", "content": "operation finished"},
            "result": {"status": "failed", "error": {"message": "permission denied"}},
        }
    )
    assert not nanoclaw_tool_result_is_error(
        {
            "response": {"role": "tool", "content": "Success: operation finished"},
            "result": {"status": "ok"},
        }
    )


def test_error_tool_result_positive_advantage_mask_can_be_enabled():
    agent_data = SimpleNamespace(
        termination_reason="completed_no_tool_call",
        response_mask=[1, 1, 1],
        metrics={},
        extra_fields={},
        trajectory_events=[
            {"type": "assistant", "assistant_turn": 1, "response_start": 0, "response_end": 3},
            {
                "type": "tool",
                "assistant_turn": 1,
                "tool": "bash",
                "arguments": '{"command":"false"}',
                "response": {"role": "tool", "content": "Error: bash exited with code 1"},
                "result": {},
            },
        ],
    )

    with patch.dict(
        os.environ,
        {
            "NANOCLAW_MASK_ERROR_TOOL_RESULT_TURNS": "True",
            "NANOCLAW_MASK_ONLY_POSITIVE_ADVANTAGE": "True",
            "NANOCLAW_MASK_DUPLICATE_TOOL_RESULT_TURNS": "False",
            "NANOCLAW_MASK_BUDGET_EXHAUSTED_LAST_TURN": "False",
        },
    ):
        nanoclaw_apply_posthoc_turn_masks(agent_data)

    spans = np.empty(1, dtype=object)
    spans[0] = agent_data.extra_fields["nanoclaw_bad_turn_spans"]
    batch = SimpleNamespace(
        batch={
            "response_mask": torch.tensor([[1, 1, 1]], dtype=torch.long),
            "advantages": torch.tensor([[-1.0, 0.5, 2.0]]),
        },
        non_tensor_batch={"nanoclaw_bad_turn_spans": spans},
    )
    with patch.dict(os.environ, {"NANOCLAW_MASK_ONLY_POSITIVE_ADVANTAGE": "True"}):
        metrics = apply_nanoclaw_positive_advantage_bad_turn_mask(batch)

    assert batch.batch["response_mask"].tolist() == [[1, 0, 0]]
    assert metrics == {
        "nanoclaw/positive_advantage_bad_turn_candidate_turns": 1.0,
        "nanoclaw/positive_advantage_bad_turn_candidate_tokens": 3.0,
        "nanoclaw/positive_advantage_bad_turn_masked_turns": 1.0,
        "nanoclaw/positive_advantage_bad_turn_masked_tokens": 2.0,
    }


def test_positive_advantage_bad_turn_metrics_are_reported_when_mask_is_off():
    bad_turn_spans = np.empty(1, dtype=object)
    bad_turn_spans[0] = [
        {
            "assistant_turn": 1,
            "reason": "error_tool_result_turn",
            "start": 0,
            "end": 3,
            "token_count": 3,
            "mask_enabled": False,
        }
    ]
    batch = SimpleNamespace(
        batch={
            "response_mask": torch.tensor([[1, 1, 1]], dtype=torch.long),
            "advantages": torch.tensor([[-1.0, 0.5, 2.0]]),
        },
        non_tensor_batch={"nanoclaw_bad_turn_spans": bad_turn_spans},
    )

    with patch.dict(os.environ, {"NANOCLAW_MASK_ONLY_POSITIVE_ADVANTAGE": "False"}):
        metrics = apply_nanoclaw_positive_advantage_bad_turn_mask(batch)

    assert batch.batch["response_mask"].tolist() == [[1, 1, 1]]
    assert metrics == {
        "nanoclaw/positive_advantage_bad_turn_candidate_turns": 1.0,
        "nanoclaw/positive_advantage_bad_turn_candidate_tokens": 3.0,
        "nanoclaw/positive_advantage_bad_turn_masked_turns": 0.0,
        "nanoclaw/positive_advantage_bad_turn_masked_tokens": 0.0,
    }


def test_positive_advantage_bad_turn_metrics_report_zeros_without_candidates():
    bad_turn_spans = np.empty(1, dtype=object)
    bad_turn_spans[0] = []
    batch = SimpleNamespace(
        batch={
            "response_mask": torch.tensor([[1, 1]], dtype=torch.long),
            "advantages": torch.tensor([[1.0, -1.0]]),
        },
        non_tensor_batch={"nanoclaw_bad_turn_spans": bad_turn_spans},
    )

    with patch.dict(os.environ, {"NANOCLAW_MASK_ONLY_POSITIVE_ADVANTAGE": "False"}):
        metrics = apply_nanoclaw_positive_advantage_bad_turn_mask(batch)

    assert metrics == {
        "nanoclaw/positive_advantage_bad_turn_candidate_turns": 0.0,
        "nanoclaw/positive_advantage_bad_turn_candidate_tokens": 0.0,
        "nanoclaw/positive_advantage_bad_turn_masked_turns": 0.0,
        "nanoclaw/positive_advantage_bad_turn_masked_tokens": 0.0,
    }
