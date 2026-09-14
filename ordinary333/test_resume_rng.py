"""Static-source regression test: exercise RNG restore without torch, GPU, or training."""
import ast
from pathlib import Path
from types import SimpleNamespace
import unittest


SOURCE = Path(__file__).resolve().with_name("training.py")


class FakeState:
    """A minimal tensor double whose .cpu() returns a distinct host state."""
    def __init__(self):
        self.host_state = object()
        self.cpu_calls = 0

    def cpu(self):
        self.cpu_calls += 1
        return self.host_state


def rng_restore_statement():
    """Extract only the guarded CUDA RNG restore from the real training source."""
    tree = ast.parse(SOURCE.read_text(encoding="utf-8"))
    for node in ast.walk(tree):
        if isinstance(node, ast.If) and any(
            isinstance(child, ast.Call)
            and isinstance(child.func, ast.Attribute)
            and child.func.attr == "set_rng_state_all"
            for statement in node.body for child in ast.walk(statement)
        ):
            # Choose the innermost single-statement restore, not the outer resume block.
            if len(node.body) == 1 and isinstance(node.body[0], ast.Expr):
                return compile(ast.fix_missing_locations(ast.Module(body=[node], type_ignores=[])),
                               str(SOURCE), "exec")
    raise ValueError("Guarded CUDA RNG restore was not found in the training source")


class ResumeRngTests(unittest.TestCase):
    def restore(self, saved, device="cuda"):
        received = []
        namespace = dict(saved=saved, device=SimpleNamespace(type=device),
                         torch=SimpleNamespace(cuda=SimpleNamespace(set_rng_state_all=received.append)))
        exec(rng_restore_statement(), namespace)
        return received

    def test_cuda_resume_passes_only_host_states(self):
        states = [FakeState(), FakeState()]
        received = self.restore({"cuda_rng": states})
        self.assertEqual(received, [[state.host_state for state in states]])
        self.assertEqual([state.cpu_calls for state in states], [1, 1])

    def test_legacy_checkpoint_without_cuda_rng_is_accepted(self):
        self.assertEqual(self.restore({}), [])

    def test_cpu_resume_does_not_touch_cuda_rng(self):
        state = FakeState()
        self.assertEqual(self.restore({"cuda_rng": [state]}, device="cpu"), [])
        self.assertEqual(state.cpu_calls, 0)


if __name__ == "__main__":
    unittest.main()


