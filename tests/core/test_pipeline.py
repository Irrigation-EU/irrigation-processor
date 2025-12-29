import unittest
from unittest.mock import Mock

from irrigation_processor.core import Pipeline, FromStep


class DummyStepMeta:
    def __init__(self, name, depends_on=None, inputs=None):
        self.name = name
        self.depends_on = depends_on or []
        self.inputs = inputs or []


class TestPipeline(unittest.TestCase):

    def setUp(self):
        self.service = Mock()
        self.pipeline = Pipeline(service=self.service, pipeline_name="test_pipeline")


    def test_add_step_success(self):
        step = DummyStepMeta("step1")

        self.pipeline.add(step)

        self.assertIn("step1", self.pipeline.steps)
        self.assertIs(self.pipeline.steps["step1"], step)

    def test_add_duplicate_step_raises_key_error(self):
        step = DummyStepMeta("step1")
        self.pipeline.add(step)

        with self.assertRaises(KeyError):
            self.pipeline.add(step)

    def test_add_steps_from_registry(self):
        registry = Mock()
        step1 = DummyStepMeta("step1")
        step2 = DummyStepMeta("step2")
        registry.all.return_value = [step1, step2]

        self.pipeline.add_steps_from_registry(registry)

        self.assertEqual(set(self.pipeline.steps.keys()), {"step1", "step2"})


    def test_build_graph_with_depends_on(self):
        step1 = DummyStepMeta("step1")
        step2 = DummyStepMeta("step2", depends_on=["step1"])

        self.pipeline.add(step1)
        self.pipeline.add(step2)

        graph = self.pipeline._build_graph()

        self.assertEqual(graph, {"step1": [], "step2": ["step1"]})

    def test_build_graph_with_fromstep_input(self):
        step1 = DummyStepMeta("step1")
        step2 = DummyStepMeta(
            "step2",
            inputs=[FromStep("step1", "value1")]
        )

        self.pipeline.add(step1)
        self.pipeline.add(step2)

        graph = self.pipeline._build_graph()

        self.assertEqual(graph, {"step1": [], "step2": ["step1"]})

    def test_build_graph_unknown_dependency_raises(self):
        step = DummyStepMeta("step1", depends_on=["missing"])

        self.pipeline.add(step)

        with self.assertRaises(ValueError) as ctx:
            self.pipeline._build_graph()

        self.assertIn("depends on unknown step", str(ctx.exception))

    def test_toposort_simple_linear(self):
        deps = {
            "a": [],
            "b": ["a"],
            "c": ["b"],
        }

        order = Pipeline._toposort(deps)

        self.assertEqual(order, ["a", "b", "c"])

    def test_toposort_branching(self):
        deps = {
            "a": [],
            "b": ["a"],
            "c": ["a"],
        }

        order = Pipeline._toposort(deps)

        self.assertEqual(order[0], "a")
        self.assertCountEqual(order[1:], ["b", "c"])

    def test_toposort_cycle_raises(self):
        deps = {
            "a": ["b"],
            "b": ["a"],
        }

        with self.assertRaises(RuntimeError) as ctx:
            Pipeline._toposort(deps)

        self.assertIn("Cycle detected", str(ctx.exception))


    def test_visualize_dot_output(self):
        step1 = DummyStepMeta("step1")
        step2 = DummyStepMeta("step2", depends_on=["step1"])

        self.pipeline.add(step1)
        self.pipeline.add(step2)

        dot = self.pipeline.visualize_dot()

        self.assertIn('digraph pipeline', dot)
        self.assertIn('"step1";', dot)
        self.assertIn('"step2";', dot)
        self.assertIn('"step1" -> "step2";', dot)


    def test_run_no_steps_returns_none(self):
        result = self.pipeline.run()

        self.assertIsNone(result)
        self.service.run.assert_not_called()

    def test_run_executes_service(self):
        step1 = DummyStepMeta("step1")
        step2 = DummyStepMeta("step2", depends_on=["step1"])

        self.pipeline.add(step1)
        self.pipeline.add(step2)

        self.service.run.return_value = "RESULT"

        result = self.pipeline.run()

        self.assertEqual(result, "RESULT")
        self.service.run.assert_called_once()

        args = self.service.run.call_args[0]
        self.assertEqual(args[0], "test_pipeline")
        self.assertEqual(args[1], ["step1", "step2"])
        self.assertIs(args[2], self.pipeline.steps)
