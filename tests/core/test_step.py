import unittest

from pydantic import BaseModel

from irrigation_processor.core.pipeline import (
    StepRegistry,
    StepMeta,
    FromStep,
)


def dummy_func():
    return None


def another_func():
    return None


class TestStepRegistry(unittest.TestCase):
    def setUp(self):
        self.registry = StepRegistry()

    def test_register_and_get_step(self):
        meta = StepMeta(func=dummy_func)

        self.registry.register(meta)

        retrieved = self.registry.get(meta.name)
        self.assertIs(retrieved, meta)

    def test_register_duplicate_name_raises(self):
        meta1 = StepMeta(func=dummy_func)
        meta2 = StepMeta(func=dummy_func)

        self.registry.register(meta1)

        with self.assertRaises(KeyError) as ctx:
            self.registry.register(meta2)

        self.assertIn("already registered", str(ctx.exception))

    def test_get_unknown_step_raises(self):
        with self.assertRaises(KeyError) as ctx:
            self.registry.get("missing")

        self.assertIn("No step named", str(ctx.exception))

    def test_disable_and_enable_step(self):
        meta = StepMeta(func=dummy_func)
        self.registry.register(meta)

        self.registry.disable(meta.name)
        self.assertEqual(self.registry.all(), [])

        self.registry.enable(meta.name)
        self.assertEqual(self.registry.all(), [meta])

    def test_disable_unknown_step_raises(self):
        with self.assertRaises(KeyError):
            self.registry.disable("missing")

    def test_enable_unknown_step_raises(self):
        with self.assertRaises(KeyError):
            self.registry.enable("missing")

    def test_all_include_disabled(self):
        meta = StepMeta(func=dummy_func)
        self.registry.register(meta)

        self.registry.disable(meta.name)

        all_steps = self.registry.all(include_disabled=True)
        self.assertEqual(all_steps, [meta])

    def test_step_decorator_without_arguments(self):
        @self.registry.step
        def step_func():
            return 1

        meta = self.registry.get("step_func")

        self.assertEqual(meta.func, step_func)
        self.assertEqual(meta.name, "step_func")

    def test_step_decorator_with_arguments(self):
        @self.registry.step(
            inputs=("a",),
            outputs=("b",),
            depends_on=("x",),
            name="custom_name",
            context_cls=BaseModel,
        )
        def decorated():
            return 2

        meta = self.registry.get("custom_name")

        self.assertEqual(meta.func, decorated)
        self.assertEqual(meta.inputs, ("a",))
        self.assertEqual(meta.outputs, ("b",))
        self.assertEqual(meta.depends_on, ("x",))
        self.assertEqual(meta.context_cls, BaseModel)

    def test_fromstep_to_dict(self):
        fs = FromStep(step="step1", key="out")

        self.assertEqual(
            fs.to_dict(),
            {"step": "step1", "key": "out"},
        )

    def test_stepmeta_attributes_and_defaults(self):
        meta = StepMeta(
            func=dummy_func,
            inputs=("x",),
            outputs=("y",),
            depends_on=("z",),
            name="my_step",
            context_cls=BaseModel,
        )

        self.assertEqual(meta.func, dummy_func)
        self.assertEqual(meta.func_path, f"{dummy_func.__module__}:dummy_func")
        self.assertEqual(meta.name, "my_step")
        self.assertEqual(meta.inputs, ("x",))
        self.assertEqual(meta.outputs, ("y",))
        self.assertEqual(meta.depends_on, ("z",))
        self.assertEqual(meta.context_cls, BaseModel)

    def test_stepmeta_repr_contains_fields(self):
        meta = StepMeta(func=dummy_func)

        repr_str = repr(meta)

        self.assertIn("StepMeta(name=", repr_str)
        self.assertIn("func_path=", repr_str)
        self.assertIn("inputs=", repr_str)
        self.assertIn("outputs=", repr_str)
        self.assertIn("depends_on=", repr_str)
