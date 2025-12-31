import os
import tempfile
import unittest
from unittest.mock import Mock, patch

from irrigation_processor.core.service import (
    Service,
    LocalService,
    save_pipeline_step_state,
    load_pipeline_step_state,
)
from irrigation_processor.core.step import FromStep


class DummyStepMeta:
    def __init__(
        self,
        name,
        func=None,
        inputs=None,
        outputs=None,
        context_cls=None,
    ):
        self.name = name
        self.func = func
        self.inputs = inputs or []
        self.outputs = outputs or []
        self.context_cls = context_cls


class TestService(unittest.TestCase):
    def setUp(self):
        self.storage = Mock()
        self.storage.save.side_effect = lambda k, v: {"key": k, "value": v}
        self.storage.load.side_effect = lambda v: v["value"]

        self.tmpdir = tempfile.TemporaryDirectory()

        patcher = patch(
            "irrigation_processor.core.service.PIPELINE_RESULTS_CACHE_DIR",
            self.tmpdir.name,
        )
        patcher.start()

        self.addCleanup(self.tmpdir.cleanup)
        self.addCleanup(patcher.stop)

    def test_service_run_not_implemented(self):
        svc = Service(storage=self.storage)

        with self.assertRaises(NotImplementedError):
            svc.run("pipeline", [], {})

    def test_save_and_load_pipeline_step_state(self):
        data = {"a": 1, "b": "c", "d": True}
        path = save_pipeline_step_state("pipe", "step1", data)

        self.assertTrue(os.path.exists(path))

        loaded = load_pipeline_step_state("pipe", "step1")
        self.assertEqual(loaded, data)

    def test_load_missing_state_raises(self):
        with self.assertRaises(FileNotFoundError):
            load_pipeline_step_state("pipe", "missing")

    def test_normalize_outputs_dict(self):
        svc = LocalService(self.storage)
        meta = DummyStepMeta("step", outputs=["a", "b"])

        result = {"a": 1, "b": 2}
        out = svc._normalize_outputs("step", meta, result)
        self.assertEqual(out["a"]["key"], "a")
        self.assertEqual(out["b"]["key"], "b")
        self.assertEqual(out["a"]["value"], 1)
        self.assertEqual(out["b"]["value"], 2)

    def test_normalize_outputs_dict_key_mismatch_raises(self):
        svc = LocalService(self.storage)
        meta = DummyStepMeta("step", outputs=["a"])

        with self.assertRaises(ValueError):
            svc._normalize_outputs("step", meta, {"b": 1})

    def test_normalize_outputs_tuple(self):
        svc = LocalService(self.storage)
        meta = DummyStepMeta("step", outputs=["x", "y"])

        out = svc._normalize_outputs("step", meta, (10, 20))

        self.assertEqual(out["x"]["value"], 10)
        self.assertEqual(out["y"]["value"], 20)

    def test_normalize_outputs_tuple_length_mismatch(self):
        svc = LocalService(self.storage)
        meta = DummyStepMeta("step", outputs=["x", "y"])

        with self.assertRaises(ValueError):
            svc._normalize_outputs("step", meta, (1,))

    def test_normalize_outputs_single_value(self):
        svc = LocalService(self.storage)
        meta = DummyStepMeta("step", outputs=["x"])

        out = svc._normalize_outputs("step", meta, 42)

        self.assertEqual(out["x"]["value"], 42)

    def test_normalize_outputs_too_many_outputs_raises(self):
        svc = LocalService(self.storage)
        meta = DummyStepMeta("step", outputs=["x", "y"])

        with self.assertRaises(ValueError):
            svc._normalize_outputs("step", meta, 1)

    def test_normalize_outputs_no_outputs_defined_raises(self):
        svc = LocalService(self.storage)
        meta = DummyStepMeta("step", outputs=[])

        with self.assertRaises(ValueError):
            svc._normalize_outputs("step", meta, 1)

    def test_resolve_inputs_positional(self):
        svc = LocalService(self.storage)
        svc._state = {"s1": {"out": {"value": 5}}}

        meta = DummyStepMeta(
            "s2",
            inputs=[FromStep("s1", "out"), 3],
        )

        args, kwargs = svc._resolve_inputs("s2", meta, self.storage, "pipe")

        self.assertEqual(args, [5, 3])
        self.assertEqual(kwargs, {})

    def test_resolve_inputs_kwargs(self):
        svc = LocalService(self.storage)
        svc._state = {"s1": {"out": {"value": 7}}}

        meta = DummyStepMeta(
            "s2",
            inputs={"x": FromStep("s1", "out")},
        )

        args, kwargs = svc._resolve_inputs("s2", meta, self.storage, "pipe")

        self.assertEqual(args, [])
        self.assertEqual(kwargs["x"], 7)

    def test_resolve_inputs_missing_dependency_raises(self):
        svc = LocalService(self.storage)

        meta = DummyStepMeta(
            "s2",
            inputs=[FromStep("s1", "out")],
        )

        with self.assertRaises(KeyError):
            svc._resolve_inputs("s2", meta, self.storage, "pipe")

    def test_run_executes_pipeline(self):
        def step_fn(ctx, x):
            return x + 1

        svc = LocalService(self.storage)

        step1 = DummyStepMeta(
            "step1",
            lambda ctx: 1,
            outputs=["out"],
        )
        step2 = DummyStepMeta(
            "step2",
            step_fn,
            inputs=[FromStep("step1", "out")],
            outputs=["result"],
        )

        result = svc.run(
            "pipe",
            ["step1", "step2"],
            {"step1": step1, "step2": step2},
        )

        self.assertEqual(result["step1"]["out"]["value"], 1)
        self.assertEqual(result["step2"]["result"]["value"], 2)

    @patch("irrigation_processor.core.service.Client")
    @patch("irrigation_processor.core.service.LocalCluster")
    def test_run_with_dask_client(self, mock_cluster, mock_client):
        client_instance = Mock()
        mock_client.return_value = client_instance

        def step_fn(ctx, dask_client):
            self.assertIs(dask_client, client_instance)
            return 1

        svc = LocalService(self.storage)

        step = DummyStepMeta(
            "step1",
            step_fn,
            outputs=["out"],
        )

        result = svc.run(
            "pipe",
            ["step1"],
            {"step1": step},
        )

        mock_cluster.assert_called_once()
        mock_client.assert_called_once()
        client_instance.close.assert_called_once()
        self.assertEqual(result["step1"]["out"]["value"], 1)

    @patch("irrigation_processor.core.service.load_pipeline_step_state")
    def test_resolve_inputs_with_cache(self, mock_load_state):
        mock_load_state.return_value = {"out": {"value": 10}}

        svc = LocalService(self.storage, use_cache=True)

        meta_args = DummyStepMeta(
            "step2",
            inputs=[FromStep("step1", "out")],
        )

        args, kwargs = svc._resolve_inputs(
            "step2",
            meta_args,
            self.storage,
            "pipe",
        )

        self.assertEqual(args, [10])
        self.assertEqual(kwargs, {})

        meta_kwargs = DummyStepMeta(
            "step3",
            inputs={"x": FromStep("step1", "out")},
        )

        args, kwargs = svc._resolve_inputs(
            "step3",
            meta_kwargs,
            self.storage,
            "pipe",
        )

        self.assertEqual(args, [])
        self.assertEqual(kwargs["x"], 10)
