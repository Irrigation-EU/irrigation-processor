import json
import os
import tempfile
import unittest
from typing import Any
from unittest.mock import Mock, patch

import xarray as xr
from pydantic import BaseModel, Field

from irrigation_processor.core.service import (LocalService,
                                               save_pipeline_step_state)
from irrigation_processor.core.step import FromStep
from tests.helpers import DummyStepMeta


class TestService(unittest.TestCase):
    def setUp(self):
        self.storage = Mock()
        self.storage.save.side_effect = lambda k, v: f"id_{k}"
        self.storage.load.side_effect = lambda id: f"loaded_{id}"

        self.tmpdir = tempfile.TemporaryDirectory()

        patcher = patch(
            "irrigation_processor.core.service.PIPELINE_RESULTS_DIR",
            self.tmpdir.name,
        )
        patcher.start()

        self.addCleanup(self.tmpdir.cleanup)
        self.addCleanup(patcher.stop)

    def test_save_pipeline_step_state(self):
        data = {"a": 1, "b": "c", "d": True}
        path = save_pipeline_step_state("pipe", "step1", data)

        self.assertTrue(os.path.exists(path))

        with open(path, "r") as f:
            saved_data = json.load(f)

        self.assertEqual(saved_data, data)

    def test_normalize_outputs_dict(self):
        svc = LocalService(self.storage, Mock())
        meta = DummyStepMeta("step", outputs=["a", "b"])

        result = {"a": 1, "b": 2}
        out = svc._normalize_outputs("step", meta, result)
        
        # Inline values
        self.assertEqual(out["a"]["type"], "inline")
        self.assertEqual(out["a"]["value"], 1)
        self.assertEqual(out["b"]["type"], "inline")
        self.assertEqual(out["b"]["value"], 2)

    def test_normalize_outputs_stored(self):
        svc = LocalService(self.storage, Mock())
        meta = DummyStepMeta("step", outputs=["heavy"])

        complex_obj = Mock()
        
        out = svc._normalize_outputs("step", meta, {"heavy": complex_obj})
        
        self.assertEqual(out["heavy"]["type"], "stored")
        self.assertEqual(out["heavy"]["data_id"], "heavy.zarr")
        self.storage.save.assert_called_with("heavy.zarr", complex_obj)


    def test_normalize_outputs_dict_key_mismatch_raises(self):
        svc = LocalService(self.storage, Mock())
        meta = DummyStepMeta("step", outputs=["a"])

        with self.assertRaises(ValueError):
            svc._normalize_outputs("step", meta, {"b": 1})

    def test_normalize_outputs_tuple(self):
        svc = LocalService(self.storage, Mock())
        meta = DummyStepMeta("step", outputs=["x", "y"])

        out = svc._normalize_outputs("step", meta, (10, 20))

        self.assertEqual(out["x"]["value"], 10)
        self.assertEqual(out["y"]["value"], 20)

    def test_normalize_outputs_tuple_length_mismatch(self):
        svc = LocalService(self.storage, Mock())
        meta = DummyStepMeta("step", outputs=["x", "y"])

        with self.assertRaises(ValueError):
            svc._normalize_outputs("step", meta, (1,))

    def test_normalize_outputs_single_value(self):
        svc = LocalService(self.storage, Mock())
        meta = DummyStepMeta("step", outputs=["x"])

        out = svc._normalize_outputs("step", meta, 42)

        self.assertEqual(out["x"]["value"], 42)

    def test_normalize_outputs_too_many_outputs_raises(self):
        svc = LocalService(self.storage, Mock())
        meta = DummyStepMeta("step", outputs=["x", "y"])

        with self.assertRaises(ValueError):
            svc._normalize_outputs("step", meta, 1)

    def test_resolve_inputs_positional(self):
        svc = LocalService(self.storage, Mock())
        svc._state = {"s1": {"out": {"type": "inline", "value": 5}}}

        meta = DummyStepMeta(
            "s2",
            inputs=[FromStep("s1", "out"), 3],
        )

        args, kwargs = svc._resolve_inputs("s2", meta)

        self.assertEqual(args, [5, 3])
        self.assertEqual(kwargs, {})

    def test_resolve_inputs_kwargs(self):
        svc = LocalService(self.storage, Mock())
        svc._state = {"s1": {"out": {"type": "inline", "value": 7}}}

        meta = DummyStepMeta(
            "s2",
            inputs={"x": FromStep("s1", "out")},
        )

        args, kwargs = svc._resolve_inputs("s2", meta)

        self.assertEqual(args, [])
        self.assertEqual(kwargs["x"], 7)
    
    def test_resolve_inputs_stored(self):
        svc = LocalService(self.storage, Mock())
        svc._state = {"s1": {"out": {"type": "stored", "data_id": "abc"}}}

        meta = DummyStepMeta(
            "s2",
            inputs=[FromStep("s1", "out")],
        )

        args, kwargs = svc._resolve_inputs("s2", meta)

        self.storage.load.assert_called_with("abc")
        self.assertEqual(args, ["loaded_abc"])

    def test_resolve_inputs_missing_dependency_raises(self):
        svc = LocalService(self.storage, Mock())

        meta = DummyStepMeta(
            "s2",
            inputs=[FromStep("s1", "out")],
        )

        with self.assertRaises(KeyError):
            svc._resolve_inputs("s2", meta)

    def test_run_executes_pipeline(self):
        def step_fn(ctx, x):
            return x + 1

        svc = LocalService(self.storage, Mock())
        svc.app_config.dask.dask_kwargs.model_dump.return_value = {}

        step1 = DummyStepMeta(
            "step1",
            lambda x: 1,
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


    def test_normalize_outputs_nested_inline(self):
        svc = LocalService(self.storage, Mock())
        meta = DummyStepMeta("step", outputs=["nested"])

        # Nested collections of primitives should be inline
        result = {"nested": [1, {"a": "b"}, (True, None)]}
        out = svc._normalize_outputs("step", meta, result)

        self.assertEqual(out["nested"]["type"], "inline")
        self.assertEqual(out["nested"]["value"], [1, {"a": "b"}, (True, None)])

    def test_normalize_outputs_nested_stored_list(self):
        svc = LocalService(self.storage, Mock())
        meta = DummyStepMeta("step", outputs=["data_list"])

        ds = xr.Dataset()
        result = {"data_list": [1, ds]}
        
        out = svc._normalize_outputs("step", meta, result)

        self.assertEqual(out["data_list"]["type"], "list")
        self.assertEqual(out["data_list"]["items"][0]["type"], "inline")
        self.assertEqual(out["data_list"]["items"][1]["type"], "stored")
        self.assertEqual(out["data_list"]["items"][1]["data_id"], "data_list.1.zarr")
        self.storage.save.assert_called_with("data_list.1.zarr", ds)

    def test_normalize_outputs_nested_stored_dict(self):
        svc = LocalService(self.storage, Mock())
        meta = DummyStepMeta("step", outputs=["data_dict"])

        ds = xr.Dataset()
        result = {"data_dict": {"metadata": "info", "data": ds}}
        
        out = svc._normalize_outputs("step", meta, result)

        self.assertEqual(out["data_dict"]["type"], "dict")
        self.assertEqual(out["data_dict"]["items"]["metadata"]["type"], "inline")
        self.assertEqual(out["data_dict"]["items"]["data"]["type"], "stored")
        self.assertEqual(out["data_dict"]["items"]["data"]["data_id"], "data_dict.data.zarr")
        self.storage.save.assert_called_with("data_dict.data.zarr", ds)

    def test_resolve_inputs_recursive(self):
        svc = LocalService(self.storage, Mock())
        ds = xr.Dataset()
        # Reset side_effect from setUp to allow return_value
        self.storage.load.side_effect = None
        self.storage.load.return_value = ds
        
        svc._state = {
            "s1": {
                "out": {
                    "type": "dict",
                    "items": {
                        "a": {"type": "inline", "value": 1},
                        "b": {"type": "stored", "data_id": "ds_id"}
                    }
                }
            }
        }

        meta = DummyStepMeta("s2", inputs=[FromStep("s1", "out")])
        args, kwargs = svc._resolve_inputs("s2", meta)

        self.assertEqual(args[0]["a"], 1)
        self.assertIs(args[0]["b"], ds)
        self.storage.load.assert_called_with("ds_id")

    def test_normalize_outputs_multiple_outputs_mixed(self):
        svc = LocalService(self.storage, Mock())
        meta = DummyStepMeta("step", outputs=["config", "dataset"])

        ds = xr.Dataset()
        # Return tuple: first is a dict of primitives (inline), second is dataset (stored)
        result = ({"param": 42}, ds)
        
        out = svc._normalize_outputs("step", meta, result)

        self.assertEqual(out["config"]["type"], "inline")
        self.assertEqual(out["config"]["value"], {"param": 42})
        self.assertEqual(out["dataset"]["type"], "stored")
        self.assertEqual(out["dataset"]["data_id"], "dataset.zarr")

    def test_normalize_outputs_default_return_value(self):
        svc = LocalService(self.storage, Mock())
        meta = DummyStepMeta("step", outputs=[]) # No outputs

        out = svc._normalize_outputs("step", meta, 42)
        
        self.assertEqual(out["return_value"]["type"], "inline")
        self.assertEqual(out["return_value"]["value"], 42)

    def test_normalize_outputs_default_return_value_dict(self):
        svc = LocalService(self.storage, Mock())
        meta = DummyStepMeta("step", outputs=[]) # No outputs

        result = {"a": 1, "b": 2}
        out = svc._normalize_outputs("step", meta, result)
        self.assertEqual(out["return_value"]["type"], "inline")
        self.assertEqual(out["return_value"]["value"], {'a': 1, 'b': 2})

    def test_normalize_outputs_default_return_value_stored(self):
        svc = LocalService(self.storage, Mock())
        meta = DummyStepMeta("step", outputs=[])

        ds = xr.Dataset()
        out = svc._normalize_outputs("step", meta, ds)

        self.assertEqual(out["return_value"]["type"], "stored")
        self.assertEqual(out["return_value"]["data_id"], "return_value.zarr")
        self.storage.save.assert_called_with("return_value.zarr", ds)

    @patch("irrigation_processor.core.service.Client")
    @patch("irrigation_processor.core.service.LocalCluster")
    def test_run_with_dask_client(self, mock_cluster, mock_client):
        client_instance = Mock()
        mock_client.return_value = client_instance
        
        # dask_kwargs need to be mocked
        mock_config = Mock()
        mock_config.dask.dask_kwargs.model_dump.return_value = {}

        def step_fn(ctx, storage):
            return 1

        svc = LocalService(self.storage, mock_config)

        step = DummyStepMeta(
            "step1", step_fn, outputs=["out"]
        )

        result = svc.run(
            "pipe",
            ["step1"],
            {"step1": step},
        )

        mock_cluster.assert_called_once()
        mock_client.assert_called_once()
        # client closed at end of pipeline run
        client_instance.close.assert_called_once()
        self.assertEqual(result["step1"]["out"]["value"], 1)
