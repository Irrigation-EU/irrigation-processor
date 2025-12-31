import unittest
from pathlib import Path
from unittest.mock import patch, Mock

from irrigation_processor.main import execute_pipeline


class TestExecutePipeline(unittest.TestCase):
    def test_config_file_not_found(self):
        fake_path = Path("does_not_exist.yml")

        with self.assertRaises(FileNotFoundError):
            execute_pipeline(fake_path)

    @patch("irrigation_processor.main.registry")
    @patch("irrigation_processor.main.Pipeline")
    @patch("irrigation_processor.main.LocalService")
    @patch("irrigation_processor.main.inject_dynamic_context_from_config")
    @patch("irrigation_processor.main.XcubeDataStoreStorage")
    def test_execute_pipeline_flow(
        self,
        mock_storage_cls,
        mock_inject,
        mock_service_cls,
        mock_pipeline_cls,
        mock_registry,
    ):
        tmp_dir = Path.cwd() / ".tmp_test_pipeline"
        tmp_dir.mkdir(exist_ok=True)
        config_file = tmp_dir / "config.yml"
        config_file.write_text(
            """
            base:
              foo: bar
            """
        )

        try:
            pipeline = Mock()
            pipeline.run.return_value = {"status": "ok"}
            mock_pipeline_cls.return_value = pipeline

            execute_pipeline(config_file)

            mock_storage_cls.assert_called_once()
            mock_inject.assert_called_once()

            mock_service_cls.assert_called_once()
            mock_pipeline_cls.assert_called_once()

            pipeline.add_steps_from_registry.assert_called_once()
            pipeline.run.assert_called_once()

            execute_pipeline(
                config_file=config_file,
                disable_steps=["step_a", "step_b"],
            )

            mock_registry.disable.assert_any_call("step_a")
            mock_registry.disable.assert_any_call("step_b")

        finally:
            if config_file.exists():
                config_file.unlink()
            if tmp_dir.exists():
                tmp_dir.rmdir()
