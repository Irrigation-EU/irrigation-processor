from irrigation_processor.constants import logger


def irrigation_simulator(context, calibrated_path, preprocessed_path) -> dict:
    logger.info(
        f"simulating rainfall... {context} {calibrated_path} {preprocessed_path}"
    )
    simulated_path = "/path/to/simulated_data"
    logger.info(f"simulation complete... {simulated_path}")
    return {"simulated_path": simulated_path}
