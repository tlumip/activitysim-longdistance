# ActivitySim
# See full license in LICENSE.txt.
import logging

import pandas as pd
from activitysim.core import expressions, workflow

logger = logging.getLogger(__name__)


@workflow.step
def annotate_accessibility(
    state: workflow.State,
    accessibility: pd.DataFrame,
):
    """
    Annotate accessibility
    """

    trace_label = "annotate_accessibility"
    model_settings = state.filesystem.read_model_settings("accessibility.yaml")

    logger.info(f"{trace_label} computed accessibilities {accessibility.shape}")

    annotate = model_settings.get("annotate_accessibility", None)
    if annotate:
        expressions.assign_columns(
            state,
            df=accessibility,
            model_settings=annotate,
            trace_label="annotate_accessibility",
        )

    # - write table to pipeline
    state.add_table("accessibility", accessibility)
