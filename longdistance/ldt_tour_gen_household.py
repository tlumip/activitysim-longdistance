# ActivitySim
# See full license in LICENSE.txt.
import logging

import pandas as pd
from activitysim.core import (
    config,
    estimation,
    expressions,
    simulate,
    tracing,
    workflow,
)

logger = logging.getLogger(__name__)


@workflow.step
def ldt_tour_gen_household(
    state: workflow.State,
    households: pd.DataFrame,
    households_merged: pd.DataFrame,
):
    """
    This model predicts whether a household will go on an LDT trip over a 2 week period.

    - *Configuration File*: `ldt_tour_gen_household.yaml`
    - *Core Table*: `households`
    - *Result Field*: `ldt_tour_gen_household`
    - *Result dtype*: `bool`
    """

    trace_label = "ldt_tour_gen_household"
    model_settings_file_name = "ldt_tour_gen_household.yaml"

    # convert the households_merged to a dataframe as a choosers file
    choosers = households_merged
    # if we want to limit choosers, we can do so here
    # choosers = choosers[choosers.workplace_zone_id > -1]
    logger.info("Running %s with %d households", trace_label, len(choosers))

    # read in the model settings from the specified path
    model_settings = state.filesystem.read_model_settings(model_settings_file_name)
    estimator = estimation.manager.begin_estimation(state, trace_label)

    # reading in some category constants
    constants = config.get_model_constants(model_settings)

    # merging in global constants
    categories = state.get_global_constants()
    constants.update(categories)

    # preprocessor - adds nothing
    preprocessor_settings = model_settings.get("preprocessor", None)
    if preprocessor_settings:
        locals_d = {}
        if constants is not None:
            locals_d.update(constants)

        expressions.assign_columns(
            state,
            df=choosers,
            model_settings=preprocessor_settings,
            locals_dict=locals_d,
            trace_label=trace_label,
        )

    # reading in model specification/coefficients
    model_spec = simulate.read_model_spec(
        state.filesystem, file_name=model_settings["SPEC"]
    )
    coefficients_df = simulate.read_model_coefficients(state.filesystem, model_settings)
    model_spec = simulate.eval_coefficients(
        state, model_spec, coefficients_df, estimator
    )

    nest_spec = config.get_logit_model_settings(model_settings)  # MNL

    if estimator:
        estimator.write_model_settings(model_settings, model_settings_file_name)
        estimator.write_spec(model_settings)
        estimator.write_coefficients(coefficients_df, model_settings)
        estimator.write_choosers(choosers)

    # running the tour gen multinomial logit model
    choices = simulate.simple_simulate(
        state,
        choosers=choosers,
        spec=model_spec,
        nest_spec=nest_spec,
        locals_d=constants,
        trace_label=trace_label,
        trace_choice_name="ldt_tour_gen_household",
        estimator=estimator,
    )

    if estimator:
        estimator.write_choices(choices)
        choices = estimator.get_survey_values(
            choices, "households", "ldt_tour_gen_household"
        )
        estimator.write_override_choices(choices)
        estimator.end_estimation()

    # merging in tour gen results to households df
    households["ldt_tour_gen_household"] = (
        choices.reindex(households.index).fillna(0).astype(bool)
    )

    # merging into final_households csv
    state.add_table("households", households)

    tracing.print_summary(
        "ldt_tour_gen_household",
        choices,
        value_counts=True,
    )

    if state.settings.trace_hh_id:
        state.tracing.trace_df(households, label=trace_label, warn_if_empty=True)
