# ActivitySim
# See full license in LICENSE.txt

import logging

import numpy as np
import pandas as pd
from activitysim.core import config, estimation, expressions, logit, tracing, workflow

from .ldt_pattern_person import LDT_PATTERN
from .ldt_tour_gen import process_longdist_tours

logger = logging.getLogger(__name__)


@workflow.step
def ldt_pattern_household(
    state: workflow.State,
    households: pd.DataFrame,
    households_merged: pd.DataFrame,
):
    """
    Assign a LDT pattern to each household that had a generated LDT trip.

    This model gives each LDT household one of the possible LDT categories for a given day --
    NOTOUR = 0
    BEGIN = 1     # leave home, do not return home today
    END = 2       # arrive home, did not start at home today
    COMPLETE = 3  # long distance day-trip completed today
    AWAY = 4      # away from home, in the middle of a multi-day tour

    - *Configuration File*: `ldt_pattern_household.yaml`
    - *Core Table*: `households`
    - *Result Field*: `ldt_pattern_household`
    - *Result dtype*: `int8`
    """
    trace_label = "ldt_pattern_household"
    model_settings_file_name = "ldt_pattern_household.yaml"

    choosers = households_merged
    # if we want to limit choosers, we can do so here
    # limiting ldt_pattern_household to households that go on LDTs
    choosers = choosers[choosers.ldt_tour_gen_household]
    logger.info("Running %s with %d households", trace_label, len(choosers))

    # preliminary estimation steps
    model_settings = state.filesystem.read_model_settings(model_settings_file_name)
    estimator = estimation.manager.begin_estimation(state, trace_label)

    # reading in the probability distribution of household patterns
    constants = config.get_model_constants(model_settings)

    # preprocessor - adds nothing
    preprocessor_settings = model_settings.get("preprocessor", None)
    if preprocessor_settings:

        locals_d = {"LDT_PATTERN": LDT_PATTERN}
        if constants is not None:
            locals_d.update(constants)

        expressions.assign_columns(
            state,
            df=choosers,
            model_settings=preprocessor_settings,
            locals_dict=locals_d,
            trace_label=trace_label,
        )

    # base estimator
    if estimator:
        estimator.write_model_settings(model_settings, model_settings_file_name)
        estimator.write_spec(model_settings)
        # estimator.write_coefficients(coefficients_df, model_settings)
        estimator.write_choosers(choosers)

    # calculating complementary probability
    notour_prob = (
        1
        - constants["COMPLETE"]
        - constants["BEGIN"]
        - constants["END"]
        - constants["AWAY"]
    )

    # broadcast probs to array, without using so much memory
    pr = np.broadcast_to(
        np.asarray(
            [
                notour_prob,
                constants["BEGIN"],
                constants["END"],
                constants["COMPLETE"],
                constants["AWAY"],
            ]
        ),
        (len(choosers.index), 5),
    )
    # sampling probabilities to draw from
    df = pd.DataFrame(
        pr,
        index=choosers.index,
        columns=[
            LDT_PATTERN.NOTOUR,
            LDT_PATTERN.BEGIN,
            LDT_PATTERN.END,
            LDT_PATTERN.COMPLETE,
            LDT_PATTERN.AWAY,
        ],
    )
    # _ is the random value used to make the monte carlo draws, not used
    # this is safe, case where trace_hh_id isn't being considered here is covered
    choices, _ = logit.make_choices(
        state, df, trace_choosers=state.settings.trace_hh_id
    )

    # overwriting estimator
    if estimator:
        estimator.write_choices(choices)
        choices = estimator.get_survey_values(
            choices, "households", "ldt_pattern_household"
        )
        estimator.write_override_choices(choices)
        estimator.end_estimation()

    # setting NOTOUR to non-LDT households
    households = households
    households["ldt_pattern_household"] = (
        choices.reindex(households.index).fillna(LDT_PATTERN.NOTOUR).astype("int8")
    )

    # adding some convenient fields
    # household is scheduled to go on hh ldt (not including away)
    households["on_hh_ldt"] = (
        households["ldt_pattern_household"] != LDT_PATTERN.NOTOUR
    ) & (households["ldt_pattern_household"] != LDT_PATTERN.AWAY)

    # merging into households
    state.add_table("households", households)

    tracing.print_summary("ldt_pattern_household", choices, value_counts=True)

    if state.settings.trace_hh_id:
        state.tracing.trace_df(households, label=trace_label)

    # initializing the longdist tours table with actual household ldt trips (both genereated and scheduled)
    hh_making_longdist_tours = households[households["on_hh_ldt"]]
    tour_counts = (
        hh_making_longdist_tours[["on_hh_ldt"]]
        .astype(int)
        .rename(columns={"on_hh_ldt": "longdist_household"})
    )
    hh_longdist_tours = process_longdist_tours(
        # making longdist the braoder tour category instead of segmenting by all types of ldt
        households,
        tour_counts,
        "longdist",
    )

    hh_longdist_tours = pd.merge(
        hh_longdist_tours,
        households[["ldt_pattern_household"]],
        how="left",
        left_on="household_id",
        right_index=True,
    ).rename(columns={"ldt_pattern_household": "ldt_pattern"})

    # convenient field to differentiate between person and household tours
    # actor type for hh longdist tours is household
    hh_longdist_tours["actor_type"] = "household"

    state.extend_table("longdist_tours", hh_longdist_tours)

    if state.settings.trace_hh_id:
        state.tracing.trace_df(households, label=trace_label, warn_if_empty=True)
