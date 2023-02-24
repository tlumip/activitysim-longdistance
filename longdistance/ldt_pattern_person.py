# ActivitySim
# See full license in LICENSE.txt

import logging

import numpy as np
import pandas as pd
from activitysim.core import config, estimation, expressions, logit, tracing, workflow

from .ldt_pattern import LDT_PATTERN, LDT_PATTERN_BITSHIFT
from .ldt_tour_gen import process_longdist_tours

logger = logging.getLogger(__name__)


@workflow.step
def ldt_pattern_person(
    state: workflow.State,
    persons: pd.DataFrame,
    persons_merged: pd.DataFrame,
):
    """
    Assign a LDT pattern to each person that had a generated LDT trip.

    This model gives each LDT household one of the possible LDT categories for a given day --
    NOTOUR = 0
    BEGIN = 9/17     # leave home, do not return home today
    END = 10/18       # arrive home, did not start at home today
    COMPLETE = 11/19  # long distance day-trip completed today
    AWAY = 12/20      # away from home, in the middle of a multi-day tour

    These categories are equal to the base pattern num (0/1/2/3/4) plus a bitshift of the
    category num (0 for household, 1 for workrelated, 2 for other)
    adjusted pattern num = base pattern num + (category num << 3)

    - *Configuration File*: `ldt_pattern_person.yaml`
    - *Core Table*: `personss`
    - *Result Field*: `ldt_pattern_person`
    - *Result dtype*: `int8`
    """
    trace_label = "ldt_pattern_person"
    model_settings_file_name = "ldt_pattern_person.yaml"

    choosers_full = persons_merged

    logger.info("Running %s with %d persons", trace_label, len(choosers_full))

    # preliminary estimation steps
    model_settings = state.filesystem.read_model_settings(model_settings_file_name)
    estimator = estimation.manager.begin_estimation(state, trace_label)

    constants = config.get_model_constants(model_settings)

    # preprocessor - adds nothing
    preprocessor_settings = model_settings.get("preprocessor", None)
    if preprocessor_settings:
        locals_d = {}
        if constants is not None:
            locals_d.update(constants)

        expressions.assign_columns(
            state,
            df=choosers_full,
            model_settings=preprocessor_settings,
            locals_dict=locals_d,
            trace_label=trace_label,
        )
    # read in settings for all purpose (other/workrelated)
    spec_purposes = model_settings.get("SPEC_PURPOSES", {})

    # create dataframe that contains whether a person is already generated to go on a household LDT trip
    person_on_hh_ldt = choosers_full["ldt_pattern_household"] != LDT_PATTERN.NOTOUR

    # set default pattern to the notour enum
    persons["ldt_pattern_person"] = pd.Series(
        LDT_PATTERN.NOTOUR, index=persons.index, dtype=np.uint8
    )
    # use choosers_full with normal (non bit shifted) pattern numbers as proxy to avoid complex calculations
    # all changes to choosers_full will be dumped after this step
    choosers_full["ldt_pattern_person"] = pd.Series(
        LDT_PATTERN.NOTOUR, index=persons.index, dtype=np.uint8
    )

    # run pattern assignment for the two individual purposes (workrelated/other)
    for purpose_num, purpose_settings in enumerate(spec_purposes, start=1):
        # WORKRELATED and OTHER
        purpose_name = purpose_settings["NAME"]

        # only consider people who are predicted to go on LDT tour over 2 week period
        # and who are not already on an LDT tour today
        # this follows the hierarchy for LDT pattern: household takes precedence
        # over workrelated and workrelated takes precedence over other
        choosers = choosers_full[
            (choosers_full["ldt_tour_gen_person_" + purpose_name])
            & (~person_on_hh_ldt)
            & (choosers_full["ldt_pattern_person"] == LDT_PATTERN.NOTOUR)
        ]

        # if temp:
        #     # only consider people who aren't scheduled to go on work LDT when scheduling other LDT
        #     choosers = choosers[
        #         choosers["ldt_pattern_person_WORKRELATED"] == LDT_PATTERN.NOTOUR
        #     ]

        # reading in the probability distribution for the current pattern type
        constants = config.get_model_constants(purpose_settings)

        if estimator:
            estimator.write_model_settings(model_settings, model_settings_file_name)
            estimator.write_spec(model_settings)
            # estimator.write_coefficients(coefficients_df, model_settings)
            estimator.write_choosers(choosers)

        # calculating complementary probability of not going on a tour
        notour_prob = (
            1
            - constants["COMPLETE"]
            - constants["BEGIN"]
            - constants["END"]
            - constants["AWAY"]
        )

        # broadcast probabilities
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
        choices, _ = logit.make_choices(
            state, df, trace_choosers=state.settings.trace_hh_id
        )

        if estimator:
            estimator.write_choices(choices)
            choices = estimator.get_survey_values(
                choices, "persons", "ldt_pattern_person"
            )
            estimator.write_override_choices(choices)
            estimator.end_estimation()

        # making one ldt pattern field instead of segmenting by person/household
        # bitshift makes all combinations of pattern/purpose unique
        persons.loc[choices.index, "ldt_pattern_person"] = (
            choices.values + (purpose_num << LDT_PATTERN_BITSHIFT)
        ).astype(np.uint8)
        choosers_full.loc[choices.index, "ldt_pattern_person"] = (
            choices.values
        ).astype(np.uint8)

        tracing.print_summary(
            f"ldt_pattern_person/{purpose_name}",
            choices,
            value_counts=True,
        )
        # TODO: fix logic for the below statement; currently doesn't consider the varying ldt pattern codes due to bitshift
        # person_already_on_ldt |= persons["ldt_pattern_person"] != LDT_PATTERN.NOTOUR

    # adding convenient fields
    # whether or not person is scheduled to be on LDT trip (not including away)
    persons["on_person_ldt"] = (
        choosers_full["ldt_pattern_person"].isin(
            [LDT_PATTERN.COMPLETE, LDT_PATTERN.BEGIN, LDT_PATTERN.END]
        )
    ) & (~person_on_hh_ldt)
    # persons["ldt_pattern_person"].isin(
    #     [x + (y << LDT_PATTERN_BITSHIFT) for x in [LDT_PATTERN.COMPLETE, LDT_PATTERN.BEGIN, LDT_PATTERN.END] for y in [1, 2]]
    # ) & (
    #     choosers_full["ldt_pattern_household"] == LDT_PATTERN.NOTOUR
    # )
    # persons["on_person_ldt"] = person_already_on_ldt & (
    #     choosers_full["ldt_pattern_household"] == LDT_PATTERN.NOTOUR
    # )

    # -1 is no LDT trip (whether a trip was not generated/not scheduled), 0 is work related, 1 is other
    # persons["ldt_purpose"] = np.where(persons["on_person_ldt"], 1, -1)
    # persons["ldt_purpose"] = np.where(
    #     persons["ldt_pattern_person_WORKRELATED"].isin([0, 1, 2, 3]),
    #     0,
    #     persons["ldt_purpose"],
    # )

    # -1 is no LDT trip (whether a trip was not generated/not scheduled), others match up to the pattern for a
    # person's specified ldt_purpose (excluding 4, which means no scheduled LDT--changed to -1)
    # persons["ldt_pattern"] = np.where(persons["on_person_ldt"], 0, -1)
    # persons["ldt_pattern"] = np.where(
    #     persons["ldt_purpose"] == 0,
    #     persons["ldt_pattern_person_WORKRELATED"],
    #     persons["ldt_pattern"],
    # )
    # persons["ldt_pattern"] = np.where(
    #     persons["ldt_purpose"] == 1,
    #     persons["ldt_pattern_person_OTHER"],
    #     persons["ldt_pattern"],
    # )

    # merging changes to persons table to the final_persons csv
    state.add_table("persons", persons)

    # add all scheduled/generated tours to the longdist_tours csv
    ldt_tours = None
    for purpose_num, purpose_settings in enumerate(spec_purposes, start=1):
        # purpose_num zero is for household patterns
        purpose_name = purpose_settings["NAME"]
        ldt_tours = process_person_tours(state, persons, purpose_name, purpose_num)

    if ldt_tours is not None:
        state.get_rn_generator().add_channel("longdist_tours", ldt_tours)

    logger.debug("ldt_pattern_person complete")

    if state.settings.trace_hh_id:
        state.tracing.trace_df(persons, label=trace_label)


def process_person_tours(
    state: workflow.State, persons, purpose: str, purpose_num: int
):
    """
    This function adds the generated individual ldt trips to the longdist_trips csv.
    """
    # consider the people actually making longdist tours (generated/valid pattern)
    persons_making_longdist_tours = persons[
        (persons["ldt_pattern_person"].values >> LDT_PATTERN_BITSHIFT) == purpose_num
    ]
    # number of tours generated, always 1, but in this format for compatability
    tour_counts = (
        persons_making_longdist_tours[["on_person_ldt"]]
        .astype(int)
        .rename(columns={"on_person_ldt": f"longdist_person_{purpose}"})
    )

    # processing the generated longdist tours to add to longdist_tours csv
    longdist_tours_person = process_longdist_tours(persons, tour_counts, "longdist")

    # merging ldt pattern into generated longdist tours
    longdist_tours_person = pd.merge(
        longdist_tours_person,
        persons[["ldt_pattern_person"]],
        how="left",
        left_on="person_id",
        right_index=True,
    ).rename(columns={"ldt_pattern_person": "ldt_pattern"})

    # adding a convenience field to differentiate between person/household ldt trips
    longdist_tours_person["actor_type"] = "person"

    # merging current individual ldt trips into longdist_tours csv
    return state.extend_table("longdist_tours", longdist_tours_person)
