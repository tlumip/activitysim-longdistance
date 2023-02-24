import logging

import pandas as pd
from activitysim.core import estimation  # noqa: F401
from activitysim.core import config, expressions, los, tracing, workflow

from .ldt_internal_external import LDT_IE_INTERNAL
from .ldt_pattern import LDT_PATTERN
from .util import tour_destination

# ActivitySim
# See full license in LICENSE.txt.


logger = logging.getLogger(__name__)


@workflow.step
def ldt_internal_tour_destination(
    state: workflow.State,
    longdist_tours: pd.DataFrame,
    persons_merged: pd.DataFrame,
    network_los: los.Network_LOS,
):

    """
    Given the tour generation from the above, each tour needs to have a
    destination, so in this case tours are the choosers (with the associated
    person that's making the tour)
    """

    trace_label = "ldt_internal_tour_destination"
    model_settings_file_name = "ldt_internal_tour_destination.yaml"
    model_settings = state.filesystem.read_model_settings(model_settings_file_name)

    logsum_column_name = model_settings.get("DEST_CHOICE_LOGSUM_COLUMN_NAME")
    want_logsums = logsum_column_name is not None

    sample_table_name = model_settings.get("DEST_CHOICE_SAMPLE_TABLE_NAME")
    want_sample_table = (
        state.settings.want_dest_choice_sample_tables and sample_table_name is not None
    )

    # choosers are tours - in a sense tours are choosing their destination
    ldt_tours = longdist_tours
    # pipeline.get_rn_generator().add_channel("longdist_tours", ldt_tours)

    # we do NOT filter for tours with trips on the travel day, as we want
    # to know the destination of long distance tours even if they
    # are "AWAY" on the travel day, so they can be handed to a short distance
    # travel behavior process at that location

    persons_merged = persons_merged

    persons_merged["ldt_pattern"] = (
        persons_merged["ldt_pattern_person"] | persons_merged["ldt_pattern_household"]
    )

    # - if no ldt tours
    if ldt_tours.shape[0] == 0:
        tracing.no_results("ldt_tour_destination")
        return

    # - preprocessor
    preprocessor_settings = model_settings.get("preprocessor", None)
    if preprocessor_settings:
        constants = config.get_model_constants(model_settings)
        locals_d = {
            "LDT_PATTERN": LDT_PATTERN,
        }
        if constants is not None:
            locals_d.update(constants)
        expressions.assign_columns(
            state,
            df=ldt_tours,
            model_settings=preprocessor_settings,
            locals_dict=locals_d,
            trace_label=trace_label,
        )

    # TODO
    # estimator = estimation.manager.begin_estimation(state, "ldt_tour_destination")
    # if estimator:
    #     estimator.write_coefficients(model_settings=model_settings)
    #     # estimator.write_spec(model_settings, tag='SAMPLE_SPEC')
    #     estimator.write_spec(model_settings, tag="SPEC")
    #     estimator.set_alt_id(model_settings["ALT_DEST_COL_NAME"])
    #     estimator.write_table(
    #         inject.get_injectable("size_terms"), "size_terms", append=False
    #     )
    #     estimator.write_table(
    #         inject.get_table("land_use"), "landuse", append=False
    #     )
    #     estimator.write_model_settings(model_settings, model_settings_file_name)

    choices_df, save_sample_df = tour_destination.run_tour_destination(
        state,
        ldt_tours[ldt_tours.internal_external == LDT_IE_INTERNAL],
        persons_merged,
        want_logsums,
        want_sample_table,
        model_settings,
        network_los,
        None,  # estimator
        state.settings.chunk_size,
        state.settings.trace_hh_id,
        trace_label,
        out_period_col="ldt_start_hour",
        in_period_col="ldt_end_hour",
    )

    # TODO
    # if estimator:
    #     estimator.write_choices(choices_df.choice)
    #     choices_df.choice = estimator.get_survey_values(
    #         choices_df.choice, "tours", "destination"
    #     )
    #     estimator.write_override_choices(choices_df.choice)
    #     estimator.end_estimation()

    # merge choices and logsums into table
    renaming = {"choice": "internal_destination"}
    if want_logsums:
        renaming["logsum"] = logsum_column_name
    ldt_tours = ldt_tours.join(
        choices_df[list(renaming.keys())].rename(columns=renaming)
    ).fillna({"internal_destination": -1}, downcast={"internal_destination": "int32"})

    if "annotate_ldt_tours" in model_settings:
        expressions.assign_columns(
            state,
            df=ldt_tours,
            model_settings=model_settings.get("annotate_ldt_tours"),
            trace_label=tracing.extend_trace_label(trace_label, "annotate_ldt_tours"),
        )

    state.add_table("longdist_tours", ldt_tours)

    tracing.print_summary(
        "internal_destination", ldt_tours.internal_destination, describe=True
    )

    if want_sample_table:
        assert len(save_sample_df.index.get_level_values(0).unique()) == len(choices_df)
        # save_sample_df.set_index(model_settings['ALT_DEST_COL_NAME'], append=True, inplace=True)
        state.extend_table(sample_table_name, save_sample_df)

    if state.settings.trace_hh_id:
        state.tracing.trace_df(ldt_tours, label="longdist_destination.ldt_tours")
