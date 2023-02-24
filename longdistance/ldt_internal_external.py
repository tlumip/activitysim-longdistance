# ActivitySim
# See full license in LICENSE.txt

import logging

import numpy as np
import pandas as pd
from activitysim.core import config, estimation, los, simulate, tracing, workflow
from activitysim.core.util import assign_in_place

logger = logging.getLogger(__name__)

LDT_IE_INTERNAL = 0
LDT_IE_EXTERNAL = 1
LDT_IE_NULL = -1


@workflow.step
def ldt_internal_external(
    state: workflow.State,
    longdist_tours: pd.DataFrame,
    persons_merged: pd.DataFrame,
    network_los: los.Network_LOS,
    land_use: pd.DataFrame,
):
    """
    This model determines if a person on an LDT is going/will go/is at an internal location (0)
    or at an external location (1) with respect to the model area

    - *Configuration File*: `ldt_internal_external.yaml`
    - *Core Table*: `longdist_tours`
    - *Result Field*: `ldt_internal_external`
    - *Result dtype*: `int8`
    """
    trace_label = "ldt_internal_external"
    colname = "internal_external"
    model_settings_file_name = "ldt_internal_external.yaml"
    segment_column_name = "tour_type"

    # preliminary estimation steps
    model_settings = state.filesystem.read_model_settings(model_settings_file_name)
    estimator = estimation.manager.begin_estimation(state, trace_label)
    constants = config.get_model_constants(model_settings)  # constants shared by all

    # merging in global constants
    categories = state.get_global_constants()
    constants.update(categories)

    # converting parameters to dataframes
    ldt_tours = longdist_tours
    logger.info("Running %s with %d tours" % (trace_label, ldt_tours.shape[0]))

    # merge in persons_merged data as source of external data for estimation
    ldt_tours_merged = pd.merge(
        ldt_tours,
        persons_merged,
        left_on="person_id",
        right_index=True,
        how="left",
        suffixes=("", "_r"),
    )

    model_spec = simulate.read_model_spec(
        state.filesystem, file_name=model_settings["SPEC"]
    )  # reading in generic model spec
    nest_spec = config.get_logit_model_settings(model_settings)  # MNL

    # import data from settings file
    dim3 = model_settings.get("SKIM_KEY", None)
    segment_key = model_settings.get("SEGMENT_KEY", None)
    model_area_key = model_settings.get("MODEL_AREA_KEY", None)
    assert dim3 is not None
    assert segment_key is not None
    assert model_area_key is not None

    # create a taz_times dataframe to get distances to external TAZs from all other TAZs
    taz_times = get_car_time_skim(
        network_los, land_use, dim3, segment_key, model_area_key
    )
    # create a new field for the minimum time to exit the model area
    ldt_tours_merged["min_external_taz_time"] = ldt_tours_merged["home_zone_id"].apply(
        lambda x: np.min(taz_times[x - 1])
    )

    # list to append all result files into
    choices_list = []
    # run model for each purpose (household/workrelated/other) separately due to
    # different coefficients for each
    for tour_purpose, tours_segment in ldt_tours_merged.groupby(segment_column_name):
        # get the name of the purpose
        if tour_purpose.startswith("longdist_"):
            tour_purpose = tour_purpose[9:]
        tour_purpose = tour_purpose.lower()

        # the specific segment to estimate on
        choosers = tours_segment

        logger.info(
            "ldt_internal_external tour_type '%s' (%s tours)"
            % (
                tour_purpose,
                len(choosers.index),
            )
        )

        # logic if there are no choosers - set default to -1
        if choosers.empty:
            choices_list.append(
                pd.Series(-1, index=tours_segment.index, name=colname).to_frame()
            )
            continue

        # read in coefficietns and specification specific to the purpose
        coefficients_df = simulate.get_segment_coefficients(
            state.filesystem, model_settings, tour_purpose
        )
        category_spec = simulate.eval_coefficients(
            state, model_spec, coefficients_df, estimator
        )

        if estimator:
            estimator.write_model_settings(model_settings, model_settings_file_name)
            estimator.write_spec(model_settings)
            estimator.write_coefficients(coefficients_df, model_settings)
            estimator.write_choosers(choosers)

        # run the MNL model
        choices = simulate.simple_simulate(
            state,
            choosers=choosers,
            spec=category_spec,
            nest_spec=nest_spec,
            locals_d=constants,
            trace_label=tracing.extend_trace_label(trace_label, tour_purpose),
            trace_choice_name=colname,
            estimator=estimator,
        )

        # convert choices to dataframe
        if isinstance(choices, pd.Series):
            choices = choices.to_frame(colname)

        # fit choices to the original estimation segment and set result to -1
        choices = choices.reindex(tours_segment.index).fillna(
            {colname: LDT_IE_NULL}, downcast="infer"
        )

        tracing.print_summary(
            "ldt_internal_external %s choices" % tour_purpose,
            choices[colname],
            value_counts=True,
        )

        # append estimated results to the list
        choices_list.append(choices)

    # merge all results into one big dataframe
    choices_df = pd.concat(choices_list)

    tracing.print_summary(
        "ldt_internal_external all tour type choices",
        choices_df[colname],
        value_counts=True,
    )

    # merge into ldt tours & replace the pipeline value
    assign_in_place(ldt_tours, choices_df)

    state.add_table("longdist_tours", ldt_tours)

    if state.settings.trace_hh_id:
        state.tracing.trace_df(
            ldt_tours,
            label=trace_label,
            slicer="tour_id",
            index_label="tour_id",
            warn_if_empty=True,
        )


def get_car_time_skim(network_los, land_use, dim3, segment_key, model_area_key):
    """
    This model handles the logic for converting the network_los into a dataframe for travel times between
    TAZs and all external TAZs
    """
    from activitysim.core.skim_dataset import SkimDataset
    from activitysim.core.skim_dictionary import SkimDict

    skim_dict = network_los.get_default_skim_dict()
    if isinstance(skim_dict, SkimDict):
        skims = skim_dict.skim_data._skim_data
        key_dict = network_los.get_default_skim_dict().skim_dim3
        key = key_dict[dim3][segment_key]
        skim = skims[key]

        external_tazs = land_use[land_use[model_area_key] == 0].index

        return skim[:, external_tazs - 1]
    elif isinstance(skim_dict, SkimDataset):
        time_array = skim_dict.dataset[dim3].sel(time_period=segment_key)
        external_tazs = land_use[land_use[model_area_key] == 0].index
        return time_array.isel(dtaz=external_tazs).to_numpy()
