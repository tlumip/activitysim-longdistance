# ActivitySim
# See full license in LICENSE.txt

import logging

import numpy as np
import pandas as pd
from activitysim.core import config, estimation, logit, los, tracing, workflow
from activitysim.core.skim_dataset import SkimDataset
from activitysim.core.skim_dictionary import SkimDict
from activitysim.core.util import assign_in_place

from longdistance.ldt_internal_external import LDT_IE_EXTERNAL

logger = logging.getLogger(__name__)


@workflow.step
def ldt_external_destchoice(
    state: workflow.State,
    longdist_tours: pd.DataFrame,
    persons_merged: pd.DataFrame,
    network_los: los.Network_LOS,
    land_use: pd.DataFrame,
):
    """
    This model determines the chosen destination of those traveling externally
    based on a probability distribution and a 50 mile limit for LDTs.

    - *Configuration File*: `ldt_external_destchoice.yaml`
    - *Core Table*: `longdist_tours`
    - *Result Field*: `ldt_external_destchoice`
    - *Result dtype*: `int16`
    """
    trace_label = "ldt_external_destchoice"
    colname = "external_destchoice"
    model_settings_file_name = "ldt_external_destchoice.yaml"
    segment_column_name = "tour_type"

    # preliminary estimation steps
    model_settings = state.filesystem.read_model_settings(model_settings_file_name)
    estimator = estimation.manager.begin_estimation(state, trace_label)
    constants = config.get_model_constants(model_settings)  # constants shared by all

    # merging in global constants
    category_file_name = model_settings.get("CATEGORY_CONSTANTS", None)
    if category_file_name is not None:
        categories = config.get_model_constants(
            state.filesystem.read_model_settings(category_file_name)
        )
        constants.update(categories)

    # converting parameters to dataframes
    ldt_tours = longdist_tours
    logger.info("Running %s with %d tours" % (trace_label, ldt_tours.shape[0]))

    # merging external persons data into ldt_tours
    persons_merged = persons_merged
    ldt_tours_merged = pd.merge(
        ldt_tours,
        persons_merged,
        left_on="person_id",
        right_index=True,
        how="left",
        suffixes=("", "_r"),
    )

    # read in settings parameters relevant to getting cardist skim
    dim3 = model_settings.get("SKIM_KEY", None)
    time_key = model_settings.get("SEGMENT_KEY", None)
    model_area_key = model_settings.get("MODEL_AREA_KEY", None)
    assert dim3 is not None
    assert time_key is not None
    assert model_area_key is not None

    # get skim for travel distances between origin TAZs to external TAZs
    taz_dists = get_car_dist_skim(network_los, land_use, dim3, time_key, model_area_key)

    # read in external probabilities file & respective csv
    external_probabilities_file_path = state.filesystem.get_config_file_path(
        model_settings.get("REGION_PROBABILITIES")
    )
    external_probabilities = pd.read_csv(external_probabilities_file_path, index_col=0)

    # recode external_probabilities index values to match ActivitySim's recoded TAZs
    from activitysim.core.cleaning import recode_based_on_table

    external_probabilities.index = recode_based_on_table(
        state, external_probabilities.index, "land_use"
    )

    # read in list of all regions (NE, NW, SE, SW, Central)
    region_categories = model_settings.get(
        "REGION_CATEGORIES", {}
    )  # reading in category-specific things

    # list to append all result files to
    choices_list = []
    # run model for each purpose separately
    for tour_purpose, tours_segment in ldt_tours_merged.groupby(segment_column_name):
        # get tour purpose in lowercase
        if tour_purpose.startswith("longdist_"):
            tour_purpose = tour_purpose[9:]
        tour_purpose = tour_purpose.lower()

        # only estimate on the segment that was estimated to go on
        # an external trip
        choosers = tours_segment[tours_segment.internal_external == LDT_IE_EXTERNAL]

        # logic if there are no choosers
        if choosers.empty:
            choices_list.append(
                pd.Series(-1, index=tours_segment.index, name=colname).to_frame()
            )
            continue

        # list to append all region-level results to
        region_choices_list = []
        # estimate for people in each region separately (due to different probability distributions)
        for region_category in region_categories:
            # the name of the region
            region = region_category["NAME"]

            # only consider the people on the current region to estimate
            region_choosers = choosers[choosers["LDTdistrict"] == region]

            logger.info(
                "ldt_external_destchoice tour_type '%s' region '%s' (%s tours)"
                % (
                    tour_purpose,
                    region,
                    len(region_choosers.index),
                )
            )

            # logic if there are no region choosers
            if region_choosers.empty:
                choices_list.append(
                    pd.Series(-1, index=region_choosers.index, name=colname).to_frame()
                )
                continue

            if estimator:
                estimator.write_model_settings(model_settings, model_settings_file_name)
                estimator.write_spec(model_settings)
                # estimator.write_coefficients(coefficients_df, model_settings)
                estimator.write_choosers(choosers)

            # create the probability distribution to draw from
            prob_list = np.zeros(len(external_probabilities))

            for i, taz in enumerate(external_probabilities.index):
                prob_list[i] = external_probabilities.loc[taz][region]
            # prob_list[-1] = 1 - np.sum(prob_list[:-1])

            pr = np.broadcast_to(
                prob_list, (len(region_choosers.index), len(external_probabilities))
            ).copy()
            df = pd.DataFrame(
                pr, index=region_choosers.index, columns=external_probabilities.index
            )

            # for each chooser, block off destinations that would be less than 50 miles away from
            # their respective origins
            for i in df.index:
                origin = ldt_tours_merged.loc[i, "home_zone_id"]
                df.loc[i] = df.loc[i] * np.where(
                    taz_dists.loc[origin, df.columns] >= 50, 1, 0
                )
            df = df.apply(lambda x: x / x.sum(), axis=1)

            # choose the exteranl destination choices; _ is discarded
            choices, _ = logit.make_choices(
                state, df, trace_choosers=state.settings.trace_hh_id
            )

            if estimator:
                estimator.write_choices(choices)
                choices = estimator.get_survey_values(choices, "persons", colname)
                estimator.write_override_choices(choices)
                estimator.end_estimation()

            # convert choices to chosen destinations
            destinations = pd.DataFrame(
                data=pd.Series(data=external_probabilities.index)[choices].values,
                index=region_choosers.index,
                columns=[colname],
            )

            destinations = destinations.reindex(region_choosers.index)

            # append the region-specific results to the region list
            region_choices_list.append(destinations)

        # merge all region choices into one, set the default to -1, and merge them into
        # the master choice list
        region_choices = pd.concat(region_choices_list)
        region_choices = region_choices.reindex(tours_segment.index).fillna(
            {colname: -1}, downcast="infer"
        )
        choices_list.append(region_choices)

    # combine all choices into one dataframe
    choices_df = pd.concat(choices_list)

    tracing.print_summary(
        "ldt_external_destchoice of all tour types",
        choices_df[choices_df[colname] != -1][colname],
        describe=True,
    )

    # merge into ldt_tours and replace the existing table in pipeline
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


def get_car_dist_skim(network_los, land_use, dim3, dist_key, model_area_key):
    """
    This method handles the logic for converting network_los to a skim for distances between
    TAZs and external TAZs
    """

    skim_dict = network_los.get_default_skim_dict()
    external_tazs = land_use[land_use[model_area_key] == 0].index
    if isinstance(skim_dict, SkimDict):
        skims = network_los.get_default_skim_dict().skim_data._skim_data
        key_dict = network_los.get_default_skim_dict().skim_dim3
        key = key_dict[dim3][dist_key]
        skim = skims[key]
        return pd.DataFrame(
            skim[:, external_tazs - 1],
            index=land_use.index,
            columns=external_tazs,
        )
    elif isinstance(skim_dict, SkimDataset):
        dist_array = skim_dict.dataset[dim3].sel(time_period=dist_key)
        return pd.DataFrame(
            dist_array.isel(dtaz=external_tazs).to_numpy(),
            index=dist_array.coords["otaz"],
            columns=external_tazs,
        )
