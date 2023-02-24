# ActivitySim
# See full license in LICENSE.txt.
import logging

import numpy as np
import pandas as pd
from activitysim.abm.models.util.tour_destination import destination_presample
from activitysim.abm.tables.size_terms import tour_destination_size_terms
from activitysim.core import config, los, simulate, tracing, workflow
from activitysim.core.interaction_sample import interaction_sample
from activitysim.core.interaction_sample_simulate import interaction_sample_simulate
from activitysim.core.util import reindex

from . import logsums as logsum
from .logsums import convert_time_periods_to_skim_periods

logger = logging.getLogger(__name__)
DUMP = False


class SizeTermCalculator:
    """
    convenience object to provide size_terms for a selector (e.g. non_mandatory)
    for various segments (e.g. tour_type or purpose)
    returns size terms for specified segment in df or series form
    """

    def __init__(self, state: workflow.State, size_term_selector, size_term_file=None):
        # do this once so they can request size_terms for various segments (tour_type or purpose)
        land_use = state.get_dataframe("land_use")
        if size_term_file is not None:
            f = state.filesystem.get_config_file_path(size_term_file)
            size_terms = pd.read_csv(f, comment="#", index_col="segment").fillna(0)
        else:
            size_terms = state.get_injectable("size_terms")
        self.destination_size_terms = tour_destination_size_terms(
            land_use, size_terms, size_term_selector
        )

        assert not self.destination_size_terms.isna().any(axis=None)

    # def omnibus_size_terms_df(self):
    #     return self.destination_size_terms

    def dest_size_terms_df(self, segment_name, trace_label):
        # return size terms as df with one column named 'size_term'
        # convenient if creating or merging with alts

        size_terms = self.destination_size_terms[[segment_name]].copy()
        size_terms.columns = ["size_term"]

        # FIXME - no point in considering impossible alternatives (where dest size term is zero)
        logger.debug(
            f"SizeTermCalculator dropping {(~(size_terms.size_term > 0)).sum()} "
            f"of {len(size_terms)} rows where size_term is zero for {segment_name}"
        )
        size_terms = size_terms[size_terms.size_term > 0]

        if len(size_terms) == 0:
            logger.warning(
                f"SizeTermCalculator: no zones with non-zero size terms for {segment_name} in {trace_label}"
            )

        return size_terms

    # def dest_size_terms_series(self, segment_name):
    #     # return size terms as as series
    #     # convenient (and no copy overhead) if reindexing and assigning into alts column
    #     return self.destination_size_terms[segment_name]


def _destination_sample(
    state: workflow.State,
    spec_segment_name,
    choosers,
    destination_size_terms,
    skims,
    estimator,
    model_settings,
    alt_dest_col_name,
    chunk_size,
    chunk_tag,
    trace_label,
    zone_layer=None,
    known_time_periods=None,
):
    model_spec = simulate.spec_for_segment(
        state,
        model_settings,
        spec_id="SAMPLE_SPEC",
        segment_name=spec_segment_name,
        estimator=estimator,
    )

    logger.info("running %s with %d tours", trace_label, len(choosers))

    sample_size = model_settings["SAMPLE_SIZE"]
    if state.settings.disable_destination_sampling or (
        estimator and estimator.want_unsampled_alternatives
    ):
        # FIXME interaction_sample will return unsampled complete alternatives with probs and pick_count
        logger.info(
            "Estimation mode for %s using unsampled alternatives short_circuit_choices"
            % (trace_label,)
        )
        sample_size = 0

    if known_time_periods is None:
        locals_d = {
            "skims": skims,
            "orig_col_name": skims.orig_key,  # added for sharrow flows
            "dest_col_name": skims.dest_key,  # added for sharrow flows
            "timeframe": "timeless",
        }
    else:
        locals_d = {
            "skims": skims,
            "orig_col_name": skims.orig_key,  # added for sharrow flows
            "dest_col_name": skims.dest_key,  # added for sharrow flows
            "timeframe": "tour",
            "odt_skims": known_time_periods["odt_skims"],
            "dot_skims": known_time_periods["dot_skims"],
        }
        skims = [
            skims,
            known_time_periods["odt_skims"],
            known_time_periods["dot_skims"],
        ]
    locals_d.update(state.get_global_constants())
    constants = config.get_model_constants(model_settings)
    if constants is not None:
        locals_d.update(constants)

    categories = state.get_global_constants()
    locals_d.update(categories)

    log_alt_losers = state.settings.log_alt_losers

    choices = interaction_sample(
        state,
        choosers,
        alternatives=destination_size_terms,
        sample_size=sample_size,
        alt_col_name=alt_dest_col_name,
        log_alt_losers=log_alt_losers,
        spec=model_spec,
        skims=skims,
        locals_d=locals_d,
        chunk_size=chunk_size,
        chunk_tag=chunk_tag,
        trace_label=trace_label,
        zone_layer=zone_layer,
    )

    # remember person_id in chosen alts so we can merge with persons in subsequent steps
    # (broadcasts person_id onto all alternatives sharing the same tour_id index value)
    choices["person_id"] = choosers.person_id

    return choices


def destination_sample(
    state: workflow.State,
    spec_segment_name,
    choosers,
    model_settings,
    network_los,
    destination_size_terms,
    estimator,
    chunk_size,
    trace_label,
    known_time_periods=None,
):
    chunk_tag = "tour_destination.sample"

    # create wrapper with keys for this lookup
    # the skims will be available under the name "skims" for any @ expressions
    skim_origin_col_name = model_settings["CHOOSER_ORIG_COL_NAME"]
    skim_dest_col_name = destination_size_terms.index.name
    # (logit.interaction_dataset suffixes duplicate chooser column with '_chooser')
    if skim_origin_col_name == skim_dest_col_name:
        skim_origin_col_name = f"{skim_origin_col_name}_chooser"

    skim_dict = network_los.get_default_skim_dict()
    skims = skim_dict.wrap(skim_origin_col_name, skim_dest_col_name)
    if known_time_periods is not None:
        odt_skims = skim_dict.wrap_3d(
            skim_origin_col_name, skim_dest_col_name, "out_period"
        )
        dot_skims = skim_dict.wrap_3d(
            skim_dest_col_name, skim_origin_col_name, "in_period"
        )
        known_time_periods_ = {
            "odt_skims": odt_skims,
            "dot_skims": dot_skims,
        }
        known_time_periods_.update(known_time_periods)

        # add in_periods and out_period columns to choosers...
        choosers = convert_time_periods_to_skim_periods(
            known_time_periods["in_period"],
            known_time_periods["out_period"],
            choosers,
            model_settings,
            "<tour_purpose>",
            network_los,
        )
    else:
        known_time_periods_ = None

    # the name of the dest column to be returned in choices
    alt_dest_col_name = model_settings["ALT_DEST_COL_NAME"]

    choices = _destination_sample(
        state,
        spec_segment_name,
        choosers,
        destination_size_terms,
        skims,
        estimator,
        model_settings,
        alt_dest_col_name,
        chunk_size,
        chunk_tag=chunk_tag,
        trace_label=trace_label,
        known_time_periods=known_time_periods_,
    )

    return choices


# temp column names for presampling
DEST_MAZ = "dest_MAZ"
DEST_TAZ = "dest_TAZ"
ORIG_TAZ = "TAZ"  # likewise a temp, but if already in choosers, we assume we can use it opportunistically


def aggregate_size_terms(dest_size_terms, network_los):
    #
    # aggregate MAZ_size_terms to TAZ_size_terms
    #

    MAZ_size_terms = dest_size_terms.copy()

    # add crosswalk DEST_TAZ column to MAZ_size_terms
    MAZ_size_terms[DEST_TAZ] = network_los.map_maz_to_taz(MAZ_size_terms.index)
    if MAZ_size_terms[DEST_TAZ].isna().any():
        raise ValueError("found NaN MAZ")

    # aggregate to TAZ
    TAZ_size_terms = MAZ_size_terms.groupby(DEST_TAZ).agg({"size_term": "sum"})
    TAZ_size_terms[DEST_TAZ] = TAZ_size_terms.index
    assert not TAZ_size_terms["size_term"].isna().any()

    #           size_term
    # dest_TAZ
    # 2              45.0
    # 3              44.0
    # 4              59.0

    # add crosswalk DEST_TAZ column to MAZ_size_terms
    # MAZ_size_terms = MAZ_size_terms.sort_values([DEST_TAZ, 'size_term'])  # maybe helpful for debugging
    MAZ_size_terms = MAZ_size_terms[[DEST_TAZ, "size_term"]].reset_index(drop=False)
    MAZ_size_terms = MAZ_size_terms.sort_values([DEST_TAZ, "zone_id"]).reset_index(
        drop=True
    )

    #       zone_id  dest_TAZ  size_term
    # 0        6097         2       10.0
    # 1       16421         2       13.0
    # 2       24251         3       14.0

    # print(f"TAZ_size_terms ({TAZ_size_terms.shape})\n{TAZ_size_terms}")
    # print(f"MAZ_size_terms ({MAZ_size_terms.shape})\n{MAZ_size_terms}")

    if np.issubdtype(TAZ_size_terms[DEST_TAZ], np.floating):
        raise TypeError("TAZ indexes are not integer")

    return MAZ_size_terms, TAZ_size_terms


def run_destination_sample(  # KEEP
    state: workflow.State,
    spec_segment_name,
    tours,
    persons_merged,
    model_settings,
    network_los,
    destination_size_terms,
    estimator,
    chunk_size,
    trace_label,
    known_time_periods=None,
):
    # FIXME - MEMORY HACK - only include columns actually used in spec (omit them pre-merge)
    chooser_columns = model_settings["SIMULATE_CHOOSER_COLUMNS"]
    persons_merged = persons_merged[
        [
            c
            for c in persons_merged.columns
            if c in chooser_columns and c not in tours.columns
        ]
    ]
    tours = tours[
        [c for c in tours.columns if c in chooser_columns or c == "person_id"]
    ]
    choosers = pd.merge(
        tours, persons_merged, left_on="person_id", right_index=True, how="left"
    )

    if "ldt_start_hour" not in choosers.columns:
        print()
    else:
        _temp = choosers["ldt_start_hour"]
        print(_temp)

    # interaction_sample requires that choosers.index.is_monotonic_increasing
    if not choosers.index.is_monotonic_increasing:
        logger.debug(
            f"run_destination_sample {trace_label} sorting choosers because not monotonic_increasing"
        )
        choosers = choosers.sort_index()

    # by default, enable presampling for multizone systems, unless they disable it in settings file
    pre_sample_taz = not (network_los.zone_system == los.ONE_ZONE)
    if pre_sample_taz and not state.settings.want_dest_choice_presampling:
        pre_sample_taz = False
        logger.info(
            f"Disabled destination zone presampling for {trace_label} "
            f"because 'want_dest_choice_presampling' setting is False"
        )

    if pre_sample_taz:
        logger.info(
            "Running %s destination_presample with %d tours" % (trace_label, len(tours))
        )

        choices = destination_presample(
            state,
            spec_segment_name,
            choosers,
            model_settings,
            network_los,
            destination_size_terms,
            estimator,
            trace_label,
        )
    else:
        choices = destination_sample(
            state,
            spec_segment_name,
            choosers,
            model_settings,
            network_los,
            destination_size_terms,
            estimator,
            chunk_size,
            trace_label,
            known_time_periods=known_time_periods,
        )

    # remember person_id in chosen alts so we can merge with persons in subsequent steps
    # (broadcasts person_id onto all alternatives sharing the same tour_id index value)
    choices["person_id"] = tours.person_id

    return choices


def run_destination_logsums(  # KEEP
    state: workflow.State,
    tour_purpose,
    persons_merged,
    destination_sample,
    model_settings,
    network_los,
    chunk_size,
    trace_label,
    in_period_col=None,
    out_period_col=None,
    duration_col=None,
):
    """
    add logsum column to existing tour_destination_sample table

    logsum is calculated by running the mode_choice model for each sample (person, dest_zone_id) pair
    in destination_sample, and computing the logsum of all the utilities

    +-----------+--------------+----------------+------------+----------------+
    | person_id | dest_zone_id | rand           | pick_count | logsum (added) |
    +===========+==============+================+============+================+
    | 23750     |  14          | 0.565502716034 | 4          |  1.85659498857 |
    +-----------+--------------+----------------+------------+----------------+
    + 23750     | 16           | 0.711135838871 | 6          | 1.92315598631  |
    +-----------+--------------+----------------+------------+----------------+
    + ...       |              |                |            |                |
    +-----------+--------------+----------------+------------+----------------+
    | 23751     | 12           | 0.408038878552 | 1          | 2.40612135416  |
    +-----------+--------------+----------------+------------+----------------+
    | 23751     | 14           | 0.972732479292 | 2          |  1.44009018355 |
    +-----------+--------------+----------------+------------+----------------+
    """

    logsum_settings = state.filesystem.read_model_settings(
        model_settings["LOGSUM_SETTINGS"]
    )

    chunk_tag = "tour_destination.logsums"

    # FIXME - MEMORY HACK - only include columns actually used in spec
    persons_merged = logsum.filter_chooser_columns(
        persons_merged, logsum_settings, model_settings
    )

    # merge persons into tours
    choosers = pd.merge(
        destination_sample,
        persons_merged,
        left_on="person_id",
        right_index=True,
        how="left",
    )

    logger.info("Running %s with %s rows", trace_label, len(choosers))

    state.tracing.dump_df(DUMP, persons_merged, trace_label, "persons_merged")
    state.tracing.dump_df(DUMP, choosers, trace_label, "choosers")

    logsums = logsum.compute_logsums(
        state,
        choosers,
        tour_purpose,
        logsum_settings,
        model_settings,
        network_los,
        chunk_size,
        chunk_tag,
        trace_label,
        in_period_col=in_period_col,
        out_period_col=out_period_col,
        duration_col=duration_col,
    )

    destination_sample["mode_choice_logsum"] = logsums

    return destination_sample


def run_destination_simulate(
    state: workflow.State,
    spec_segment_name,
    tours,
    persons_merged,
    destination_sample,
    want_logsums,
    model_settings,
    network_los,
    destination_size_terms,
    estimator,
    chunk_size,
    trace_label,
    known_time_periods=None,
):
    """
    run destination_simulate on tour_destination_sample
    annotated with mode_choice logsum to select a destination from sample alternatives
    """
    chunk_tag = "tour_destination.simulate"

    model_spec = simulate.spec_for_segment(
        state,
        model_settings,
        spec_id="SPEC",
        segment_name=spec_segment_name,
        estimator=estimator,
    )

    # FIXME - MEMORY HACK - only include columns actually used in spec (omit them pre-merge)
    chooser_columns = model_settings["SIMULATE_CHOOSER_COLUMNS"]
    persons_merged = persons_merged[
        [
            c
            for c in persons_merged.columns
            if c in chooser_columns and c not in tours.columns
        ]
    ]
    tours = tours[
        [c for c in tours.columns if c in chooser_columns or c == "person_id"]
    ]
    choosers = pd.merge(
        tours, persons_merged, left_on="person_id", right_index=True, how="left"
    )

    # interaction_sample requires that choosers.index.is_monotonic_increasing
    if not choosers.index.is_monotonic_increasing:
        logger.debug(
            f"run_destination_simulate {trace_label} sorting choosers because not monotonic_increasing"
        )
        choosers = choosers.sort_index()

    if estimator:
        estimator.write_choosers(choosers)

    alt_dest_col_name = model_settings["ALT_DEST_COL_NAME"]
    origin_col_name = model_settings["CHOOSER_ORIG_COL_NAME"]

    # alternatives are pre-sampled and annotated with logsums and pick_count
    # but we have to merge size_terms column into alt sample list
    destination_sample["size_term"] = reindex(
        destination_size_terms.size_term, destination_sample[alt_dest_col_name]
    )

    state.tracing.dump_df(DUMP, destination_sample, trace_label, "alternatives")

    constants = config.get_model_constants(model_settings)

    logger.info("Running tour_destination_simulate with %d persons", len(choosers))

    # create wrapper with keys for this lookup - in this case there is a home_zone_id in the choosers
    # and a zone_id in the alternatives which get merged during interaction
    # the skims will be available under the name "skims" for any @ expressions
    skim_dict = network_los.get_default_skim_dict()
    skims = skim_dict.wrap(origin_col_name, alt_dest_col_name)

    if known_time_periods is None:
        locals_d = {
            "skims": skims,
            "orig_col_name": skims.orig_key,  # added for sharrow flows
            "dest_col_name": skims.dest_key,  # added for sharrow flows
            "timeframe": "timeless",
        }
    else:
        # (logit.interaction_dataset suffixes duplicate chooser column with '_chooser')
        if origin_col_name == alt_dest_col_name:
            origin_col_name = f"{origin_col_name}_chooser"

        odt_skims = skim_dict.wrap_3d(origin_col_name, alt_dest_col_name, "out_period")
        dot_skims = skim_dict.wrap_3d(alt_dest_col_name, origin_col_name, "in_period")
        locals_d = {
            "skims": skims,
            "orig_col_name": skims.orig_key,  # added for sharrow flows
            "dest_col_name": skims.dest_key,  # added for sharrow flows
            "timeframe": "tour",
            "odt_skims": odt_skims,
            "dot_skims": dot_skims,
            # "out_period": known_time_periods["out_period"],
            # "in_period": known_time_periods["in_period"],
        }
        skims = [
            skims,
            odt_skims,
            dot_skims,
        ]
        # add in_periods and out_period columns to choosers...
        choosers = convert_time_periods_to_skim_periods(
            known_time_periods["in_period"],
            known_time_periods["out_period"],
            choosers,
            model_settings,
            "<tour_purpose>",
            network_los,
        )

    if constants is not None:
        locals_d.update(constants)

    state.tracing.dump_df(DUMP, choosers, trace_label, "choosers")

    log_alt_losers = state.settings.log_alt_losers

    choices = interaction_sample_simulate(
        state,
        choosers,
        destination_sample,
        spec=model_spec,
        choice_column=alt_dest_col_name,
        log_alt_losers=log_alt_losers,
        want_logsums=want_logsums,
        skims=skims,
        locals_d=locals_d,
        chunk_size=chunk_size,
        chunk_tag=chunk_tag,
        trace_label=trace_label,
        trace_choice_name="destination",
        estimator=estimator,
    )

    if not want_logsums:
        # for consistency, always return a dataframe with canonical column name
        assert isinstance(choices, pd.Series)
        choices = choices.to_frame("choice")

    return choices


def run_tour_destination(
    state: workflow.State,
    tours: pd.DataFrame,
    persons_merged: pd.DataFrame,
    want_logsums,
    want_sample_table,
    model_settings,
    network_los,
    estimator,
    chunk_size,
    trace_hh_id,
    trace_label,
    in_period_col=None,
    out_period_col=None,
    duration_col=None,
):
    size_term_calculator = SizeTermCalculator(
        state,
        model_settings["SIZE_TERM_SELECTOR"],
        model_settings.get("SIZE_TERM_PATH", None),
    )

    # maps segment names to compact (integer) ids
    segments = model_settings["SEGMENTS"]

    chooser_segment_column = model_settings.get("CHOOSER_SEGMENT_COLUMN_NAME", None)
    if chooser_segment_column is None:
        assert (
            len(segments) == 1
        ), f"CHOOSER_SEGMENT_COLUMN_NAME not specified in model_settings to slice SEGMENTS: {segments}"

    known_time_periods = {}
    if in_period_col is not None:
        known_time_periods["in_period"] = in_period_col
    if out_period_col is not None:
        known_time_periods["out_period"] = out_period_col

    choices_list = []
    sample_list = []
    for segment_name in segments:
        if isinstance(segment_name, (tuple, list)) and len(segment_name) == 2:
            # when segments are 2-tuples, they give segment_name for
            # destination segmenting and tour_purpose for mode segmenting
            segment_name, tour_purpose = segment_name
        else:
            tour_purpose = segment_name

        segment_trace_label = tracing.extend_trace_label(trace_label, segment_name)

        if chooser_segment_column is not None:
            choosers = tours[tours[chooser_segment_column] == segment_name]
        else:
            choosers = tours.copy()

        # Note: size_term_calculator omits zones with impossible alternatives (where dest size term is zero)
        segment_destination_size_terms = size_term_calculator.dest_size_terms_df(
            segment_name, segment_trace_label
        )

        if choosers.shape[0] == 0:
            logger.info(
                "%s skipping segment %s: no choosers", trace_label, segment_name
            )
            continue

        # - destination_sample
        spec_segment_name = segment_name  # spec_segment_name is segment_name
        location_sample_df = run_destination_sample(
            state,
            spec_segment_name,
            choosers,
            persons_merged,
            model_settings,
            network_los,
            segment_destination_size_terms,
            estimator,
            chunk_size=chunk_size,
            trace_label=tracing.extend_trace_label(segment_trace_label, "sample"),
            known_time_periods=known_time_periods,
        )

        # - destination_logsums
        location_sample_df = run_destination_logsums(
            state,
            tour_purpose,
            persons_merged,
            location_sample_df,
            model_settings,
            network_los,
            chunk_size=chunk_size,
            trace_label=tracing.extend_trace_label(segment_trace_label, "logsums"),
            in_period_col=in_period_col,
            out_period_col=out_period_col,
            duration_col=duration_col,
        )

        # - destination_simulate
        spec_segment_name = segment_name  # spec_segment_name is segment_name
        choices = run_destination_simulate(
            state,
            spec_segment_name,
            choosers,
            persons_merged,
            destination_sample=location_sample_df,
            want_logsums=want_logsums,
            model_settings=model_settings,
            network_los=network_los,
            destination_size_terms=segment_destination_size_terms,
            estimator=estimator,
            chunk_size=chunk_size,
            trace_label=tracing.extend_trace_label(segment_trace_label, "simulate"),
            known_time_periods=known_time_periods,
        )

        choices_list.append(choices)

        if want_sample_table:
            # FIXME - sample_table
            location_sample_df.set_index(
                model_settings["ALT_DEST_COL_NAME"], append=True, inplace=True
            )
            sample_list.append(location_sample_df)
        else:
            # del this so we dont hold active reference to it while run_location_sample is creating its replacement
            del location_sample_df

    if len(choices_list) > 0:
        choices_df = pd.concat(choices_list)
    else:
        # this will only happen with small samples (e.g. singleton) with no (e.g.) school segs
        logger.warning("%s no choices", trace_label)
        choices_df = pd.DataFrame(columns=["choice", "logsum"])

    if len(sample_list) > 0:
        save_sample_df = pd.concat(sample_list)
    else:
        # this could happen either with small samples as above, or if no saved sample desired
        save_sample_df = None

    return choices_df, save_sample_df
