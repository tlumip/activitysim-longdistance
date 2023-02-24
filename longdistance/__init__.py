from pathlib import Path
from typing import Any

import activitysim.abm  # load regular ActivitySim abm components # noqa: F401
from activitysim.core.workflow import State

from . import (
    ldt_accessibility,
    ldt_create_longdist_trips,
    ldt_external_destchoice,
    ldt_external_mode_choice,
    ldt_internal_external,
    ldt_internal_mode_choice,
    ldt_internal_tour_destination,
    ldt_pattern,
    ldt_pattern_household,
    ldt_pattern_person,
    ldt_scheduling,
    ldt_tour_gen,
    ldt_tour_gen_household,
    ldt_tour_gen_person,
)


def make_longdist_model(
    working_dir: Path = None, settings: dict[str, Any] = None, **kwargs
):
    """Initialize a State object for a new long-distance model."""
    state = State.make_default(working_dir=working_dir, settings=settings, **kwargs)
    state.extend.declare_table(
        "longdist_tours",
        traceable=True,
        random_channel=True,
        index_name="longdist_tour_id",
    )
    state.import_extensions(__path__[0])
    return state
