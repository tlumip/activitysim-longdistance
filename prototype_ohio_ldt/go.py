import longdistance

state = longdistance.make_longdist_model()
state.import_extensions("extensions")
state.filesystem.persist_sharrow_cache()
state.logging.config_logger()

# from activitysim.abm.models.accessibility import compute_accessibility

state.run.all(
    # resume_after="_"
)
