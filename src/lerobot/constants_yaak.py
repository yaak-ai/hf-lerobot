
# Both waypoints and speed tokenized "state-style" with embedding projection

# Waypoints the 1st embedded elemeny because of the language instruction
# "Follow ... given the waypoints and speed"
OBS_STATE = "observation.state.waypoints"

# Speed is the 2nd embedded element: merged or separate
OBS_STATE_VEHICLE = "observation.state.vehicle"


OBS_IMAGE = "observation.images.front_left"
OBS_IMAGES = "observation.images.front_left"

# turn signal is currently not used
ACTION = "action.continuous"
