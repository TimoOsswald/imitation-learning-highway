"""
Rule-based expert policy for highway-env's DiscreteMetaAction space.

Used to generate demonstrations for training a Behavior Cloning policy. 
The policy follows the vehicle ahead at a safe distance, attempts lane 
changes when blocked, and otherwise targets a cruising speed.
"""

import numpy as np

SAFE_DISTANCE = 0.15                # minimal distance to vehicle infront
SAME_LANE_THRESEHOLD = 0.1          # y-coordinate tolerance for the same lane
TARGET_LANE_THERESEHOLD = 0.1       # y-coordinate tolerance for target lane
TARGET_SPEED = 0.37                 # aim for this velocity
MATCHED_SPEED_THRESEHOLD = 0.05     # tolerance for the same velocity as the vehicle infront
LANE_WIDTH = 0.25                   # normalized y-distance of a line (4-line-highway)
LANE_CHANGE_SAFE_DISTANCE = 0.25    # x-safety-distance to change lanes


def expert_policy(obs, action_indexes, available_actions):
    """
    Checks the current situation in the environment and decides what action is next for safe and fast driving

    Args: 
        obs: kinematics array of vehicle in situation, shape [V, F]
        action_indexes: dictionary of action-names to index ['LANE_LEFT' : 0, ...]
        available_actions: list of currently valid actions (used for edge cases)

    Returns: 
        int: the chosen action index for the ego-vehicle
    """

    closest_vehicle = None
    ego_vehicle = obs[0]

    for i in range(1, len(obs)):
        vehicle = obs[i]
        if  (vehicle[0] == 1                            and      # check for real car
            abs(vehicle[2]) <= SAME_LANE_THRESEHOLD      and      # check for same lane
            vehicle[1] >= 0):                                    # check for car infront
            if closest_vehicle is None or vehicle[1] <= closest_vehicle[1]:
                closest_vehicle = vehicle     # remember closest vehicle on my lane

    # decide action on closest_vehicle speed
    if closest_vehicle is not None:
        ego_vehicle_vx = ego_vehicle[3]
        closest_vx = closest_vehicle[3]

        # slower vehicle infront --> try lane change or slow down
        if closest_vehicle[1] < SAFE_DISTANCE and closest_vx < 0:
            lane_change_action = try_lane_change(obs, action_indexes, available_actions)
            if lane_change_action is not None:
                return lane_change_action
            return action_indexes['SLOWER']

        elif ego_vehicle_vx > TARGET_SPEED:
            return action_indexes['IDLE']  
        
        # try lane change or keep speed
        elif closest_vehicle[1] < SAFE_DISTANCE and abs(closest_vx) < MATCHED_SPEED_THRESEHOLD:
            lane_change_action = try_lane_change(obs, action_indexes, available_actions)
            if lane_change_action is not None:
                 return lane_change_action
            return action_indexes['IDLE']
        
        elif ego_vehicle_vx < TARGET_SPEED:
            return action_indexes['FASTER']
        
        else:
            return action_indexes['IDLE']
        
    else:
        return action_indexes['IDLE']


def try_lane_change(obs, action_indexes, available_actions):
    """
    Attempts a lane change to the left, then to the right, if either is safe and currently available

    Args: 
        obs: kinematics array of vehicle in situation, shape [V, F]
        action_indexes: dictionary of action-names to index ['LANE_LEFT' : 0, ...]
        available_actions: list of currently valid actions (used for edge cases)
    
    Returns: 
        int: the chosen action index (LANE_LEFT or LANE_RIGHT), or None if no lane change is safe
    """
    if is_lane_safe(obs, -LANE_WIDTH) and action_indexes['LANE_LEFT'] in available_actions:
        return action_indexes['LANE_LEFT']
    elif is_lane_safe(obs, + LANE_WIDTH) and action_indexes['LANE_RIGHT'] in available_actions:
        return action_indexes['LANE_RIGHT']
    else:
        return None


def is_lane_safe(obs, target_y):
    """
    Checks for the safety of a lane change by scanning for nearby vehilcles on this lane

    Args: 
        obs: kinematics array of vehicle in situation, shape [V, F]
        target_y: target y-coordinate relatively to the ego-vehicle

    Returns: 
        bool: true if the lane is safe and false if not
    """
    is_safe = True
    for i in range(1, len(obs)):
        vehicle = obs[i]
        if (vehicle[0] == 1                                                                                and
            (target_y - TARGET_LANE_THERESEHOLD) < vehicle[2] and (target_y + TARGET_LANE_THERESEHOLD) > vehicle[2] and
            abs(vehicle[1]) < LANE_CHANGE_SAFE_DISTANCE):
            is_safe = False

    return is_safe