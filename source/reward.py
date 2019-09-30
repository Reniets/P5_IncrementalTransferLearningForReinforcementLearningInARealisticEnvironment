
class Reward:
    def __init__(self, carlaEnv):
        self.carlaEnv = carlaEnv

    def calcReward(self):
        reward = 0

        reward += self._rewardDriveFarOnRoad() * 1.00  # Reward (points pr. meter driven on road pr. tire on road)
        reward += self._rewardAvoidGrass() * 0.25  # Penalty (Points pr. tick on grass pr. tire on road)
        # reward += self._rewardSubGoal()             * weight
        # reward += self._rewardDriveShortOnGrass()   * 1.50  # Penalty
        # reward += self._rewardReturnToRoad()        * 1.00  # Reward / Penalty
        # reward += self._rewardStayOnRoad()          * 0.05  # Reward
        # reward += self._rewardDriveFast()         * 0.10

        self._updateLastTickVariables()  # MUST BE LAST THING IN REWARD FUNCTION

        return reward

    def _rewardStayOnRoad(self):
        return self.carlaEnv.wheelsOnRoad()

    def _rewardAvoidGrass(self):
        return -self.carlaEnv.wheelsOnGrass

    def _rewardDriveFast(self):
        return (self.carlaEnv.getCarVelocity() / 50) * self._rewardStayOnRoad()

    def _rewardDriveFar(self):
        return self.carlaEnv.metersTraveledSinceLastTick()

    def _rewardDriveFarOnRoad(self):
        return self._rewardDriveFar() * self.carlaEnv.wheelsOnRoad()

    def _rewardDriveShortOnGrass(self):
        return -(self._rewardDriveFar() * self.carlaEnv.wheelsOnGrass)

    def _rewardReturnToRoad(self):
        wheel_diff = self._wheelsOnRoadDiffFromLastTick()

        if wheel_diff > 0:
            return wheel_diff * 25
        elif wheel_diff < 0:
            return wheel_diff * (-50)
        else:
            return 0

    def _updateLastTickVariables(self):
        self.carlaEnv.car_last_tick_pos = self.carlaEnv.vehicle.get_location()
        self.carlaEnv.car_last_tick_wheels_on_road = self.carlaEnv.wheelsOnRoad()

        # Returns the difference from current tick to last tick of how many wheels are currently on the road
        # Also updates last to current tick

    def _wheelsOnRoadDiffFromLastTick(self):
        last = self.carlaEnv.car_last_tick_wheels_on_road
        current = self.carlaEnv.wheelsOnRoad()
        diff = current - last

        return diff

    # Returns the reward for the current state
    def calcRewardOld(self):
        reward = 0
        speed = self.carlaEnv.getCarVelocity()
        expected_speed = 20

        reward += (speed / expected_speed) * self.carlaEnv.wheelsOnRoad()
        reward -= self.carlaEnv.wheelsOnGrass

        return reward
