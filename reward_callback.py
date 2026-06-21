from stable_baselines3.common.callbacks import BaseCallback

class RewardCallback(BaseCallback):

    def __init__(self):
        super().__init__()
        self.episode_rewards = []

    def _on_step(self):

        for info in self.locals["infos"]:

            if "episode" in info:
                self.episode_rewards.append(
                    info["episode"]["r"]
                )

        return True