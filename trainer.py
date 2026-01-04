from stable_baselines3 import PPO

class LiveTrainer:
    def __init__(self, env):
        self.env = env
        self.model = PPO(
            "MlpPolicy",
            env,
            learning_rate=3e-4,
            n_steps=256,
            batch_size=64,
            gamma=0.99,
            verbose=0
        )
        self.obs, _ = env.reset()

    def step(self):
        action, _ = self.model.predict(self.obs, deterministic=False)
        self.obs, reward, done, _, info = self.env.step(action)

        if done:
            self.obs, _ = self.env.reset()

        return reward, info
