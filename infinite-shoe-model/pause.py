import msvcrt
from stable_baselines3.common.callbacks import BaseCallback

class KeyboardPauseCallback(BaseCallback):
    def _on_step(self) -> bool:
        if msvcrt.kbhit():
            key = msvcrt.getwch()
            if key == 'p':
                print("\n⏸ Paused. Press any key to resume...")
                msvcrt.getwch()  # wait for any key
                print("▶ Resuming...")
        return True