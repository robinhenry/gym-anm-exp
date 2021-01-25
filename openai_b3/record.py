import os
import base64
from pathlib import Path
from IPython import display as ipythondisplay
from stable_baselines3.common.vec_env import VecVideoRecorder

from openai_b3.hyperparameters import ENV_ID, LOG_DIR, ALGO

os.system("Xvfb :1 -screen 0 1024x768x24 &")
os.environ['DISPLAY'] = ':1'


def show_videos(video_path='', prefix=''):
    """
    Taken from https://github.com/eleurent/highway-env

    :param video_path: (str) Path to the folder containing videos
    :param prefix: (str) Filter the video, showing only the only starting with this prefix
    """
    html = []
    for mp4 in Path(video_path).glob("{}*.mp4".format(prefix)):
        video_b64 = base64.b64encode(mp4.read_bytes())
        html.append('''<video alt="{}" autoplay 
                       loop controls style="height: 400px;">
                       <source src="data:video/mp4;base64,{}" type="video/mp4" />
                       </video>'''.format(mp4, video_b64.decode('ascii')))
    ipythondisplay.display(ipythondisplay.HTML(data="<br>".join(html)))


def record_video(eval_env, model, video_length=500, prefix='', video_folder='videos/'):
    """
    :param eval_env: (DummyVecEnv)
    :param model: (RL model)
    :param video_length: (int)
    :param prefix: (str)
    :param video_folder: (str)
    """
    # Start the video at step=0 and record 500 steps
    eval_env = VecVideoRecorder(eval_env, video_folder=video_folder,
                                record_video_trigger=lambda step: step == 0, video_length=video_length,
                                name_prefix=prefix)

    obs = eval_env.reset()
    for _ in range(video_length):
        action, _ = model.predict(obs)
        obs, _, _, _ = eval_env.step(action)

    # Close the video recorder
    eval_env.close()


if __name__ == '__main__':

    RECORD_STEPS = 500

    # Load trained model.
    model = ALGO.load(LOG_DIR + '/best_model', verbose=1)
    print("n_steps =", model.n_steps)

    video_folder = LOG_DIR + '/videos'
    record_video(ENV_ID, model, video_length=RECORD_STEPS, prefix='', video_folder=video_folder)
    show_videos(video_folder, prefix='')
