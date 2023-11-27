import os
import random

import torch
import numpy as np
from scipy.signal import savgol_filter

from face_module.EmoTalk_release.demo import load_STF


class Mesh2Talk:
    def __init__(self, args):
        self.args = args
        self.stf_model = load_STF(args)
        self.result_path = args["ttf_param"]["result_path"]
        self.device = args["general"]["device"]
        
        self.eye1 = np.array(
            [
                0.36537236,
                0.950235724,
                0.95593375,
                0.916715622,
                0.367256105,
                0.119113259,
                0.025357503,
            ]
        )
        self.eye2 = np.array(
            [
                0.234776169,
                0.909951985,
                0.944758058,
                0.777862132,
                0.191071674,
                0.235437036,
                0.089163929,
            ]
        )
        self.eye3 = np.array(
            [
                0.870040774,
                0.949833691,
                0.949418545,
                0.695911646,
                0.191071674,
                0.072576277,
                0.007108896,
            ]
        )
        self.eye4 = np.array(
            [
                0.000307991,
                0.556701422,
                0.952656746,
                0.942345619,
                0.425857186,
                0.148335218,
                0.017659493,
            ]
        )

    def __call__(self,speech_array, file_name, result_path):
        os.makedirs(result_path, exist_ok=True)
        audio = torch.FloatTensor(speech_array).unsqueeze(0).to(self.device)
        level = torch.tensor([1]).to(self.device)
        person = torch.tensor([0]).to(self.device)
        prediction = self.stf_model.predict(audio, level, person)
        prediction = prediction.squeeze().detach().cpu().numpy()
        if self.args["ttf_param"]["post_processing"]:
            output = np.zeros((prediction.shape[0], prediction.shape[1]))
            for i in range(prediction.shape[1]):
                output[:, i] = savgol_filter(prediction[:, i], 5, 2)
            output[:, 8] = 0
            output[:, 9] = 0
            i = random.randint(0, 60)
            while i < output.shape[0] - 7:
                eye_num = random.randint(1, 4)
                if eye_num == 1:
                    output[i : i + 7, 8] = self.eye1
                    output[i : i + 7, 9] = self.eye1
                elif eye_num == 2:
                    output[i : i + 7, 8] = self.eye2
                    output[i : i + 7, 9] = self.eye2
                elif eye_num == 3:
                    output[i : i + 7, 8] = self.eye3
                    output[i : i + 7, 9] = self.eye3
                else:
                    output[i : i + 7, 8] = self.eye4
                    output[i : i + 7, 9] = self.eye4
                time1 = random.randint(60, 180)
                i = i + time1
            # np.save(os.path.join(result_path, "{}.npy".format(file_name)), output)
            np.save(os.path.join(result_path, "{}.npy".format(file_name)), output)
            print(f"\033[1;3;33mMesh Matrix Saved to : {result_path}/{file_name}.npy\033[0m")
            # with postprocessing (smoothing and blinking)
        else:
            np.save(os.path.join(result_path, "{}.npy".format(file_name)), prediction)
            print(f"\033[1;3;33mMesh Matrix Saved to : {result_path}/{file_name}.npy\033[0m")
            # without post-processing
