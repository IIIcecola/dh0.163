

import json 
import os 
import subprocess

ctrl_expressions = [
    "CTRL_expressions_browDownL",
    "CTRL_expressions_browDownR",
    "CTRL_expressions_browLateralL",
    "CTRL_expressions_browLateralR",
    "CTRL_expressions_browRaiseInL",
    "CTRL_expressions_browRaiseInR",
    "CTRL_expressions_browRaiseOuterL",
    "CTRL_expressions_browRaiseOuterR",
    "CTRL_expressions_eyeBlinkL",
    "CTRL_expressions_eyeBlinkR",
    "CTRL_expressions_eyeWidenL",
    "CTRL_expressions_eyeWidenR",
    "CTRL_expressions_eyeSquintInnerL",
    "CTRL_expressions_eyeSquintInnerR",
    "CTRL_expressions_eyeCheekRaiseL",
    "CTRL_expressions_eyeCheekRaiseR",
    "CTRL_expressions_eyeFaceScrunchL",
    "CTRL_expressions_eyeFaceScrunchR",
    "CTRL_expressions_eyeLookUpL",
    "CTRL_expressions_eyeLookUpR",
    "CTRL_expressions_eyeLookDownL",
    "CTRL_expressions_eyeLookDownR",
    "CTRL_expressions_eyeLookLeftL",
    "CTRL_expressions_eyeLookLeftR",
    "CTRL_expressions_eyeLookRightL",
    "CTRL_expressions_eyeLookRightR",
    "CTRL_expressions_noseWrinkleL",
    "CTRL_expressions_noseWrinkleR",
    "CTRL_expressions_noseNostrilDepressL",
    "CTRL_expressions_noseNostrilDepressR",
    "CTRL_expressions_noseNostrilDilateL",
    "CTRL_expressions_noseNostrilDilateR",
    "CTRL_expressions_noseNostrilCompressL",
    "CTRL_expressions_noseNostrilCompressR",
    "CTRL_expressions_noseNasolabialDeepenL",
    "CTRL_expressions_noseNasolabialDeepenR",
    "CTRL_expressions_mouthCheekBlowL",
    "CTRL_expressions_mouthCheekBlowR",
    "CTRL_expressions_mouthLeft",
    "CTRL_expressions_mouthRight",
    "CTRL_expressions_mouthUpperLipRaiseL",
    "CTRL_expressions_mouthUpperLipRaiseR",
    "CTRL_expressions_mouthLowerLipDepressL",
    "CTRL_expressions_mouthLowerLipDepressR",
    "CTRL_expressions_mouthCornerPullL",
    "CTRL_expressions_mouthCornerPullR",
    "CTRL_expressions_mouthStretchL",
    "CTRL_expressions_mouthStretchR",
    "CTRL_expressions_mouthDimpleL",
    "CTRL_expressions_mouthDimpleR",
    "CTRL_expressions_mouthCornerDepressL",
    "CTRL_expressions_mouthCornerDepressR",
    "CTRL_expressions_mouthLipsPurseUL",
    "CTRL_expressions_mouthLipsPurseUR",
    "CTRL_expressions_mouthLipsPurseDL",
    "CTRL_expressions_mouthLipsPurseDR",
    "CTRL_expressions_mouthLipsTowardsUL",
    "CTRL_expressions_mouthLipsTowardsUR",
    "CTRL_expressions_mouthLipsTowardsDL",
    "CTRL_expressions_mouthLipsTowardsDR",
    "CTRL_expressions_mouthFunnelUL",
    "CTRL_expressions_mouthFunnelUR",
    "CTRL_expressions_mouthFunnelDL",
    "CTRL_expressions_mouthFunnelDR",
    "CTRL_expressions_mouthLipsTogetherUL",
    "CTRL_expressions_mouthLipsTogetherUR",
    "CTRL_expressions_mouthLipsTogetherDL",
    "CTRL_expressions_mouthLipsTogetherDR",
    "CTRL_expressions_mouthUpperLipBiteL",
    "CTRL_expressions_mouthUpperLipBiteR",
    "CTRL_expressions_mouthLowerLipBiteL",
    "CTRL_expressions_mouthLowerLipBiteR",
    "CTRL_expressions_mouthLipsTightenUL",
    "CTRL_expressions_mouthLipsTightenUR",
    "CTRL_expressions_mouthLipsTightenDL",
    "CTRL_expressions_mouthLipsTightenDR",
    "CTRL_expressions_mouthLipsPressL",
    "CTRL_expressions_mouthLipsPressR",
    "CTRL_expressions_mouthSharpCornerPullL",
    "CTRL_expressions_mouthSharpCornerPullR",
    "CTRL_expressions_mouthStickyUC",
    "CTRL_expressions_mouthStickyUINL",
    "CTRL_expressions_mouthStickyUINR",
    "CTRL_expressions_mouthStickyUOUTL",
    "CTRL_expressions_mouthStickyUOUTR",
    "CTRL_expressions_mouthStickyDC",
    "CTRL_expressions_mouthStickyDINL",
    "CTRL_expressions_mouthStickyDINR",
    "CTRL_expressions_mouthStickyDOUTL",
    "CTRL_expressions_mouthStickyDOUTR",
    "CTRL_expressions_mouthLipsPushUL",
    "CTRL_expressions_mouthLipsPushUR",
    "CTRL_expressions_mouthLipsPushDL",
    "CTRL_expressions_mouthLipsPushDR",
    "CTRL_expressions_mouthLipsPullUL",
    "CTRL_expressions_mouthLipsPullUR",
    "CTRL_expressions_mouthLipsPullDL",
    "CTRL_expressions_mouthLipsPullDR",
    "CTRL_expressions_mouthLipsThinUL",
    "CTRL_expressions_mouthLipsThinUR",
    "CTRL_expressions_mouthLipsThinDL",
    "CTRL_expressions_mouthLipsThinDR",
    "CTRL_expressions_mouthLipsThickUL",
    "CTRL_expressions_mouthLipsThickUR",
    "CTRL_expressions_mouthLipsThickDL",
    "CTRL_expressions_mouthLipsThickDR",
    "CTRL_expressions_mouthCornerSharpenUL",
    "CTRL_expressions_mouthCornerSharpenUR",
    "CTRL_expressions_mouthCornerSharpenDL",
    "CTRL_expressions_mouthCornerSharpenDR",
    "CTRL_expressions_mouthCornerRounderUL",
    "CTRL_expressions_mouthCornerRounderUR",
    "CTRL_expressions_mouthCornerRounderDL",
    "CTRL_expressions_mouthCornerRounderDR",
    "CTRL_expressions_mouthUpperLipShiftLeft",
    "CTRL_expressions_mouthLowerLipShiftLeft",
    "CTRL_expressions_mouthLowerLipShiftRight",
    "CTRL_expressions_mouthUpperLipRollInL",
    "CTRL_expressions_mouthUpperLipRollInR",
    "CTRL_expressions_mouthUpperLipRollOutL",
    "CTRL_expressions_mouthUpperLipRollOutR",
    "CTRL_expressions_mouthLowerLipRollInL",
    "CTRL_expressions_mouthLowerLipRollInR",
    "CTRL_expressions_mouthLowerLipRollOutL",
    "CTRL_expressions_mouthLowerLipRollOutR",
    "CTRL_expressions_mouthCornerUpL",
    "CTRL_expressions_mouthCornerUpR",
    "CTRL_expressions_mouthCornerDownL",
    "CTRL_expressions_mouthCornerDownR",
    "CTRL_expressions_jawOpen",
    "CTRL_expressions_jawLeft",
    "CTRL_expressions_jawRight",
    "CTRL_expressions_jawBack",
    "CTRL_expressions_jawChinRaiseDL",
    "CTRL_expressions_jawChinRaiseDR",
    "CTRL_expressions_jawOpenExtreme"
]



# 在UE把这一步做了
def split_and_save_json(data, output_dir):
    """
    将大 JSON 按 key 拆分保存到独立文件中，
    并提取每个动画中所有曲线的最大 time 值作为 time_long。

    Args:
        data (dict): 原始 JSON 数据
        output_dir (str): 保存目录
    """
    os.makedirs(output_dir, exist_ok=True)

    for key, value in data.items():
        max_time = 0.0

        # 遍历每个子曲线，取 time 的最大值
        for sub_key, sub_val in value.items():
            if isinstance(sub_val, dict) and "time" in sub_val:
                t = sub_val["time"]
                if isinstance(t, list) and len(t) >= 2:
                    # 取列表中最大值（兼容 time = [0, end] 或 time = [t1, t2, ...] 的情况）
                    local_max = max(t)
                    max_time = max(max_time, local_max)

        # 将最大时间值写入
        value["time_long"] = max_time

        save_path = os.path.join(output_dir, f"{key}.json")
        with open(save_path, "w", encoding="utf-8") as f:
            json.dump(value, f, ensure_ascii=False, indent=2)

        print(f"✅ 已保存: {save_path} (time_long={max_time})")

def extract_wav_from_mp4(
    input_dir: str,
    output_dir: str,
    sample_rate: int = 16000,
    channels: int = 1
):
    """
    批量提取 input_dir 下所有 mp4 的音频为 wav
    """
    os.makedirs(output_dir, exist_ok=True)

    for filename in os.listdir(input_dir):
        if not filename.lower().endswith(".mp4"):
            continue

        input_path = os.path.join(input_dir, filename)
        wav_name = os.path.splitext(filename)[0] + ".wav"
        output_path = os.path.join(output_dir, wav_name)

        cmd = [
            "ffmpeg",
            "-y",                # 覆盖输出
            "-i", input_path,
            "-vn",               # 不要视频
            "-ac", str(channels),
            "-ar", str(sample_rate),
            "-f", "wav",
            output_path
        ]

        try:
            subprocess.run(
                cmd,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                check=True
            )
            print(f"✅ 已生成: {output_path}")
        except subprocess.CalledProcessError:
            print(f"❌ 失败: {input_path}")


def add_time_timelong(input_dir):
    json_paths = os.listdir(input_dir)
    for j_path in json_paths:
        json_path = os.path.join(input_dir,j_path)
        with open(json_path, "r", encoding="utf-8") as f:
            data = json.load(f)

            AnimName = data['AnimName']
            value = data[AnimName]
            max_time = 0.0
            for sub_key, sub_val in value.items():
                if isinstance(sub_val, dict) and "time" in sub_val:
                    t = sub_val["time"]
                    if isinstance(t, list) and len(t) >= 2:
                        # 取列表中最大值（兼容 time = [0, end] 或 time = [t1, t2, ...] 的情况）
                        local_max = max(t)
                        max_time = max(max_time, local_max)
            

            data["time_long"] = max_time
            # 3️⃣ 保存
            with open(json_path, "w", encoding="utf-8") as f:
                json.dump(data, f, ensure_ascii=False, indent=4)

        print("✅ {} JSON 已更新".format(json_path))


class UE_CurvesManager:
    def __init__(self,json_path,key,fps):
        json_path = os.path.join(json_path,key+'.json')
        with open(json_path, "r") as f:
            self.data = json.load(f) 
        self.time_long = self.data['time_long']
        self.fps = fps
    
    def get_match_data(self,key,frame_index):
        """
        输入时间 t，返回对应插值的值
        times: list[float]
        values: list[float]
        """
        t = frame_index/self.fps
        times = self.data[key]["time"]
        values = self.data[key]["value"]


        if not times:
            return None
        if t <= times[0]:
            return values[0]
        if t >= times[-1]:
            return values[-1]

        # 遍历找到区间
        for i in range(len(times) - 1):
            t0, t1 = times[i], times[i + 1]
            if t0 <= t <= t1:
                v0, v1 = values[i], values[i + 1]
                alpha = (t - t0) / (t1 - t0)
                return v0 * (1 - alpha) + v1 * alpha




if __name__ == '__main__':
    ''' '''
    # input_dir = './Dataset/zhuboshuolianbo/video'
    # output_dir = './Dataset/zhuboshuolianbo/wav'
    # extract_wav_from_mp4(input_dir, output_dir)
    input_dir = './Dataset/zhuboshuolianbo/json'
    add_time_timelong(input_dir)



































