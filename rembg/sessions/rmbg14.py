import os
from typing import List

import numpy as np
import pooch
from PIL import Image
from PIL.Image import Image as PILImage

from .base import BaseSession

class RMBG14Session(BaseSession):
    """This class represents a session for using the RMBG14 model."""

    def predict(self, img: Image.Image, *args, **kwargs) -> [Image.Image]:
        """
        使用 RMBG-1.4 模型预测给定图像的遮罩。

        参数:
            img (PILImage): 输入图像。

        返回:
            List[PILImage]: 预测的遮罩。
        """
        # 根据配置文件进行图像预处理
        normalized_img = self.normalize(
            img,
            mean=(0.5, 0.5, 0.5),  # 使用配置文件中的均值
            std=(1, 1, 1),  # 使用配置文件中的标准差
            size=(1024, 1024)  # 调整到配置文件指定的大小
        )

        # 运行模型进行预测
        ort_outs = self.inner_session.run(None, normalized_img)

        # 处理输出，创建遮罩
        pred = ort_outs[0][:, 0, :, :]
        ma = np.max(pred)
        mi = np.min(pred)

        # 将预测结果归一化到0和1之间
        pred = (pred - mi) / (ma - mi)
        pred = np.squeeze(pred)  # 从形状中去除单一维度

        # 将结果转换为PIL图像，并调整回原始图像大小
        mask = Image.fromarray((pred * 255).astype("uint8"), mode="L")
        mask = mask.resize(img.size, Image.LANCZOS)

        return [mask]

    @classmethod
    def download_models(cls, *args, **kwargs):
        """
        Downloads the RMBG14 model.

        Returns:
            str: The path to the downloaded model.
        """
        fname = f"{cls.name(*args, **kwargs)}.onnx"
        pooch.retrieve(
            "https://huggingface.co/briaai/RMBG-1.4/resolve/main/onnx/model.onnx",
            None,
            fname=fname,
            path=cls.u2net_home(*args, **kwargs),
            progressbar=True,
        )

        return os.path.join(cls.u2net_home(*args, **kwargs), fname)

    @classmethod
    def name(cls, *args, **kwargs):
        """
        Returns the name of the RMBG14 model.

        Returns:
            str: The name of the model.
        """
        return "rmbg14"

    @classmethod
    def u2net_home(cls, *args, **kwargs):
        """
        Returns the home directory for the U2net models.

        Returns:
            str: The home directory path.
        """
        return os.path.expanduser(os.path.join("~", ".u2net"))

    @classmethod
    def checksum_disabled(cls, *args, **kwargs):
        """
        Indicates if checksum verification is disabled.

        Returns:
            bool: True if checksum verification is disabled, otherwise False.
        """
        return False
