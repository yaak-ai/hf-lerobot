from collections import deque
from typing import Any

import torch
from PIL import Image
from torch import Tensor
from transformers import AutoModelForCausalLM, AutoTokenizer, CLIPImageProcessor
from transformers.image_utils import ImageInput

from lerobot.constants_yaak import ACTION
from lerobot.policies.fastvla.configuration_fastvla import FastVLAConfig
from lerobot.policies.pretrained import PreTrainedPolicy


def resize_and_pad_to_square(
    img: Image, target_res: int, pad_color: tuple = (0, 0, 0)
) -> Image:
    img = img.convert("RGB")
    w, h = img.size

    scale = target_res / max(w, h)

    new_w = int(w * scale)
    new_h = int(h * scale)

    img_resized = img.resize((new_w, new_h), Image.LANCZOS)

    square_img = Image.new(img.mode, (target_res, target_res), pad_color)

    paste_y = target_res - new_h
    square_img.paste(img_resized, (0, paste_y))

    return square_img


class FastImageTokenizer(CLIPImageProcessor):
    def __init__(
        self,
        do_resize: bool = True,  # noqa: FBT001, FBT002
        size: dict[str, int] | None = None,
        resample: Image.Resampling = Image.Resampling.BICUBIC,
        do_center_crop: bool = True,  # noqa: FBT001, FBT002
        crop_size: dict[str, int] | None = None,
        do_rescale: bool = True,  # noqa: FBT001, FBT002
        rescale_factor: float = 1 / 255,
        do_normalize: bool = True,  # noqa: FBT001, FBT002
        image_mean: float | list[float] | None = None,
        image_std: float | list[float] | None = None,
        do_convert_rgb: bool = True,  # noqa: FBT001, FBT002
        **kwargs,  # noqa: ANN003
    ):
        super().__init__(
            do_resize,
            size,
            resample,
            do_center_crop,
            crop_size,
            do_rescale,
            rescale_factor,
            do_normalize,
            image_mean,
            image_std,
            do_convert_rgb,
            **kwargs,
        )

    def preprocess(self, images: ImageInput, **kwargs) -> Tensor:
        images = resize_and_pad_to_square(images, self.size["shortest_edge"])
        return super().preprocess(images, return_tensors="pt", **kwargs)["pixel_values"]


class FastVLAPolicy(PreTrainedPolicy):
    config_class = FastVLAConfig
    name = "fastvla"

    def __init__(self, config: FastVLAConfig) -> None:
        super().__init__(config)

        self.config = config
        self.tok = AutoTokenizer.from_pretrained(
            config.vlm_model_name, trust_remote_code=True
        )
        vlm: Any = AutoModelForCausalLM.from_pretrained(
            config.vlm_model_name,
            torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
            device_map="auto",
            trust_remote_code=True,
        )
        self.model = vlm.get_model().get_vision_tower().vision_tower

    def forward(self, concat_images: Tensor | list) -> Tensor:
        return self.model(concat_images, return_image_embeddings=True)[
            "image_embeddings"
        ]

    def get_optim_params(self) -> dict:
        return self.model.parameters()

    def _get_action_chunk(
        self, batch: dict[str, Tensor], noise: Tensor | None = None
    ) -> Tensor:
        msg = "TBD"
        raise NotImplementedError(msg)

    @torch.no_grad()
    def predict_action_chunk(
        self, batch: dict[str, Tensor], noise: Tensor | None = None
    ) -> Tensor:
        msg = "TBD"
        raise NotImplementedError(msg)

    def reset(self):
        """This should be called whenever the environment is reset."""
        self._queues = {
            ACTION: deque(maxlen=self.config.n_action_steps),
        }

    @torch.no_grad()
    def select_action(
        self, batch: dict[str, Tensor], noise: Tensor | None = None
    ) -> Tensor:
        msg = "TBD"
        raise NotImplementedError(msg)
