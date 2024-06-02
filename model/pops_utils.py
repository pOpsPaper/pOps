
from typing import List, Tuple

import torch
from torch import nn

def preprocess(image_a: torch.Tensor, image_b: torch.Tensor, image_encoder: nn.Module, clip_mean: torch.Tensor,
            clip_std: torch.Tensor, should_drop_cond: List[Tuple[bool, bool]] = None, concat_hidden_states=None,
            image_list=None):
    with torch.no_grad():
        image_list = [] if image_list is None else image_list
        additional_list = []
        if image_a is not None:
            additional_list.append(image_a)
        if image_b is not None:
            additional_list.append(image_b)
        image_list = additional_list + image_list
        embeds_list = []
        for image in image_list:
            # If already is vector skip encoder
            if len(image.shape) == 2:
                image_embeds = image
            else:
                encoder_outs = image_encoder(image, output_hidden_states=False)
                image_embeds = encoder_outs.image_embeds
            image_embeds = (image_embeds - clip_mean) / clip_std
            embeds_list.append(image_embeds.unsqueeze(1))
        if should_drop_cond is not None:
            for b_ind in range(embeds_list[0].shape[0]):
                should_drop_a, should_drop_b = should_drop_cond[b_ind]
                if should_drop_a:
                    embeds_list[0][b_ind] = torch.zeros_like(embeds_list[0][b_ind])
                if should_drop_b and image_b is not None:
                    embeds_list[1][b_ind] = torch.zeros_like(embeds_list[1][b_ind])
        if concat_hidden_states is not None:
            embeds_list.append(concat_hidden_states)
        out_hidden_states = torch.concat(embeds_list, dim=1)

        image_embeds = torch.zeros_like(embeds_list[0].squeeze(1))

    return image_embeds, out_hidden_states
