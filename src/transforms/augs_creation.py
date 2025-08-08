import torch
import torchaudio


class AugsCreation:
    def __call__(self, audio):
        aug_num = torch.randint(
            low=0, high=3, size=(1,)
        ).item()  # choose 1 random aug from augs
        augs = [
            lambda x: x,
            lambda x: (x + torch.distributions.Normal(0, 0.01).sample(x.size())).clamp_(
                -1, 1
            ),
            lambda x: torchaudio.transforms.Vol(0.25)(x),
        ]

        return augs[aug_num](audio)
