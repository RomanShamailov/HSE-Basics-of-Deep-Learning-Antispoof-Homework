import warnings

import hydra
import kagglehub
import pandas as pd
import torch
from hydra.utils import instantiate
from omegaconf import OmegaConf
from tqdm.auto import tqdm

from src.datasets.data_utils import get_dataloaders
from src.trainer import Trainer
from src.utils.init_utils import set_random_seed, setup_saving_and_logging

warnings.filterwarnings("ignore", category=UserWarning)


@hydra.main(version_base=None, config_path="src/configs", config_name="baseline")
def main(config):
    """
    Main script for training. Instantiates the model, optimizer, scheduler,
    metrics, logger, writer, and dataloaders. Runs Trainer to train and
    evaluate the model.

    Args:
        config (DictConfig): hydra experiment config.
    """
    set_random_seed(config.trainer.seed)

    project_config = OmegaConf.to_container(config)
    logger = setup_saving_and_logging(config)
    writer = instantiate(config.writer, logger, project_config)

    if config.trainer.device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    else:
        device = config.trainer.device

    # setup data_loader instances
    # batch_transforms should be put on device
    dataloaders, batch_transforms = get_dataloaders(config, device)

    # build model architecture, then print to console
    model = instantiate(config.model).to(device)
    logger.info(model)

    # get function handles of loss and metrics
    loss_function = instantiate(config.loss_function).to(device)
    metrics = instantiate(config.metrics)

    # build optimizer, learning rate scheduler
    trainable_params = filter(lambda p: p.requires_grad, model.parameters())
    optimizer = instantiate(config.optimizer, params=trainable_params)
    step_size = 10 * len(dataloaders["train"])  # multiply by 0.5 every 10 epochs
    lr_scheduler = instantiate(
        config.lr_scheduler, optimizer=optimizer, step_size=step_size
    )

    # epoch_len = number of iterations for iteration-based training
    # epoch_len = None or len(dataloader) for epoch-based training
    epoch_len = config.trainer.get("epoch_len")

    trainer = Trainer(
        model=model,
        criterion=loss_function,
        metrics=metrics,
        optimizer=optimizer,
        lr_scheduler=lr_scheduler,
        config=config,
        device=device,
        dataloaders=dataloaders,
        epoch_len=epoch_len,
        logger=logger,
        writer=writer,
        batch_transforms=batch_transforms,
        skip_oom=config.trainer.get("skip_oom", True),
    )

    trainer.train()

    # saving model outputs after the last epoch to csv for grading
    print("Saving to csv...")
    eval_id = pd.read_csv(
        kagglehub.dataset_download("awsaf49/asvpoof-2019-dataset")
        + "/LA/LA/ASVspoof2019_LA_cm_protocols/ASVspoof2019.LA.cm.eval.trl.txt",
        sep=" ",
        header=None,
        usecols=[1, 4],
    )[1]
    res = []
    model.eval()
    for i, batch in tqdm(
        enumerate(dataloaders["test"]), total=len(dataloaders["test"])
    ):
        spectrogram = batch["spectrogram"].to(device)
        with torch.no_grad():
            output = model(spectrogram)  # calculate model output
        probs = torch.nn.functional.softmax(output["logits"], dim=1)
        for j in range(spectrogram.shape[0]):
            res.append([eval_id[64 * i + j], float(probs[j][1])])
    df = pd.DataFrame(res)
    df.to_csv("rgshamailov.csv", index=False, header=False)


if __name__ == "__main__":
    main()
