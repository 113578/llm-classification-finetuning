import fire

from llm_classification_finetuning.data.download_data import download_data
from llm_classification_finetuning.train import train


if __name__ == "__main__":
    fire.Fire(
        component={
            "download_data": download_data,
            "train": train,
        }
    )
