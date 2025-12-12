import shutil

import kagglehub


def download_data(dataset_url: str, target_dir: str) -> None:
    data_path = kagglehub.dataset_download(handle=dataset_url)

    shutil.copytree(src=data_path, dst=target_dir, dirs_exist_ok=True)
    shutil.rmtree(data_path)
