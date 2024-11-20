from huggingface_hub import HfFileSystem, hf_hub_download


def path_to_file_name(path: str):
    return "/".join(path.split("/")[2:])


def path_to_repo_id(path: str):
    return "/".join(path.split("/")[:2])


def hf_hub_download_folder(path: str):
    fs = HfFileSystem()
    files = fs.ls(path, detail=True)

    repo_id = path_to_repo_id(path)
    folder_path = path_to_file_name(path)

    folder_local_path = ""
    for file in files:
        if file["type"] == "file":
            file_name = path_to_file_name(file["name"])
            local_path = hf_hub_download(
                repo_id=repo_id,
                filename=file_name,
                repo_type="model",
            )

            if not folder_local_path:
                folder_local_path = (
                    local_path.replace(file_name, "") + folder_path
                )

        if file["type"] == "directory":
            hf_hub_download_folder(file["name"])

    return folder_local_path
