from pathlib import Path
import gdown
import zipfile

#pseudocode
#  it checks if images exist
#    if they do, return
#    if not, images are downloaded
def download_calibration_images(test_images_path, file_id):
    # check if images exists in base temp path
    test_data_path = Path(test_images_path)
    if test_data_path.absolute().exists():
        return
    else:
        test_data_path.mkdir(parents=True)

    downloaded_folder_path = str(test_data_path.joinpath("camera-calibration.zip"))
    # download images
    gdown.download(id=file_id, output=str(downloaded_folder_path))
    with zipfile.ZipFile(downloaded_folder_path, 'r') as zip_ref:
        zip_ref.extractall(str(test_data_path))

