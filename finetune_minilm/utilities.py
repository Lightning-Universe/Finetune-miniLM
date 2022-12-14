import warnings

import lightning as L


def warn_if_drive_not_empty(drive: L.app.storage.Drive):
    if drive.list():
        warnings.warn(
            "Drive is not empty! This may result in wrong logging behaviour if your app doesn't have a built-in resume"
            " mechanism. Consider deleting the .lightning file and restarting the app or giving it a new name with the"
            " --name flag of 'lightning run app'."
        )
