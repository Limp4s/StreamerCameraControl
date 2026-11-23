# StrimerCameraControl

A Windows application for controlling your computer via gestures using a webcam. Features include hand tracking, optional face detection, a live preview, gesture-to-action mapping, and the ability to execute custom scripts.

## Requirements

- Windows 10 (22H2 recommended) or Windows 11, 64-bit.
- Python 3.12 x64 (the installer will automatically attempt to install Python 3.12 via Winget if it is not already present).
- A webcam.

## Installation and Launch

1. Download and extract the archive into a folder.
2. Run `INSTALL.bat` and wait for the setup to complete (1–2 minutes).
3. Launch the application using `START.vbs`.

## Custom Scripts

The application supports `.ahk`, `.exe`, `.bat`, and `.cmd` files. You can add files via the selection dialog or by dragging and dropping them into the **My scripts** list. For `.ahk` scripts, AutoHotkey must be installed, or the file association must be correctly configured.

## Troubleshooting

- **Physical camera fails to open:** Close other applications that may be using the webcam, check your Windows privacy/permission settings, or try a different camera index.
- **Virtual camera fails to start:** Re-run `INSTALL.bat` or `assets\virtual_camera\InstallDriver.bat` as an administrator, then restart both StrimerCameraControl and the target application.
- Do not extract the project into a folder path that contains Cyrillic characters or special symbols.

## Virtual Camera (OBS / Discord / Zoom)

1. The project includes the Unity Capture driver (a DirectShow filter), which registers in Windows under the name **StrimerCameraControl**. When you run `INSTALL.bat` for the first time, it will install the necessary Python dependencies and request administrator privileges to register the virtual camera. If a Windows UAC prompt appears, please grant permission for the installation. Once installed, restart OBS, Discord, and any other applications so they can refresh their list of available cameras.

2. In OBS, add a **Video Capture Device** source and select **StrimerCameraControl**. Choose the same device in Discord or Zoom. Note that the application itself will be using your physical webcam. Do not use the "OBS Virtual Camera" feature to receive frames from StrimerCameraControl; the OBS Virtual Camera is intended for outputting your OBS scene to other applications.

3. You can manually remove the driver by running `assets\virtual_camera\UninstallDriver.bat` (administrator privileges required). The driver files and license information are located in the `assets\virtual_camera` directory; specific license details can be found in `THIRD_PARTY_NOTICES.txt`.