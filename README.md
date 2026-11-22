# StrimerCameraContro

A Windows program that enables gesture-based control; it includes a camera preview, hand recognition, optional face detection, gesture-to-action mapping, and support for custom scripts.

## Dependencies to download

Python 3.13 or higher

## Install / run

1. Extract this ZIP to a normal folder (not inside a protected system folder).
2. Run `INSTALL.bat` to install Python dependencies if needed.
3. Run `START.bat` or `START.vbs`.

## Warning!

The application needs a webcam. Face detector model is downloaded on first use when face detection is enabled and an internet connection is available.
The path to the project folder should not contain Cyrillic or special characters.

## Supported script types

`.ahk`, `.exe`, `.bat`, `.cmd`. AutoHotkey scripts require AutoHotkey or a working Windows file association.

## 0.9.5

- User scripts registered in **My scripts** can be selected in gesture bindings and refreshed when the script list changes.
- Per-binding hold threshold (0.35 s default; independent per gesture binding).
- Configurable persistence of the hand skeleton overlay after tracking loss (0.5 s default).
- Old configuration files are migrated by adding missing settings without resetting existing settings.
- Frame-processing errors stop the camera processing loop and display an error instead of repeatedly invoking a broken frame callback.
