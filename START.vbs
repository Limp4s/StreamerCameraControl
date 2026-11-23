Option Explicit
Dim shell, fso, folder, pythonw, app
Set shell = CreateObject("WScript.Shell")
Set fso = CreateObject("Scripting.FileSystemObject")
folder = fso.GetParentFolderName(WScript.ScriptFullName)
pythonw = folder & "\.venv\Scripts\pythonw.exe"
app = folder & "\app.py"

If Not fso.FileExists(pythonw) Then
    MsgBox "Please run INSTALL.bat once before starting StrimerCameraControl.", 48, "StrimerCameraControl"
ElseIf Not fso.FileExists(app) Then
    MsgBox "app.py was not found. Please extract the complete application folder.", 16, "StrimerCameraControl"
Else
    shell.CurrentDirectory = folder
    shell.Run Chr(34) & pythonw & Chr(34) & " " & Chr(34) & app & Chr(34), 0, False
End If
