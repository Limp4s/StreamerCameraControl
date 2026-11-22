Set shell = CreateObject("WScript.Shell")
Set fso = CreateObject("Scripting.FileSystemObject")
folder = fso.GetParentFolderName(WScript.ScriptFullName)
pythonw = folder & "\.venv\Scripts\pythonw.exe"
app = folder & "\app.py"
If Not fso.FileExists(pythonw) Then
  MsgBox "First, run INSTALL.bat to install the components.", 48, "Gesture Control"
Else
  shell.CurrentDirectory = folder
  shell.Run Chr(34) & pythonw & Chr(34) & " " & Chr(34) & app & Chr(34), 0, False
End If
