[Setup]
AppId=QMTDesktop-Carlos0902
AppName=QMT Desktop
AppVersion=0.1.0
AppPublisher=QMT Desktop
AppPublisherURL=https://github.com/Carlos0902/QMT-Desktop
DefaultDirName={localappdata}\Programs\QMT Desktop
DefaultGroupName=QMT Desktop
PrivilegesRequired=lowest
ArchitecturesAllowed=x64compatible
ArchitecturesInstallIn64BitMode=x64compatible
MinVersion=10.0
OutputDir=..\dist
OutputBaseFilename=QMT-Desktop-0.1.0-Windows-x64-Setup
Compression=lzma2/fast
SolidCompression=yes
WizardStyle=modern
UninstallDisplayIcon={app}\QMT Desktop.exe
CloseApplications=yes

[Files]
Source: "..\dist-windows\*"; DestDir: "{app}"; Flags: ignoreversion recursesubdirs createallsubdirs

[Icons]
Name: "{group}\QMT Desktop"; Filename: "{app}\QMT Desktop.exe"
Name: "{autodesktop}\QMT Desktop"; Filename: "{app}\QMT Desktop.exe"; Tasks: desktopicon

[Tasks]
Name: desktopicon; Description: "Create a desktop shortcut"; Flags: unchecked

[Run]
Filename: "{app}\QMT Desktop.exe"; Description: "Launch QMT Desktop"; Flags: nowait postinstall skipifsilent
