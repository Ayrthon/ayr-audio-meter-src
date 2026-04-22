; Inno Setup 6 — https://jrsoftware.org/isinfo.php
; Build: install Inno Setup 6, run ..\installer\build-installer.ps1 from repo root (or ISCC this file after `cargo build --release`).
; Bump MyAppVersion when you bump version in Cargo.toml.

#define MyAppName      "AYR Audio Meter"
#define MyAppVersion   "1.6.6"
#define MyAppPublisher "Ayrthon"
#define MyAppURL       "https://ayrthon.com"
#define MyAppExeName   "AYR Audio Meter.exe"
; Installed beside the .exe so shortcuts use an explicit .ico (Shell often ignores icons embedded in some Rust/GUI exes).
#define MyAppIconFile  "AudioMeter.ico"

#define ReleaseExe     "..\target\release\audio-meter-rs.exe"

[Setup]
AppId={{B4C8F2E9-1D3A-5C7E-9B2D-8A6F4E1C3D70}}
AppName={#MyAppName}
AppVersion={#MyAppVersion}
AppVerName={#MyAppName} {#MyAppVersion}
AppPublisher={#MyAppPublisher}
AppPublisherURL={#MyAppURL}
AppSupportURL={#MyAppURL}
SetupIconFile={#MyAppIconFile}
DefaultDirName={autopf}\{#MyAppName}
DisableDirPage=no
DisableProgramGroupPage=yes
; admin install under Program Files + uninstall in Settings
PrivilegesRequired=admin
PrivilegesRequiredOverridesAllowed=dialog
ArchitecturesInstallIn64BitMode=x64compatible
OutputDir=..\dist
; File name includes the full brand (spaces → hyphens so it's web/email-safe).
OutputBaseFilename=AYR-Audio-Meter-Setup-{#MyAppVersion}
Compression=lzma2/max
SolidCompression=yes
WizardStyle=modern
UninstallDisplayIcon={app}\{#MyAppIconFile}
; If a previous version is running, the installer can't overwrite the .exe (Windows locks it).
; `CloseApplications=yes` uses the Restart Manager API to prompt the user to close the running
; copy automatically, then relaunches it after the install via `RestartApplications=yes`.
CloseApplications=yes
RestartApplications=yes

[Languages]
Name: "english"; MessagesFile: "compiler:Default.isl"

[Tasks]
Name: "desktopicon"; Description: "{cm:CreateDesktopIcon}"; GroupDescription: "{cm:AdditionalIcons}"; Flags: unchecked

[Files]
Source: "{#ReleaseExe}"; DestDir: "{app}"; DestName: "{#MyAppExeName}"; Flags: ignoreversion
Source: "{#MyAppIconFile}"; DestDir: "{app}"; DestName: "{#MyAppIconFile}"; Flags: ignoreversion

[Icons]
Name: "{autoprograms}\{#MyAppName}"; Filename: "{app}\{#MyAppExeName}"; IconFilename: "{app}\{#MyAppIconFile}"; IconIndex: 0
Name: "{autodesktop}\{#MyAppName}"; Filename: "{app}\{#MyAppExeName}"; IconFilename: "{app}\{#MyAppIconFile}"; IconIndex: 0; Tasks: desktopicon

[Run]
Filename: "{app}\{#MyAppExeName}"; Description: "{cm:LaunchProgram,{#StringChange(MyAppName, '&', '&&')}}"; Flags: nowait postinstall skipifsilent

[UninstallDelete]
; User clips under %LOCALAPPDATA%\audio-meter-rs\clips are left on disk on purpose.
