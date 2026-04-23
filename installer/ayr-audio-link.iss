; Inno Setup 6 — AYR Audio Link (sender)
;
; Build (from repo root):
;   powershell -ExecutionPolicy Bypass -File installer\build-link-installer.ps1
;
; The sender is a free companion to AYR Audio Meter. It captures audio from a
; chosen input or WASAPI loopback device and streams it to any AYR Audio Meter
; running on the same LAN. The meter discovers senders via mDNS and shows
; them as `[NET] <name>` entries in its device dropdown.

#define MyAppName        "AYR Audio Link"
#define MyAppVersion     "0.1.0"
#define MyAppPublisher   "Ayrthon"
#define MyAppURL         "https://ayrthon.com"
#define MyAppExeName     "AYR Audio Link.exe"
; Reuse the meter's icon for visual consistency — both apps ship under the
; same "AYR Audio" brand. A separate per-product icon can be dropped in here
; later without touching the build script.
#define MyAppIconFile    "AudioMeter.ico"

#define ReleaseExe       "..\target\release\ayr-audio-link.exe"

[Setup]
; Unique to the sender. Never reuse the Meter's AppId here — Inno uses it to
; key upgrades / uninstalls, and collisions would have the two products
; clobber each other in Add/Remove Programs.
AppId={{3F7A1C52-8D4E-4B9C-8E3A-6F2D9B5A7E41}}
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
; Admin install under Program Files; required so our `netsh` firewall rule
; can land in the machine-wide profile (per-user rules would only apply to
; the installing user's sessions).
PrivilegesRequired=admin
PrivilegesRequiredOverridesAllowed=dialog
ArchitecturesInstallIn64BitMode=x64compatible
OutputDir=..\dist
OutputBaseFilename=AYR-Audio-Link-Setup-{#MyAppVersion}
Compression=lzma2/max
SolidCompression=yes
WizardStyle=modern
UninstallDisplayIcon={app}\{#MyAppIconFile}
; Same Restart Manager integration the Meter uses — if a previous copy is
; running the installer prompts to close it, upgrades, then relaunches.
CloseApplications=yes
RestartApplications=yes

[Languages]
Name: "english"; MessagesFile: "compiler:Default.isl"

[Tasks]
Name: "desktopicon"; Description: "{cm:CreateDesktopIcon}"; GroupDescription: "{cm:AdditionalIcons}"; Flags: unchecked
Name: "firewallrule"; Description: "Allow AYR Audio Link through Windows Firewall (required for the Meter to connect over the LAN)"; GroupDescription: "Network:"; Flags: checkedonce

[Files]
Source: "{#ReleaseExe}"; DestDir: "{app}"; DestName: "{#MyAppExeName}"; Flags: ignoreversion
Source: "{#MyAppIconFile}"; DestDir: "{app}"; DestName: "{#MyAppIconFile}"; Flags: ignoreversion

[Icons]
Name: "{autoprograms}\{#MyAppName}"; Filename: "{app}\{#MyAppExeName}"; IconFilename: "{app}\{#MyAppIconFile}"; IconIndex: 0
Name: "{autodesktop}\{#MyAppName}"; Filename: "{app}\{#MyAppExeName}"; IconFilename: "{app}\{#MyAppIconFile}"; IconIndex: 0; Tasks: desktopicon

[Run]
; --- Windows Firewall rule ---------------------------------------------
; Remove any stale rule from a previous install first so we never end up
; with duplicates after upgrades (netsh silently appends otherwise). The
; delete step is `runhidden` + allowed to fail (initial install has nothing
; to delete).
Filename: "{sys}\netsh.exe"; \
    Parameters: "advfirewall firewall delete rule name=""AYR Audio Link"""; \
    Flags: runhidden; \
    StatusMsg: "Updating Windows Firewall rule..."; \
    Tasks: firewallrule
; Open inbound TCP on the default Link port. The sender is a local-LAN
; broadcast tool — we intentionally scope to `profile=private,domain` so
; a roaming user on a coffee-shop Wi-Fi doesn't accept unsolicited
; connections from strangers. Users who need `public` can open the rule
; manually; the risk case is unusual enough that the default here should
; stay conservative.
Filename: "{sys}\netsh.exe"; \
    Parameters: "advfirewall firewall add rule name=""AYR Audio Link"" dir=in action=allow program=""{app}\{#MyAppExeName}"" enable=yes profile=private,domain protocol=TCP localport=45451 description=""Allow AYR Audio Link to accept meter connections on the local network"""; \
    Flags: runhidden; \
    StatusMsg: "Configuring Windows Firewall..."; \
    Tasks: firewallrule
; --- Launch on finish --------------------------------------------------
Filename: "{app}\{#MyAppExeName}"; Description: "{cm:LaunchProgram,{#StringChange(MyAppName, '&', '&&')}}"; Flags: nowait postinstall skipifsilent

[UninstallRun]
; Best-effort rule cleanup on uninstall. `runhidden` + no `Check` — if the
; rule is already gone we don't care.
Filename: "{sys}\netsh.exe"; Parameters: "advfirewall firewall delete rule name=""AYR Audio Link"""; Flags: runhidden; RunOnceId: "LinkFirewallRemove"
