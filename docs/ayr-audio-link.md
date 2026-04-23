# AYR Audio Link

**AYR Audio Link** is a free companion app for [AYR Audio Meter](https://ayrthon.com). It captures audio from one PC and streams it over the local network to the Meter running on another PC, where it shows up as a normal device in the picker — tagged `[NET]`.

Typical use: run the Meter on your main workstation, and use Link on a second PC (a DAW rig, a vocal-booth laptop, a second studio machine) to see that signal on the main monitor without cables.

## Requirements

- Windows 10 or 11 on both PCs.
- Both PCs on the same LAN (same Wi-Fi network, same office subnet, or a wired switch).
- AYR Audio Meter **2.0.0 or newer** on the receiving PC.

## Install

1. Download `AYR-Audio-Link-Setup-<version>.exe` from the [Releases page](https://github.com/Ayrthon/ayr-audio-link/releases/latest).
2. Run the installer on the PC that has the audio source you want to share. The installer will ask for admin rights — this is so it can add a Windows Firewall rule allowing incoming connections from the Meter. If you skip the firewall task, you'll need to add the rule manually or the Meter won't be able to connect.
3. Launch **AYR Audio Link** from the Start menu.

## Use

On the sending PC (the one running AYR Audio Link):

1. Pick a device from the **Source** dropdown:
   - `[IN]` entries are microphones, line-ins, and interfaces.
   - `[OUT]` entries are loopbacks of output devices — pick this if you want to capture what's playing on that PC (e.g. a DAW, a browser tab, system audio).
2. Click **Start broadcasting**.

On the receiving PC (the one running AYR Audio Meter):

1. Open the device dropdown in the Meter's left panel.
2. You should see a **NETWORK** section with an entry like `[NET] <sender hostname>`. If you don't, click the dropdown a couple of seconds later — mDNS discovery takes a moment.
3. Pick the network sender. Press **Start metering**. You should see levels moving within a second.

The Link window can be minimized to the taskbar and will keep broadcasting in the background.

## Troubleshooting

**Meter doesn't see the Link sender.**

- Make sure both PCs are on the same Wi-Fi SSID / subnet. Corporate guest networks sometimes isolate clients — switch to the main network.
- On the sender, confirm Windows Firewall has the `AYR Audio Link` rule. You can check under **Control Panel → Windows Defender Firewall → Allowed apps**. If it's missing, rerun the installer and make sure the "Allow AYR Audio Link through Windows Firewall" task is checked.
- Some enterprise networks block mDNS / Bonjour traffic. In those setups you'll need the network admin to allow UDP 5353 between the two PCs.

**Meter connects, but levels don't move.**

- On the sender, confirm the level bar is moving. If it isn't, the device you picked isn't producing audio — try a different source.
- If the sender's level bar moves but the Meter shows silence, stop + restart the stream on the Meter side. This forces a fresh handshake.

**Audio sounds glitchy.**

- Link is a **metering** transport, not a monitoring transport. It's designed for visual level checking, not for listening. On a congested Wi-Fi network you may hear occasional crackles; this is normal and has no effect on metering accuracy (the jitter buffer compensates for timing drift before the DSP sees samples).
- Move to a wired connection for the most stable experience.

**Can I run multiple senders?**

- Yes. Each sender advertises itself independently, and the Meter's NETWORK section lists all of them. Today the Meter can only stream from one at a time; multi-source mixing is on the roadmap.

## FAQ

**Is there a macOS version?** Not yet. It's on the roadmap.

**Does Link need a license?** No. Link is free, unauthenticated, and has no trial. The paid license is for the Meter.

**What's the network protocol?** Uncompressed interleaved `f32` PCM over TCP on port 45451, with an mDNS advertisement on `_ayraudiolink._tcp.local.` for discovery. Details are in `crates/ayr-audio-link-core/src/wire.rs` if you want to audit it.
