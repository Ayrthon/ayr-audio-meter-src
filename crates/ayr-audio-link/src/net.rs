//! TCP server that accepts meter connections and streams PCM.
//!
//! Lifecycle:
//!
//! 1. UI calls [`Server::start`] with a port.
//! 2. Server binds and spawns an accept thread; each accepted connection
//!    is handed off to a per-peer worker thread.
//! 3. UI calls [`Server::submit_frame`] for every block the capture
//!    thread produces. The server fans those frames out to every
//!    connected peer.
//! 4. UI calls [`Server::update_format`] any time the capture format
//!    changes (or on first block). That triggers existing peers to be
//!    dropped so they reconnect with the new handshake.
//!
//! A simple broadcast channel would be nicer here but `std::sync::mpsc`
//! doesn't support multiple consumers and pulling in `crossbeam` for one
//! feature is overkill. We use a per-peer bounded `VecDeque` protected by
//! a mutex — pedestrian but it handles the 1-to-N pattern cleanly.

use std::io::{BufWriter, Write};
use std::net::{IpAddr, Ipv4Addr, Shutdown, SocketAddr, TcpListener, TcpStream};
use std::sync::Arc;
use std::sync::atomic::{AtomicBool, AtomicU32, Ordering};
use std::thread::{self, JoinHandle};
use std::time::{Duration, Instant, SystemTime};

use anyhow::{Context, Result};
use parking_lot::Mutex;

use ayr_audio_link_core::wire::{self, Handshake};

use crate::identity::IdentityStore;

/// Maximum frames queued per connected peer before we start dropping.
/// ~2 seconds of audio at 10 ms/frame — enough to ride a WiFi blip but
/// short enough that a truly stalled peer doesn't balloon memory.
const PER_PEER_QUEUE_LIMIT: usize = 200;

/// What the UI sees about each connected meter.
#[derive(Clone, Debug)]
pub struct PeerInfo {
    pub addr: SocketAddr,
    /// Human name the meter sent in its greeting (future protocol field;
    /// empty in v1 since only the sender greets).
    pub label: String,
    pub connected_at: SystemTime,
    pub frames_sent: u64,
}

#[derive(Clone, Debug)]
pub struct ServerStatus {
    pub listening_on: Option<SocketAddr>,
    pub peers: Vec<PeerInfo>,
}

pub struct Server {
    stop: Arc<AtomicBool>,
    accept_thread: Option<JoinHandle<()>>,
    shared: Arc<ServerShared>,
    handshake: Arc<Mutex<Option<Handshake>>>,
    identity: Arc<IdentityStore>,
    display_name: Arc<Mutex<String>>,
}

/// Cloneable, `Send`/`Sync` handle used by the audio callback to push
/// frames into the server without borrowing `&Server`. It holds clones of
/// the same `Arc`s the server owns, so lifetimes and shutdown semantics
/// are unchanged.
#[derive(Clone)]
pub struct ServerHandle {
    shared: Arc<ServerShared>,
    handshake: Arc<Mutex<Option<Handshake>>>,
    display_name: Arc<Mutex<String>>,
}

struct ServerShared {
    peers: Mutex<Vec<Arc<PeerState>>>,
    listening_on: Mutex<Option<SocketAddr>>,
    next_peer_id: AtomicU32,
}

struct PeerState {
    id: u32,
    addr: SocketAddr,
    stop: AtomicBool,
    queue: Mutex<std::collections::VecDeque<FrameSlice>>,
    connected_at: SystemTime,
    frames_sent: AtomicU32,
    stream: Mutex<Option<TcpStream>>,
}

struct FrameSlice {
    pts_ns: u64,
    payload: Vec<u8>,
}

impl Server {
    pub fn new(identity: Arc<IdentityStore>) -> Self {
        let display_name = Arc::new(Mutex::new(identity.display_name()));
        Self {
            stop: Arc::new(AtomicBool::new(false)),
            accept_thread: None,
            shared: Arc::new(ServerShared {
                peers: Mutex::new(Vec::new()),
                listening_on: Mutex::new(None),
                next_peer_id: AtomicU32::new(1),
            }),
            handshake: Arc::new(Mutex::new(None)),
            identity,
            display_name,
        }
    }

    /// Return a cloneable handle the audio callback can drive. Separate
    /// type so the callback is `Send` without lifetime ties to `Server`.
    pub fn handle(&self) -> ServerHandle {
        ServerHandle {
            shared: self.shared.clone(),
            handshake: self.handshake.clone(),
            display_name: self.display_name.clone(),
        }
    }

    /// Set the display name the handshake will advertise. Called by the
    /// UI whenever the user edits it.
    pub fn set_display_name(&self, name: String) {
        *self.display_name.lock() = name;
    }

    /// Latest handshake as seen by connecting peers. Returns `None` until
    /// the audio callback publishes a format.
    pub fn current_handshake(&self) -> Option<Handshake> {
        self.handshake.lock().clone()
    }

    /// Bind on `0.0.0.0:<port>` and start accepting. Safe to call again
    /// after `stop()` to restart on a different port.
    pub fn start(&mut self, port: u16) -> Result<()> {
        self.stop();
        self.stop = Arc::new(AtomicBool::new(false));

        let listener = TcpListener::bind(SocketAddr::new(IpAddr::V4(Ipv4Addr::UNSPECIFIED), port))
            .with_context(|| format!("bind 0.0.0.0:{port}"))?;
        listener
            .set_nonblocking(false)
            .context("listener set_nonblocking(false)")?;
        let local = listener.local_addr().ok();
        *self.shared.listening_on.lock() = local;

        let stop = self.stop.clone();
        let shared = self.shared.clone();
        let handshake = self.handshake.clone();
        let identity = self.identity.clone();

        let handle = thread::Builder::new()
            .name("ayr-link-accept".into())
            .spawn(move || accept_loop(listener, stop, shared, handshake, identity))
            .context("spawn accept thread")?;
        self.accept_thread = Some(handle);
        Ok(())
    }

    pub fn stop(&mut self) {
        self.stop.store(true, Ordering::Relaxed);
        // Tear down peers so their writer threads unblock on shutdown.
        for p in self.shared.peers.lock().drain(..) {
            p.stop.store(true, Ordering::Relaxed);
            if let Some(s) = p.stream.lock().take() {
                let _ = s.shutdown(Shutdown::Both);
            }
        }
        // The accept thread wakes on listener drop — but our listener is
        // owned inside the thread. Easiest shutdown: dial ourselves to
        // break the blocking `accept()`.
        if let Some(addr) = self.shared.listening_on.lock().take() {
            let _ = TcpStream::connect_timeout(&addr, Duration::from_millis(250));
        }
        if let Some(h) = self.accept_thread.take() {
            let _ = h.join();
        }
    }

    /// UI pushes the current capture format here whenever it changes. The
    /// server stashes it for the handshake and drops existing peers so
    /// they reconnect with the new parameters. Returns `true` if this
    /// changed the format.
    pub fn update_format(&self, hs: Handshake) -> bool {
        let mut guard = self.handshake.lock();
        if guard.as_ref() == Some(&hs) {
            return false;
        }
        *guard = Some(hs);
        for p in self.shared.peers.lock().drain(..) {
            p.stop.store(true, Ordering::Relaxed);
            if let Some(s) = p.stream.lock().take() {
                let _ = s.shutdown(Shutdown::Both);
            }
        }
        true
    }

    /// Hand a block of PCM to every connected peer. Cheap (no copy inside
    /// the mutex region) when there are no peers.
    pub fn submit_frame(&self, pts_ns: u64, pcm_le_bytes: Vec<u8>) {
        let peers = self.shared.peers.lock();
        if peers.is_empty() {
            return;
        }
        // Fan out by cheap reference clone — one `Arc<FrameSlice>` is
        // shared across all peers' queues.
        let slice = Arc::new(FrameSlice {
            pts_ns,
            payload: pcm_le_bytes,
        });
        for p in peers.iter() {
            let mut q = p.queue.lock();
            if q.len() >= PER_PEER_QUEUE_LIMIT {
                // Overflow: drop the oldest frame. A peer that can't keep up
                // is better served with current audio than a stale backlog.
                q.pop_front();
            }
            q.push_back(FrameSlice {
                pts_ns: slice.pts_ns,
                payload: slice.payload.clone(),
            });
        }
    }

    pub fn status(&self) -> ServerStatus {
        let peers: Vec<PeerInfo> = self
            .shared
            .peers
            .lock()
            .iter()
            .map(|p| PeerInfo {
                addr: p.addr,
                label: String::new(),
                connected_at: p.connected_at,
                frames_sent: p.frames_sent.load(Ordering::Relaxed) as u64,
            })
            .collect();
        ServerStatus {
            listening_on: *self.shared.listening_on.lock(),
            peers,
        }
    }
}

impl Drop for Server {
    fn drop(&mut self) {
        self.stop();
    }
}

impl ServerHandle {
    /// Called from the audio callback every block. Cheap fast-path when
    /// the format already matches — one mutex acquire + compare.
    pub fn update_format_if_changed(&self, channels: u8, sample_rate: u32) {
        let display_name = self.display_name.lock().clone();
        let candidate = Handshake::new(sample_rate, channels, display_name);
        let mut guard = self.handshake.lock();
        if guard.as_ref() == Some(&candidate) {
            return;
        }
        *guard = Some(candidate);
        // Kick existing peers so they reconnect with the new handshake.
        for p in self.shared.peers.lock().drain(..) {
            p.stop.store(true, Ordering::Relaxed);
            if let Some(s) = p.stream.lock().take() {
                let _ = s.shutdown(Shutdown::Both);
            }
        }
    }

    /// Push one PCM block to every connected peer. `pcm_le_bytes` is the
    /// interleaved `f32` frame encoded as little-endian bytes.
    pub fn submit(&self, pcm_le_bytes: Vec<u8>) {
        let peers = self.shared.peers.lock();
        if peers.is_empty() {
            return;
        }
        let pts_ns = now_pts_ns();
        for p in peers.iter() {
            let mut q = p.queue.lock();
            if q.len() >= PER_PEER_QUEUE_LIMIT {
                q.pop_front();
            }
            q.push_back(FrameSlice {
                pts_ns,
                payload: pcm_le_bytes.clone(),
            });
        }
    }
}

fn accept_loop(
    listener: TcpListener,
    stop: Arc<AtomicBool>,
    shared: Arc<ServerShared>,
    handshake: Arc<Mutex<Option<Handshake>>>,
    _identity: Arc<IdentityStore>,
) {
    for incoming in listener.incoming() {
        if stop.load(Ordering::Relaxed) {
            break;
        }
        match incoming {
            Ok(stream) => {
                // Drop the sentinel "wake-up" connection from `Server::stop`
                // without spinning up a worker.
                if stop.load(Ordering::Relaxed) {
                    break;
                }
                let Some(current_hs) = handshake.lock().clone() else {
                    log::warn!("dropping connection: no capture format yet");
                    let _ = stream.shutdown(Shutdown::Both);
                    continue;
                };
                if let Err(e) = configure_stream(&stream) {
                    log::warn!("configure peer stream: {e:#}");
                }
                let addr = stream.peer_addr().unwrap_or_else(|_| {
                    SocketAddr::new(IpAddr::V4(Ipv4Addr::UNSPECIFIED), 0)
                });
                let id = shared.next_peer_id.fetch_add(1, Ordering::Relaxed);
                let state = Arc::new(PeerState {
                    id,
                    addr,
                    stop: AtomicBool::new(false),
                    queue: Mutex::new(std::collections::VecDeque::with_capacity(32)),
                    connected_at: SystemTime::now(),
                    frames_sent: AtomicU32::new(0),
                    stream: Mutex::new(Some(stream)),
                });
                shared.peers.lock().push(state.clone());
                let shared_peers_ref = shared.clone();
                thread::Builder::new()
                    .name(format!("ayr-link-peer-{id}"))
                    .spawn(move || {
                        if let Err(e) = run_peer(state.clone(), current_hs) {
                            log::warn!("peer {}: {:#}", state.addr, e);
                        }
                        // Garbage-collect this peer from the shared list
                        // once the writer thread exits.
                        shared_peers_ref
                            .peers
                            .lock()
                            .retain(|p| !Arc::ptr_eq(p, &state));
                    })
                    .ok();
            }
            Err(e) => {
                if stop.load(Ordering::Relaxed) {
                    break;
                }
                log::warn!("accept: {e}");
                thread::sleep(Duration::from_millis(50));
            }
        }
    }
}

fn configure_stream(stream: &TcpStream) -> std::io::Result<()> {
    stream.set_nodelay(true)?;
    stream.set_write_timeout(Some(Duration::from_secs(5)))?;
    Ok(())
}

fn run_peer(state: Arc<PeerState>, handshake: Handshake) -> Result<()> {
    let raw = state
        .stream
        .lock()
        .as_ref()
        .map(|s| s.try_clone())
        .transpose()
        .context("tcp try_clone")?
        .ok_or_else(|| anyhow::anyhow!("peer stream already taken"))?;
    let mut writer = BufWriter::with_capacity(16 * 1024, raw);
    wire::write_handshake(&mut writer, &handshake).context("write handshake")?;
    writer.flush().context("flush handshake")?;

    let idle_backoff = Duration::from_millis(2);
    while !state.stop.load(Ordering::Relaxed) {
        // Pop a batch of frames in one lock acquisition to amortise the
        // mutex cost across all connected peers.
        let batch = {
            let mut q = state.queue.lock();
            if q.is_empty() {
                None
            } else {
                let taken: Vec<FrameSlice> = q.drain(..).collect();
                Some(taken)
            }
        };
        match batch {
            None => thread::sleep(idle_backoff),
            Some(frames) => {
                for frame in frames {
                    wire::write_frame(&mut writer, frame.pts_ns, &frame.payload)
                        .context("write frame")?;
                    state.frames_sent.fetch_add(1, Ordering::Relaxed);
                }
                writer.flush().context("flush frames")?;
            }
        }
    }
    let _ = writer.flush();
    if let Some(s) = state.stream.lock().take() {
        let _ = s.shutdown(Shutdown::Both);
    }
    Ok(())
}

/// Monotonic clock timestamp in nanoseconds, relative to an arbitrary
/// fixed epoch. Used as the frame PTS so the receiver can measure jitter
/// and drift without needing synced wall clocks.
pub fn now_pts_ns() -> u64 {
    // `Instant::now()` is monotonic but doesn't expose its raw value.
    // Anchor against a process-global start and express elapsed ns.
    use std::sync::OnceLock;
    static ANCHOR: OnceLock<Instant> = OnceLock::new();
    let anchor = *ANCHOR.get_or_init(Instant::now);
    anchor.elapsed().as_nanos() as u64
}
