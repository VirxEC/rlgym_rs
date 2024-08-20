use rocketsim_rs::{
    bytes::{FromBytes, FromBytesExact, ToBytes},
    cxx::UniquePtr,
    sim::Arena,
    GameState,
};
use std::{
    io,
    net::{IpAddr, SocketAddr, UdpSocket},
    process::Command,
    str::FromStr,
    time::Duration,
};

/// Pass this into rlviser as the first argument
/// default: 45243
const RLVISER_PORT: u16 = 45243;

/// Pass this into rlviser as the second argument
/// default: 34254
const ROCKETSIM_PORT: u16 = 34254;

const RLVISER_PATH: &str = if cfg!(windows) {
    "./rlviser.exe"
} else {
    "./rlviser"
};

#[repr(u8)]
#[derive(Debug, Clone, Copy)]
enum UdpPacketTypes {
    Quit,
    GameState,
    Connection,
    Paused,
    Speed,
    Render,
}

impl From<u8> for UdpPacketTypes {
    fn from(val: u8) -> Self {
        match val {
            0 => Self::Quit,
            1 => Self::GameState,
            2 => Self::Connection,
            3 => Self::Paused,
            4 => Self::Speed,
            5 => Self::Render,
            _ => panic!("Invalid packet type"),
        }
    }
}

pub struct RLViserSocketHandler {
    socket: UdpSocket,
    rlviser_addr: SocketAddr,
    min_game_state_buf: [u8; GameState::MIN_NUM_BYTES],
    game_state_buffer: Vec<u8>,
    paused: bool,
}

impl RLViserSocketHandler {
    pub fn new() -> io::Result<Self> {
        // launch rlviser
        if let Err(e) = Command::new(RLVISER_PATH).spawn() {
            eprintln!("Failed to launch RLViser ({RLVISER_PATH}): {e}");
        }

        // open rlviser socket
        let socket = UdpSocket::bind(("0.0.0.0", ROCKETSIM_PORT))?;
        // print the socket address
        println!("Listening on {}", socket.local_addr()?);

        let rlviser_addr = SocketAddr::new(IpAddr::from_str("127.0.0.1").unwrap(), RLVISER_PORT);

        // We now don't want to wait for anything UDP so set to non-blocking
        socket.set_nonblocking(true)?;

        // notify rlviser that we're connected
        // it will send us info on the desired game speed / if the game should be paused
        // if you choose to ignore this, at least send the right game speed / paused state back
        // otherwise things like packet interpolation will be off
        socket.send_to(&[UdpPacketTypes::Connection as u8], rlviser_addr)?;

        Ok(Self {
            socket,
            rlviser_addr,
            min_game_state_buf: [0; GameState::MIN_NUM_BYTES],
            game_state_buffer: Vec::new(),
            paused: false,
        })
    }

    pub fn is_paused(&self) -> bool {
        self.paused
    }

    pub fn send_state(&mut self, game_state: &GameState) -> io::Result<()> {
        self.socket
            .send_to(&[UdpPacketTypes::GameState as u8], self.rlviser_addr)?;
        self.socket
            .send_to(&game_state.to_bytes(), self.rlviser_addr)?;

        Ok(())
    }

    pub fn handle_return_message(
        &mut self,
        arena: &mut UniquePtr<Arena>,
        interval: &mut Duration,
        tick_skip: u32,
    ) -> io::Result<()> {
        let mut byte_buffer = [0];

        while let Ok((_, src)) = self.socket.recv_from(&mut byte_buffer) {
            let packet_type = UdpPacketTypes::from(byte_buffer[0]);

            match packet_type {
                UdpPacketTypes::GameState => {
                    self.socket.peek_from(&mut self.min_game_state_buf)?;

                    let num_bytes = GameState::get_num_bytes(&self.min_game_state_buf);
                    self.game_state_buffer.resize(num_bytes, 0);
                    self.socket.recv_from(&mut self.game_state_buffer)?;

                    // set the game state
                    let game_state = GameState::from_bytes(&self.game_state_buffer);
                    if let Err(e) = arena.pin_mut().set_game_state(&game_state) {
                        println!("Error setting game state: {e}");
                    };
                }
                UdpPacketTypes::Connection => {
                    println!("Connection established to {src}");
                }
                UdpPacketTypes::Speed => {
                    let mut speed_buffer = [0; f32::NUM_BYTES];
                    self.socket.recv_from(&mut speed_buffer)?;
                    let speed = f32::from_bytes(&speed_buffer);
                    *interval = Duration::from_secs_f32(tick_skip as f32 / (120. * speed));
                }
                UdpPacketTypes::Paused => {
                    self.socket.recv_from(&mut byte_buffer)?;
                    self.paused = byte_buffer[0] == 1;
                }
                UdpPacketTypes::Quit | UdpPacketTypes::Render => {
                    panic!("We shouldn't be receiving packets of type {packet_type:?}")
                }
            }
        }

        Ok(())
    }

    pub fn quit(self) -> io::Result<()> {
        self.socket
            .send_to(&[UdpPacketTypes::Quit as u8], self.rlviser_addr)?;

        Ok(())
    }
}
