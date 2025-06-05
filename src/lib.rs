pub use rocketsim_rs;
pub use rocketsim_rs::glam_ext::glam;

mod render;

use render::RLViserSocketHandler;
use rocketsim_rs::{
    cxx::UniquePtr,
    glam_ext::GameStateA,
    sim::{Arena, CarControls},
};
use std::{io, time::Duration};

pub type FullObs = Vec<Vec<f32>>;

pub struct StepResult {
    pub obs: FullObs,
    pub rewards: Vec<f32>,
    pub is_terminal: bool,
    pub truncated: bool,
    pub state: GameStateA,
}

pub struct Env<SS, SIP, OBS, ACT, REW, TERM, TRUNC, SI>
where
    SS: StateSetter<SI>,
    SIP: SharedInfoProvider<SI>,
    OBS: Obs<SI>,
    ACT: Action<SI>,
    REW: Reward<SI>,
    TERM: Terminal<SI>,
    TRUNC: Truncate<SI>,
{
    arena: UniquePtr<Arena>,
    state_setter: SS,
    shared_info_provider: SIP,
    observations: OBS,
    action: ACT,
    reward: REW,
    terminal: TERM,
    truncate: TRUNC,
    shared_info: SI,
    tick_skip: u32,
    renderer: Option<RLViserSocketHandler>,
}

impl<SS, SIP, OBS, ACT, REW, TERM, TRUNC, SI> Env<SS, SIP, OBS, ACT, REW, TERM, TRUNC, SI>
where
    SS: StateSetter<SI>,
    SIP: SharedInfoProvider<SI>,
    OBS: Obs<SI>,
    ACT: Action<SI>,
    REW: Reward<SI>,
    TERM: Terminal<SI>,
    TRUNC: Truncate<SI>,
{
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        arena: UniquePtr<Arena>,
        state_setter: SS,
        shared_info_provider: SIP,
        observations: OBS,
        action: ACT,
        reward: REW,
        terminal: TERM,
        truncate: TRUNC,
        shared_info: SI,
    ) -> Self {
        Self {
            arena,
            state_setter,
            shared_info_provider,
            observations,
            action,
            reward,
            terminal,
            truncate,
            shared_info,
            tick_skip: ACT::get_tick_skip(),
            renderer: None,
        }
    }

    /// Call at any time to open RLViser and start rendering the environment
    pub fn enable_rendering(&mut self, try_launch_exe: bool) {
        if self.renderer.is_none() {
            self.renderer = Some(RLViserSocketHandler::new(try_launch_exe).unwrap());
        }
    }

    /// Check if the game should be paused
    pub fn is_paused(&self) -> bool {
        self.renderer
            .as_ref()
            .map(RLViserSocketHandler::is_paused)
            .unwrap_or_default()
    }

    /// Tick rate, by default, should be `Duration::from_secs_f32(TICK_SKIP as f32 / 120.)`
    pub fn handle_incoming_states(&mut self, tick_rate: &mut Duration) -> io::Result<()> {
        if let Some(renderer) = &mut self.renderer {
            renderer.handle_return_message(&mut self.arena, tick_rate, self.tick_skip)?;
        }

        Ok(())
    }

    /// Call at any time to close RLViser
    pub fn stop_rendering(&mut self) {
        if let Some(renderer) = self.renderer.take() {
            renderer.quit().unwrap();
        }
    }

    pub fn get_obs_space(&self) -> usize {
        self.observations.get_obs_space(&self.shared_info)
    }

    pub fn get_action_space(&self) -> usize {
        self.action.get_action_space(&self.shared_info)
    }

    pub fn num_cars(&self) -> usize {
        self.arena.num_cars()
    }

    pub fn get_shared_info(&self) -> &SI {
        &self.shared_info
    }

    pub fn get_mut_shared_info(&mut self) -> &mut SI {
        &mut self.shared_info
    }

    /// returns next obs
    pub fn reset(&mut self) -> (GameStateA, FullObs) {
        self.state_setter
            .apply(&mut self.arena, &mut self.shared_info);

        let state = self.arena.pin_mut().get_game_state().to_glam();
        self.shared_info_provider
            .reset(&state, &mut self.shared_info);
        self.observations.reset(&state, &mut self.shared_info);
        self.action.reset(&state, &mut self.shared_info);
        self.terminal.reset(&state, &mut self.shared_info);
        self.truncate.reset(&state, &mut self.shared_info);
        self.reward.reset(&state, &mut self.shared_info);

        let obs = self.observations.build_obs(&state, &mut self.shared_info);
        debug_assert!(
            !obs.iter().any(|a| a.iter().copied().any(f32::is_nan)),
            "NaN in obs: {obs:?}"
        );

        (state, obs)
    }

    pub fn step(&mut self, initial_state: &GameStateA, raw_actions: &[ACT::Input]) -> StepResult {
        let parsed_actions =
            self.action
                .parse_actions(raw_actions, initial_state, &mut self.shared_info);

        self.arena
            .pin_mut()
            .set_all_controls(parsed_actions)
            .unwrap();
        self.arena.pin_mut().step(self.tick_skip);

        let raw_state = self.arena.pin_mut().get_game_state();

        if let Some(renderer) = &mut self.renderer {
            renderer.send_state(&raw_state).unwrap();
        }

        let state = raw_state.to_glam();

        self.shared_info_provider
            .apply(&state, &mut self.shared_info);
        let obs = self.observations.build_obs(&state, &mut self.shared_info);
        let rewards = self.reward.get_rewards(&state, &mut self.shared_info);
        let is_terminal = self.terminal.is_terminal(&state, &mut self.shared_info);
        let truncated = self.truncate.should_truncate(&state, &mut self.shared_info);

        debug_assert!(
            !obs.iter().any(|a| a.iter().copied().any(f32::is_nan)),
            "NaN in obs: {obs:?}"
        );
        debug_assert!(
            !rewards.iter().copied().any(f32::is_nan),
            "NaN in rewards: {rewards:?}"
        );

        StepResult {
            obs,
            rewards,
            is_terminal,
            truncated,
            state,
        }
    }
}

pub trait SharedInfoProvider<SI> {
    fn reset(&mut self, initial_state: &GameStateA, shared_info: &mut SI);
    fn apply(&mut self, game_state: &GameStateA, shared_info: &mut SI);
}

pub trait StateSetter<SI> {
    fn apply(&mut self, arena: &mut UniquePtr<Arena>, shared_info: &mut SI);
}

pub trait Obs<SI> {
    fn get_obs_space(&self, shared_info: &SI) -> usize;
    fn reset(&mut self, initial_state: &GameStateA, shared_info: &mut SI);
    fn build_obs(&mut self, state: &GameStateA, shared_info: &mut SI) -> FullObs;
}

pub trait Action<SI> {
    type Input;

    fn get_tick_skip() -> u32;
    fn get_action_space(&self, shared_info: &SI) -> usize;
    fn reset(&mut self, initial_state: &GameStateA, shared_info: &mut SI);
    fn parse_actions<'a>(
        &'a mut self,
        actions: &[Self::Input],
        state: &GameStateA,
        shared_info: &'a mut SI,
    ) -> &'a [(u32, CarControls)];
}

pub trait Reward<SI> {
    fn reset(&mut self, initial_state: &GameStateA, shared_info: &mut SI);
    fn get_rewards(&mut self, state: &GameStateA, shared_info: &mut SI) -> Vec<f32>;
}

pub trait Terminal<SI> {
    fn reset(&mut self, initial_state: &GameStateA, shared_info: &mut SI);
    fn is_terminal(&mut self, state: &GameStateA, shared_info: &mut SI) -> bool;
}

pub trait Truncate<SI> {
    fn reset(&mut self, initial_state: &GameStateA, shared_info: &mut SI);
    fn should_truncate(&mut self, state: &GameStateA, shared_info: &mut SI) -> bool;
}
