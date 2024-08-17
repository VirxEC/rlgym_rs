pub use rocketsim_rs;

mod render;

use rocketsim_rs::{
    cxx::UniquePtr,
    glam_ext::GameStateA,
    sim::{Arena, CarControls},
};

pub struct StepResult {
    pub obs: Vec<(u32, Vec<f32>)>,
    pub rewards: Vec<f32>,
    pub is_terminal: bool,
}

/// SS - state setting
/// O - observations
/// A - actions
/// R - rewards
/// T - terminal
/// SI - shared info
pub struct Env<SS: StateSetter<SI>, O: Obs<SI>, A: Action<SI>, R: Reward<SI>, T: Terminal<SI>, SI> {
    arena: UniquePtr<Arena>,
    state_setter: SS,
    observations: O,
    action: A,
    reward: R,
    terminal: T,
    shared_info: SI,
    render: bool,
    last_state: Option<GameStateA>,
}

impl<SS: StateSetter<SI>, O: Obs<SI>, A: Action<SI>, R: Reward<SI>, T: Terminal<SI>, SI> Env<SS, O, A, R, T, SI> {
    pub fn new(
        arena: UniquePtr<Arena>,
        state_setter: SS,
        observations: O,
        action: A,
        rewards: R,
        terminal: T,
        shared_info: SI,
    ) -> Self {
        Self {
            arena,
            state_setter,
            observations,
            action,
            reward: rewards,
            terminal,
            shared_info,
            render: false,
            last_state: None,
        }
    }

    pub fn set_render(&mut self, render: bool) {
        self.render = render;
    }

    pub fn num_cars(&self) -> usize {
        self.arena.num_cars()
    }

    /// returns next obs
    pub fn reset(&mut self) -> Vec<(u32, Vec<f32>)> {
        self.state_setter.apply(&mut self.arena, &mut self.shared_info);

        let state = self.arena.pin_mut().get_game_state().to_glam();
        self.observations.reset(&state, &mut self.shared_info);
        self.action.reset(&state, &mut self.shared_info);
        self.terminal.reset(&state, &mut self.shared_info);
        self.reward.reset(&state, &mut self.shared_info);

        let obs = self.observations.build_obs(&state, &mut self.shared_info);
        self.last_state = Some(state);

        obs
    }

    pub fn step(&mut self, raw_actions: Vec<Vec<f32>>, tick_skip: u32) -> StepResult {
        let last_state = self.last_state.as_ref().expect("Must call reset() first!");
        let parsed_actions = self
            .action
            .parse_actions(&raw_actions, last_state, &mut self.shared_info);
        let mapped_actions = parsed_actions
            .into_iter()
            .enumerate()
            .map(|(i, controls)| (last_state.cars[i].id, controls))
            .collect::<Vec<_>>();

        self.arena.pin_mut().set_all_controls(&mapped_actions).unwrap();
        self.arena.pin_mut().step(tick_skip);

        // if self.render {
        //     // ensure nonblocking
        //     render::render(&self.arena);
        // }

        let new_state = self.arena.pin_mut().get_game_state().to_glam();
        let obs = self.observations.build_obs(last_state, &mut self.shared_info);
        let rewards = self.reward.get_rewards(last_state, &mut self.shared_info);
        let is_terminal = self.terminal.is_terminal(last_state, &mut self.shared_info);

        self.last_state = Some(new_state);

        StepResult {
            obs,
            rewards,
            is_terminal,
        }
    }
}

pub trait StateSetter<SI> {
    fn apply(&mut self, arena: &mut UniquePtr<Arena>, shared_info: &mut SI);
}

pub trait Obs<SI> {
    fn get_obs_space(&self) -> usize;
    fn reset(&mut self, initial_state: &GameStateA, shared_info: &mut SI);
    fn build_obs(&mut self, state: &GameStateA, shared_info: &mut SI) -> Vec<(u32, Vec<f32>)>;
}

pub trait Action<SI> {
    fn get_action_space(&self, agent_id: u32, shared_info: &mut SI) -> usize;
    fn reset(&mut self, initial_state: &GameStateA, shared_info: &mut SI);
    fn parse_actions(
        &mut self,
        actions: &[Vec<f32>],
        state: &GameStateA,
        shared_info: &mut SI,
    ) -> Vec<CarControls>;
}

pub trait Reward<SI> {
    fn reset(&mut self, initial_state: &GameStateA, shared_info: &mut SI);
    fn get_rewards(&mut self, state: &GameStateA, shared_info: &mut SI) -> Vec<f32>;
}

pub trait Terminal<SI> {
    fn reset(&mut self, initial_state: &GameStateA, shared_info: &mut SI);
    fn is_terminal(&mut self, state: &GameStateA, shared_info: &mut SI) -> bool;
}
