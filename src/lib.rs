pub use rocketsim_rs;

use rocketsim_rs::{
    cxx::UniquePtr,
    glam_ext::GameStateA,
    sim::{Arena, CarControls},
};
use std::rc::Rc;

pub type FullObs = Vec<Vec<f32>>;

pub struct StepResult {
    pub obs: Rc<FullObs>,
    pub rewards: Vec<f32>,
    pub is_terminal: bool,
    pub truncated: bool,
    pub state: Rc<GameStateA>,
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
    last_state: Option<Rc<GameStateA>>,
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
            last_state: None,
        }
    }

    pub fn get_obs_space(&self, agent_id: u32) -> usize {
        self.observations.get_obs_space(agent_id, &self.shared_info)
    }

    pub fn get_action_space(&self, agent_id: u32) -> usize {
        self.action.get_action_space(agent_id, &self.shared_info)
    }

    pub fn num_cars(&self) -> usize {
        self.arena.num_cars()
    }

    pub fn shared_info(&self) -> &SI {
        &self.shared_info
    }

    /// returns next obs
    pub fn reset(&mut self) -> Rc<FullObs> {
        self.state_setter
            .apply(&mut self.arena, &mut self.shared_info);

        let state = self.arena.pin_mut().get_game_state().to_glam();
        self.shared_info_provider
            .reset(&state, &mut self.shared_info);
        self.observations.reset(&state, &mut self.shared_info);
        self.action.reset(&state, &mut self.shared_info);
        self.terminal.reset(&state, &mut self.shared_info);
        self.reward.reset(&state, &mut self.shared_info);

        let obs = self.observations.build_obs(&state, &mut self.shared_info);
        self.last_state = Some(Rc::new(state));

        Rc::new(obs)
    }

    pub fn step(&mut self, raw_actions: ACT::Input) -> StepResult {
        let last_state = self.last_state.as_ref().expect("Must call reset() first!");
        let parsed_actions =
            self.action
                .parse_actions(raw_actions, last_state, &mut self.shared_info);
        let mapped_actions = parsed_actions
            .into_iter()
            .enumerate()
            .map(|(i, controls)| (last_state.cars[i].id, controls))
            .collect::<Vec<_>>();

        self.arena
            .pin_mut()
            .set_all_controls(&mapped_actions)
            .unwrap();
        self.arena.pin_mut().step(self.tick_skip);

        let state = Rc::new(self.arena.pin_mut().get_game_state().to_glam());
        self.shared_info_provider
            .apply(&state, &mut self.shared_info);
        let obs = self.observations.build_obs(&state, &mut self.shared_info);
        let rewards = self.reward.get_rewards(&state, &mut self.shared_info);
        let is_terminal = self.terminal.is_terminal(&state, &mut self.shared_info);
        let truncated = self.truncate.should_truncate(&state, &mut self.shared_info);

        self.last_state = Some(state.clone());

        StepResult {
            obs: Rc::new(obs),
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
    fn get_obs_space(&self, agent_id: u32, shared_info: &SI) -> usize;
    fn reset(&mut self, initial_state: &GameStateA, shared_info: &mut SI);
    fn build_obs(&mut self, state: &GameStateA, shared_info: &mut SI) -> FullObs;
}

pub trait Action<SI> {
    type Input;

    fn get_tick_skip() -> u32;
    fn get_action_space(&self, agent_id: u32, shared_info: &SI) -> usize;
    fn reset(&mut self, initial_state: &GameStateA, shared_info: &mut SI);
    fn parse_actions(
        &mut self,
        actions: Self::Input,
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

pub trait Truncate<SI> {
    fn reset(&mut self, initial_state: &GameStateA, shared_info: &mut SI);
    fn should_truncate(&mut self, state: &GameStateA, shared_info: &mut SI) -> bool;
}
