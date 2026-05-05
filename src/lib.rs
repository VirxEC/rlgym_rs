pub use rocketsim;
use rocketsim::{Arena, ArenaState, CarControls};

pub type FullObs = Vec<Vec<f32>>;

pub struct StepResult {
    pub obs: FullObs,
    pub rewards: Vec<f32>,
    pub is_terminal: bool,
    pub truncated: bool,
    pub state: ArenaState,
}

pub struct Env<SS, OBS, ACT, REW, TERM, TRUNC, SI>
where
    SS: StateSetter<SI>,
    OBS: Obs<SI>,
    ACT: Action<SI>,
    REW: Reward<SI>,
    TERM: Terminal<SI>,
    TRUNC: Truncate<SI>,
{
    arena: Arena,
    state_setter: SS,
    observations: OBS,
    action: ACT,
    reward: REW,
    terminal: TERM,
    truncate: TRUNC,
    shared_info: SI,
    tick_skip: u8,
}

impl<SS, OBS, ACT, REW, TERM, TRUNC, SI> Env<SS, OBS, ACT, REW, TERM, TRUNC, SI>
where
    SS: StateSetter<SI>,
    OBS: Obs<SI>,
    ACT: Action<SI>,
    REW: Reward<SI>,
    TERM: Terminal<SI>,
    TRUNC: Truncate<SI>,
    SI: SharedInfoProvider,
{
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        arena: Arena,
        state_setter: SS,
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
            observations,
            action,
            reward,
            terminal,
            truncate,
            shared_info,
            tick_skip: ACT::get_tick_skip(),
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
    pub fn reset(&mut self) -> (ArenaState, FullObs) {
        self.state_setter
            .apply(&mut self.arena, &mut self.shared_info);

        let state = self.arena.get_arena_state();
        self.shared_info.reset(&state);
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

    pub fn get_tick_skip(&self) -> u8 {
        self.tick_skip
    }

    pub fn pre_step(
        &mut self,
        initial_state: &ArenaState,
        raw_actions: &[<ACT as Action<SI>>::Input],
    ) {
        let parsed_actions =
            self.action
                .parse_actions(raw_actions, initial_state, &mut self.shared_info);

        for (car_idx, action) in parsed_actions.iter().copied() {
            self.arena.set_car_controls(car_idx, action);
        }
    }

    pub fn step_arena_one_tick(&mut self) {
        self.arena.step_tick();
    }

    pub fn post_step(&mut self) -> StepResult {
        let state = self.arena.get_arena_state();

        self.shared_info.update(&state);
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

    pub fn step(&mut self, initial_state: &ArenaState, raw_actions: &[ACT::Input]) -> StepResult {
        self.pre_step(initial_state, raw_actions);

        for _ in 0..self.tick_skip {
            self.step_arena_one_tick();
        }

        self.post_step()
    }
}

pub trait SharedInfoProvider {
    fn reset(&mut self, initial_state: &ArenaState);
    fn update(&mut self, game_state: &ArenaState);
}

pub trait StateSetter<SI> {
    fn apply(&mut self, arena: &mut Arena, shared_info: &mut SI);
}

pub trait Obs<SI> {
    fn get_obs_space(&self, shared_info: &SI) -> usize;
    fn reset(&mut self, initial_state: &ArenaState, shared_info: &mut SI);
    fn build_obs(&mut self, state: &ArenaState, shared_info: &mut SI) -> FullObs;
}

pub trait Action<SI> {
    type Input;

    fn get_tick_skip() -> u8;
    fn get_action_space(&self, shared_info: &SI) -> usize;
    fn reset(&mut self, initial_state: &ArenaState, shared_info: &mut SI);
    fn parse_actions<'a>(
        &'a mut self,
        actions: &[Self::Input],
        state: &ArenaState,
        shared_info: &'a mut SI,
    ) -> &'a [(usize, CarControls)];
}

pub trait Reward<SI> {
    fn reset(&mut self, initial_state: &ArenaState, shared_info: &mut SI);
    fn get_rewards(&mut self, state: &ArenaState, shared_info: &mut SI) -> Vec<f32>;
}

pub trait Terminal<SI> {
    fn reset(&mut self, initial_state: &ArenaState, shared_info: &mut SI);
    fn is_terminal(&mut self, state: &ArenaState, shared_info: &mut SI) -> bool;
}

pub trait Truncate<SI> {
    fn reset(&mut self, initial_state: &ArenaState, shared_info: &mut SI);
    fn should_truncate(&mut self, state: &ArenaState, shared_info: &mut SI) -> bool;
}
