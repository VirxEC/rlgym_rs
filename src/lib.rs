use rlviser_rocketsim::ArenaRlviserExt;
pub use rocketsim;
use rocketsim::{
    Arena, ArenaEvent, BallState, BoostPadConfig, BoostPadState, CarControls, CarInfo, CarState,
    GameMode, consts,
};

pub type FullObs = Vec<Vec<f32>>;
pub type ActionMasks = Vec<Vec<bool>>;

#[derive(Debug, Clone)]
pub struct GameState {
    pub game_mode: GameMode,
    pub tick_count: u64,
    pub ball: BallState,
    pub cars: Vec<(CarInfo, CarState)>,
    pub boost_pads: Vec<(BoostPadConfig, BoostPadState)>,
    pub events: Vec<ArenaEvent>,
}

impl GameState {
    fn ball_within_hoops_goal_xy_margin_eq(x: f32, y: f32) -> f32 {
        const SCALE_Y: f32 = 0.9;
        const OFFSET_Y: f32 = 2770.0;
        const RADIUS_SQ: f32 = 716.0 * 716.0;

        let dy = y.abs() * SCALE_Y - OFFSET_Y;
        let dist_sq = x * x + dy * dy;
        dist_sq - RADIUS_SQ
    }

    pub fn is_ball_scored(&self) -> bool {
        match self.game_mode {
            GameMode::Soccar | GameMode::Heatseeker | GameMode::Snowday => {
                self.ball.pos.y.abs()
                    > consts::goal::SOCCAR_GOAL_SCORE_BASE_THRESHOLD_Y
                        + consts::ball::get_radius(self.game_mode)
            }
            GameMode::Hoops => {
                if self.ball.pos.z < consts::goal::HOOPS_GOAL_SCORE_THRESHOLD_Z {
                    Self::ball_within_hoops_goal_xy_margin_eq(self.ball.pos.x, self.ball.pos.y)
                        < 0.0
                } else {
                    false
                }
            }
            GameMode::Dropshot => {
                self.ball.pos.z < -consts::ball::get_radius(self.game_mode) * 1.75
            }
            GameMode::TheVoid => false,
        }
    }

    pub fn num_cars(&self) -> usize {
        self.cars.len()
    }

    pub fn num_boost_pads(&self) -> usize {
        self.boost_pads.len()
    }
}

pub struct StepResult {
    pub obs: FullObs,
    pub action_masks: ActionMasks,
    pub rewards: Vec<f32>,
    pub is_terminal: bool,
    pub truncated: bool,
    pub state: GameState,
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
    pub arena: Arena,
    pub state_setter: SS,
    pub observations: OBS,
    pub action: ACT,
    pub reward: REW,
    pub terminal: TERM,
    pub truncate: TRUNC,
    pub shared_info: SI,
    tick_skip: u8,
    events: Vec<ArenaEvent>,
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
            events: Vec::new(),
        }
    }

    /// Enables rendering via RLViser.
    ///
    /// Note that this is not intended for high-performance use and may significantly slow down the environment.
    /// Usage is only suitable for real-time+ rendering, such as for debugging or visualization purposes.
    pub fn set_rlviser_enabled(&mut self, enabled: bool) {
        self.arena.set_rlviser_enabled(enabled).unwrap();
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

    fn get_game_state(&self) -> GameState {
        let cars = (0..self.arena.num_cars())
            .map(|i| {
                let (info, state) = self.arena.get_car_info_and_state(i);
                (*info, *state)
            })
            .collect::<Vec<_>>();
        let ball = *self.arena.get_ball_state();

        let boost_pads = match self.arena.game_mode() {
            GameMode::Soccar | GameMode::Hoops | GameMode::Snowday => {
                (0..self.arena.num_boost_pads())
                    .map(|i| {
                        (
                            *self.arena.get_boost_pad_config(i),
                            self.arena.get_boost_pad_state(i),
                        )
                    })
                    .collect()
            }
            GameMode::Dropshot | GameMode::Heatseeker | GameMode::TheVoid => Vec::new(),
        };

        GameState {
            game_mode: self.arena.game_mode(),
            tick_count: self.arena.tick_count(),
            cars,
            ball,
            boost_pads,
            events: self.events.clone(),
        }
    }

    /// returns next obs
    pub fn reset(&mut self) -> (GameState, FullObs, Vec<Vec<bool>>) {
        self.events.clear();
        self.state_setter
            .apply(&mut self.arena, &mut self.shared_info);

        let state = self.get_game_state();
        self.shared_info.reset(&state);
        self.observations.reset(&state, &mut self.shared_info);
        self.action.reset(&state, &mut self.shared_info);
        self.terminal.reset(&state, &mut self.shared_info);
        self.truncate.reset(&state, &mut self.shared_info);
        self.reward.reset(&state, &mut self.shared_info);

        let obs = self.observations.build_obs(&state, &mut self.shared_info);
        let masks = self.action.get_action_masks(&state, &mut self.shared_info);
        debug_assert!(
            !obs.iter().any(|a| a.iter().copied().any(f32::is_nan)),
            "NaN in obs: {obs:?}"
        );

        (state, obs, masks)
    }

    pub fn get_tick_skip(&self) -> u8 {
        self.tick_skip
    }

    pub fn pre_step(
        &mut self,
        initial_state: &GameState,
        raw_actions: &[<ACT as Action<SI>>::Input],
    ) {
        self.events.clear();

        let parsed_actions =
            self.action
                .parse_actions(raw_actions, initial_state, &mut self.shared_info);

        for (car_idx, action) in parsed_actions.iter().copied() {
            self.arena.set_car_controls(car_idx, action);
        }
    }

    pub fn post_step(&mut self) -> StepResult {
        let state = self.get_game_state();

        self.shared_info.update(&state);
        let obs = self.observations.build_obs(&state, &mut self.shared_info);
        let action_masks = self.action.get_action_masks(&state, &mut self.shared_info);
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
            action_masks,
            rewards,
            is_terminal,
            truncated,
            state,
        }
    }

    pub fn step(&mut self, initial_state: &GameState, raw_actions: &[ACT::Input]) -> StepResult {
        self.pre_step(initial_state, raw_actions);

        for _ in 0..self.tick_skip {
            self.events.extend_from_slice(self.arena.step_tick());
        }

        self.post_step()
    }
}

pub trait SharedInfoProvider {
    fn reset(&mut self, initial_state: &GameState);
    fn update(&mut self, game_state: &GameState);
}

pub trait StateSetter<SI> {
    fn apply(&mut self, arena: &mut Arena, shared_info: &mut SI);
}

pub trait Obs<SI> {
    fn get_obs_space(&self, shared_info: &SI) -> usize;
    fn reset(&mut self, initial_state: &GameState, shared_info: &mut SI);
    fn build_obs(&mut self, state: &GameState, shared_info: &mut SI) -> FullObs;
}

pub trait Action<SI> {
    type Input;

    fn get_tick_skip() -> u8;
    fn get_action_space(&self, shared_info: &SI) -> usize;
    fn reset(&mut self, initial_state: &GameState, shared_info: &mut SI);
    fn parse_actions<'a>(
        &'a mut self,
        actions: &[Self::Input],
        state: &GameState,
        shared_info: &'a mut SI,
    ) -> &'a [(usize, CarControls)];
    fn get_action_masks(&mut self, state: &GameState, shared_info: &mut SI) -> Vec<Vec<bool>>;
}

pub trait Reward<SI> {
    fn reset(&mut self, initial_state: &GameState, shared_info: &mut SI);
    fn get_rewards(&mut self, state: &GameState, shared_info: &mut SI) -> Vec<f32>;
}

pub trait Terminal<SI> {
    fn reset(&mut self, initial_state: &GameState, shared_info: &mut SI);
    fn is_terminal(&mut self, state: &GameState, shared_info: &mut SI) -> bool;
}

pub trait Truncate<SI> {
    fn reset(&mut self, initial_state: &GameState, shared_info: &mut SI);
    fn should_truncate(&mut self, state: &GameState, shared_info: &mut SI) -> bool;
}
