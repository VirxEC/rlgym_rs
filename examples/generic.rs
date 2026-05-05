use std::{
    iter::repeat_n,
    thread::sleep,
    time::{Duration, Instant},
};

use rand::{Rng, distr::Uniform, rngs::ThreadRng};
use rlgym::{
    Action, Env, FullObs, Obs, Reward, SharedInfoProvider, StateSetter, Terminal, Truncate,
    rocketsim::{
        Arena, ArenaState, BallState, CarBodyConfig, CarControls, CarInfo, CarState, GameMode,
        Team, consts, init_from_default,
    },
};

struct SharedInfo {
    start_tick: u64,
    rng: ThreadRng,
}

impl Default for SharedInfo {
    fn default() -> Self {
        Self {
            start_tick: 0,
            rng: rand::rng(),
        }
    }
}

impl SharedInfoProvider for SharedInfo {
    fn reset(&mut self, initial_state: &ArenaState) {
        self.start_tick = initial_state.tick_count;
    }

    fn update(&mut self, _game_state: &ArenaState) {}
}

struct MyStateSetter;

impl StateSetter<SharedInfo> for MyStateSetter {
    fn apply(&mut self, arena: &mut Arena, shared_info: &mut SharedInfo) {
        arena.reset_to_random_kickoff(Some(shared_info.rng.random()));
    }
}

struct MyObs;

impl MyObs {
    const ZERO_PADDING: usize = 2;
    const BALL_OBS: usize = 9;
    const CAR_OBS: usize = 9;

    const OBS_SPACE: usize = Self::BALL_OBS + Self::CAR_OBS * Self::ZERO_PADDING * 2;

    fn get_ball_obs(ball: &BallState) -> Vec<f32> {
        let mut obs_vec = Vec::with_capacity(Self::BALL_OBS);
        obs_vec.extend(ball.pos.to_array());
        obs_vec.extend(ball.vel.to_array());
        obs_vec.extend(ball.ang_vel.to_array());

        obs_vec
    }

    fn get_all_car_obs(cars: &[(CarInfo, CarState)]) -> Vec<(usize, Team, Vec<f32>)> {
        debug_assert!(
            cars.len() <= Self::ZERO_PADDING * 2,
            "Too many cars for obs space: {} > {}",
            cars.len(),
            Self::ZERO_PADDING * 2
        );

        cars.iter()
            .map(|(info, state)| {
                let mut obs_vec = Vec::with_capacity(Self::CAR_OBS);
                obs_vec.extend(state.pos.to_array());
                obs_vec.extend(state.vel.to_array());
                obs_vec.extend(state.ang_vel.to_array());

                (info.idx, info.team, obs_vec)
            })
            .collect()
    }
}

impl Obs<SharedInfo> for MyObs {
    fn get_obs_space(&self, _shared_info: &SharedInfo) -> usize {
        Self::OBS_SPACE
    }

    fn reset(&mut self, _initial_state: &ArenaState, _shared_info: &mut SharedInfo) {}

    fn build_obs(&mut self, state: &ArenaState, _shared_info: &mut SharedInfo) -> FullObs {
        let mut obs = Vec::with_capacity(state.num_cars());

        let ball_obs = Self::get_ball_obs(&state.ball);
        let cars = Self::get_all_car_obs(&state.cars);

        for (info, _) in &state.cars {
            let mut obs_vec: Vec<f32> = Vec::with_capacity(Self::OBS_SPACE);
            obs_vec.extend(&ball_obs);

            // current car's obs
            obs_vec.extend(
                &cars
                    .iter()
                    .find(|(car_id, _, _)| *car_id == info.idx)
                    .unwrap()
                    .2,
            );

            // teammate's obs
            let mut num_teammates = 0;
            for (car_id, team, obs) in &cars {
                if *team == info.team && *car_id != info.idx {
                    obs_vec.extend(obs);
                    num_teammates += 1;
                }
            }

            // zero padding
            for _ in 0..Self::ZERO_PADDING - num_teammates - 1 {
                obs_vec.extend(repeat_n(0.0, Self::CAR_OBS));
            }

            // opponent's obs
            let mut num_opponents = 0;
            for (_, team, obs) in &cars {
                if *team != info.team {
                    obs_vec.extend(obs);
                    num_opponents += 1;
                }
            }

            // zero padding
            for _ in 0..Self::ZERO_PADDING - num_opponents {
                obs_vec.extend(repeat_n(0.0, Self::CAR_OBS));
            }

            assert_eq!(obs_vec.len(), Self::OBS_SPACE);
            obs.push(obs_vec);
        }

        obs
    }
}

struct MyAction {
    actions_table: Vec<CarControls>,
    action_buffer: [(usize, CarControls); 8],
}

impl Default for MyAction {
    fn default() -> Self {
        let mut actions_table = Vec::new();

        for throttle in [1.0, 0.0, -1.0] {
            for steer in [1.0, 0.0, -1.0] {
                for boost in [false, true] {
                    for handbrake in [false, true] {
                        if boost && throttle != 1.0 {
                            continue;
                        }

                        actions_table.push(CarControls {
                            throttle,
                            steer,
                            boost,
                            handbrake,
                            jump: false,
                            pitch: 0.0,
                            yaw: 0.0,
                            roll: 0.0,
                        });
                    }
                }
            }
        }

        dbg!(actions_table.len());

        Self {
            actions_table,
            action_buffer: Default::default(),
        }
    }
}

impl Action<SharedInfo> for MyAction {
    type Input = usize;

    fn get_tick_skip() -> u8 {
        8
    }

    fn get_action_space(&self, _shared_info: &SharedInfo) -> usize {
        self.actions_table.len()
    }

    fn reset(&mut self, _initial_state: &ArenaState, _shared_info: &mut SharedInfo) {}

    fn parse_actions(
        &mut self,
        actions: &[usize],
        state: &ArenaState,
        _shared_info: &mut SharedInfo,
    ) -> &[(usize, CarControls)] {
        for ((buf, (info, _)), action) in
            self.action_buffer.iter_mut().zip(&state.cars).zip(actions)
        {
            *buf = (info.idx, self.actions_table[*action]);
        }

        &self.action_buffer[..state.num_cars()]
    }
}

struct WeightedReward {
    func: Box<dyn Reward<SharedInfo>>,
    weight: f32,
}

struct CombinedWeightedRewards {
    rewards: Box<[WeightedReward]>,
}

macro_rules! new_rewards {
    ($(($reward:expr, $weight:expr)),* $(,)?) => {
        CombinedWeightedRewards {
            rewards: vec![$(WeightedReward {
                func: Box::new($reward) as Box<dyn Reward<SharedInfo>>,
                weight: $weight
            }),*].into_boxed_slice(),
        }
    };
}

impl Reward<SharedInfo> for CombinedWeightedRewards {
    fn reset(&mut self, _initial_state: &ArenaState, _shared_info: &mut SharedInfo) {}

    fn get_rewards(&mut self, state: &ArenaState, shared_info: &mut SharedInfo) -> Vec<f32> {
        let mut rewards: Vec<f32> = vec![0.0; state.cars.len()];

        for reward in &mut self.rewards {
            let fn_rewards = reward.func.get_rewards(state, shared_info);

            for (total, extra) in rewards.iter_mut().zip(fn_rewards) {
                *total += extra * reward.weight;
            }
        }

        rewards
    }
}

struct DistanceToBallReward;

impl Reward<SharedInfo> for DistanceToBallReward {
    fn reset(&mut self, _initial_state: &ArenaState, _shared_info: &mut SharedInfo) {}

    fn get_rewards(&mut self, state: &ArenaState, _shared_info: &mut SharedInfo) -> Vec<f32> {
        state
            .cars
            .iter()
            .map(|(_, car)| {
                let car_ball_dist = car.pos.distance(state.ball.pos);

                -car_ball_dist
            })
            .collect()
    }
}

struct OnGoal;

fn ball_within_hoops_goal_xy_margin_eq(x: f32, y: f32) -> f32 {
    const SCALE_Y: f32 = 0.9;
    const OFFSET_Y: f32 = 2770.0;
    const RADIUS_SQ: f32 = 716.0 * 716.0;

    let dy = y.abs() * SCALE_Y - OFFSET_Y;
    let dist_sq = x * x + dy * dy;
    dist_sq - RADIUS_SQ
}

impl Terminal<SharedInfo> for OnGoal {
    fn reset(&mut self, _initial_state: &ArenaState, _shared_info: &mut SharedInfo) {}

    fn is_terminal(&mut self, state: &ArenaState, _shared_info: &mut SharedInfo) -> bool {
        match state.game_mode() {
            GameMode::Soccar | GameMode::Heatseeker | GameMode::Snowday => {
                state.ball.pos.y.abs()
                    > consts::goal::SOCCAR_GOAL_SCORE_BASE_THRESHOLD_Y
                        + consts::ball::get_radius(state.game_mode())
            }
            GameMode::Hoops => {
                if state.ball.pos.z < consts::goal::HOOPS_GOAL_SCORE_THRESHOLD_Z {
                    ball_within_hoops_goal_xy_margin_eq(state.ball.pos.x, state.ball.pos.y) < 0.0
                } else {
                    false
                }
            }
            GameMode::Dropshot => {
                state.ball.pos.z < -consts::ball::get_radius(state.game_mode()) * 1.75
            }
            GameMode::TheVoid => false,
        }
    }
}

#[derive(Default)]
struct EpisodeDurationMax {
    episode_duration: f32,
}

impl Truncate<SharedInfo> for EpisodeDurationMax {
    fn reset(&mut self, _initial_state: &ArenaState, shared_info: &mut SharedInfo) {
        self.episode_duration = shared_info.rng.random_range(0.0..5.0);
    }

    fn should_truncate(&mut self, state: &ArenaState, shared_info: &mut SharedInfo) -> bool {
        const SECS_TO_MIN: f32 = 1.0 / 60.0;

        let elapsed =
            (state.tick_count - shared_info.start_tick) as f32 * consts::TICK_TIME * SECS_TO_MIN;

        // reset after some minutes
        if elapsed < self.episode_duration {
            return false;
        }

        state.ball.pos.z < 94.5
    }
}

fn main() {
    const RENDER: bool = false;
    const GAME_SPEED: u8 = 2;

    init_from_default(cfg!(not(debug_assertions))).unwrap();

    // configure arena to our liking
    let mut arena = Arena::new(GameMode::Soccar);

    arena.add_car(Team::Orange, CarBodyConfig::OCTANE);
    arena.add_car(Team::Blue, CarBodyConfig::OCTANE);

    if RENDER {
        arena.set_vis_enabled(true);
    }

    let mut env = Env::new(
        arena,
        MyStateSetter,
        MyObs,
        MyAction::default(),
        new_rewards!((DistanceToBallReward, 1.0)),
        OnGoal,
        EpisodeDurationMax::default(),
        SharedInfo::default(),
    );

    let (mut state, mut obs) = env.reset();

    // extra render stuff
    // this method ensures no game speed slowdowns
    // and no weirdness from different game speeds
    let tick_rate = Duration::from_secs_f32(consts::TICK_TIME / f32::from(GAME_SPEED));
    let mut next_time = Instant::now() + tick_rate;

    let ticks_per_min = MyAction::get_tick_skip() as f32 * consts::TICK_TIME / 60.0;
    let start_time = Instant::now();
    let mut prev_time = Instant::now();
    let mut total_steps = 0u64;

    let mut action_rng = rand::rng().sample_iter(Uniform::new(0usize, 24).unwrap());

    loop {
        // random actions
        let actions = action_rng.by_ref().take(obs.len()).collect::<Vec<_>>();

        let result = if RENDER {
            env.pre_step(&state, &actions);

            for _ in 0..env.get_tick_skip() {
                env.step_arena_one_tick();

                if RENDER {
                    // ensure we only run at the requested game speed
                    let wait_time = next_time - Instant::now();
                    if wait_time > Duration::default() {
                        sleep(wait_time);
                    }
                    next_time += tick_rate;
                }
            }

            env.post_step()
        } else {
            env.step(&state, &actions)
        };

        total_steps += 1;
        if result.is_terminal || result.truncated {
            (state, obs) = env.reset();
        } else {
            obs = result.obs;
            state = result.state;
        }

        if Instant::now() - prev_time > Duration::from_secs(5) {
            let elapsed = (Instant::now() - start_time).as_secs_f32();
            let steps_per_sec = total_steps as f32 / elapsed;
            let min_per_sec = steps_per_sec * ticks_per_min;
            println!(
                "Steps: {}, Steps/s: {:.2}, Elapsed: {:.0}s, Simulated min/s: {:.3}",
                total_steps, steps_per_sec, elapsed, min_per_sec
            );

            prev_time = Instant::now();
        }
    }
}
