# rlgym_rs

Rocket League RL environments in Rust. Fast physics, clean traits.

Built on [`rocketsim`](https://github.com/VirxEC/rocketsim).

---

## TL;DR

- You build an `Env` from **7 small pieces**.
- Loop is just: `reset()` → `step()` → repeat.
- Start here: `examples/generic.rs`.

```bash
cargo run --example generic --release
```

---

## Quickstart

**1. Init physics once** (needs `collision_meshes/`):

```rust
use rlgym::rocketsim::{Arena, GameMode, init_from_default};

init_from_default(true).unwrap();
let mut arena = Arena::new(GameMode::Soccar);
```

**2. Build your env:**

```rust
let mut env = Env::new(
    arena,
    MyStateSetter,
    MyObs,
    MyAction::default(),
    MyReward,
    MyTerminal,
    MyTruncate,
    MyShared::default(),
);
```

**3. Run it:**

```rust
let (mut state, mut obs, mut masks) = env.reset();

loop {
    let actions = pick_actions(&masks); // your policy goes here
    let result = env.step(&state, &actions);

    state = result.state;
    obs = result.obs;
    masks = result.action_masks;

    if result.is_terminal || result.truncated {
        (state, obs, masks) = env.reset();
    }
}
```

That's the whole loop.

---

## How it works

`Env` is just a container. **You** define the behavior.

| # | Piece | Job in one line |
|---|-------|-----------------|
| 1 | `StateSetter` | Set up kickoffs / random spawns on `reset()` |
| 2 | `Obs` | Turn `GameState` into `Vec<Vec<f32>>` |
| 3 | `Action` | Map policy ints → `CarControls`, define masks |
| 4 | `Reward` | Turn `GameState` into `Vec<f32>` |
| 5 | `Terminal` | Is the episode over? (e.g. goal scored) |
| 6 | `Truncate` | Time limit? (e.g. ball on ground after X min) |
| 7 | `SharedInfo` | Shared memory between all of the above |

All pieces get `&GameState` + `&mut SharedInfo`.

See `examples/generic.rs` for a working version of all 7.

---

## The step, in 3 parts

Normally you just call `env.step()`.

Under the hood it is:

```
pre_step() → step_physics() → post_step()
```

Use the split version when you need control:

- **Action delay?** Split physics into `delay` + `rest` ticks.
- **Rendering?** Step one tick at a time in your render loop.
- **Custom collector?** Call `pre_step_neutral()` after reset.

```rust
// full control, same as env.step():
env.pre_step(&state, &actions);
env.step_physics(env.get_tick_skip());
let result = env.post_step();
```

> ✅ Rule: use `env.step_tick()` / `env.step_physics()`.
>
> ❌ Avoid `env.arena.step_tick()` directly.
> It skips `on_tick` callbacks and event tracking.

---

## Per-tick data without the cost

Rewards only see step boundaries by default.

If tick_skip = 8, you miss the 7 ticks in between.

**Fix:** accumulate in `SharedInfo::on_tick`.

```rust
impl SharedInfoProvider for MyShared {
    fn on_tick(&mut self, arena: &Arena, tick_events: &[ArenaEvent]) {
        // cheap: no GameState allocation
        // e.g. count touches, track max ball speed
    }
}
```

- Called **once per physics tick** inside `step_physics`.
- Gets `&Arena` (read-only) + that tick's events.
- Consume it later in `Reward::get_rewards` or `update`.
- Default is no-op, so old code still compiles.

---

## Key types

**`GameState`** — what you actually get each step:

```rust
pub struct GameState {
    pub game_mode: GameMode,
    pub tick_count: u64,
    pub ball: BallState,
    pub cars: Vec<(CarInfo, CarState)>,
    pub boost_pads: Vec<(BoostPadConfig, BoostPadState)>,
    pub events: Vec<ArenaEvent>,
}
```

Helpers: `is_ball_scored()`, `num_cars()`, `num_boost_pads()`.

**`StepResult`** — what `step()` returns:

- `obs: Vec<Vec<f32>>` — one obs per car
- `action_masks: Vec<Vec<bool>>`
- `rewards: Vec<f32>` — one per car
- `is_terminal: bool`
- `truncated: bool`
- `state: GameState`

**`Action`** — two knobs that matter:

```rust
fn get_tick_skip() -> u8;    // physics ticks per decision, e.g. 8
fn get_action_delay() -> u8; // extra latency sim, default 0
```

---

## Common gotchas

- [ ] Did you call `init_from_default()` before `Arena::new()`?
- [ ] Running from the wrong folder? `collision_meshes/` must be found.
- [ ] NaN in obs/rewards? Debug asserts will catch it — check div-by-zero.
- [ ] Events empty? You used `arena.step_tick()` instead of `env.step_tick()`.
- [ ] Reward needs touches? Don't use step boundaries — use `on_tick`.
- [ ] Slow render? RLViser is debug-only. Don't use it for training.

---

## RLViser (debug rendering)

Off by default. On only for debugging.

```rust
env.set_rlviser_enabled(true);
env.handle_rlviser_messages()?;
if env.rlviser_paused() { /* ... */ }
```

> Slow. Real-time only. Never for training throughput.

---

## Project layout

```
src/lib.rs            <- everything: Env, GameState, 7 traits
examples/generic.rs   <- copy-paste starter (1v1 Soccar)
collision_meshes/     <- required physics data
```

No modules. One file. Easy to read top to bottom.

---

## Next step

Open `examples/generic.rs`.

Replace `DistanceToBallReward` with your idea.

Run it:

```bash
cargo run --example generic --release
```
