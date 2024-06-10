import collections
import functools
import logging
import os
import pathlib
import re
import sys
import warnings
import time

try:
    import rich.traceback
    rich.traceback.install()
except ImportError:
    pass

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"  # tensorboard
logging.getLogger().setLevel("ERROR")
warnings.filterwarnings("ignore", ".*box bound precision lowered.*")

os.environ['MKL_SERVICE_FORCE_INTEL'] = '1'
os.environ['MUJOCO_GL'] = 'egl'

sys.path.append(str(pathlib.Path(__file__).parent))
sys.path.append(str(pathlib.Path(__file__).parent.parent))

import numpy as np
import ruamel.yaml as yaml
import torch
import random
from tqdm import trange

import wmlib
import wmlib.envs as envs
import wmlib.agents as agents
import wmlib.utils as utils


def main():

    configs = yaml.safe_load(
        (pathlib.Path(sys.argv[0]).parent.parent / "configs" / "dreamerv2.yaml").read_text()
    )
    parsed, remaining = utils.Flags(configs=["defaults"]).parse(known_only=True)
    config = utils.Config(configs["defaults"])
    for name in parsed.configs:
        config = config.update(configs[name])
    config = utils.Flags(config).parse(remaining)

    logdir = pathlib.Path(config.logdir).expanduser()
    load_logdir = pathlib.Path(config.load_logdir).expanduser()
    logdir.mkdir(parents=True, exist_ok=True)
    config.save(logdir / "config.yaml")
    print(config, "\n")
    print("Logdir", logdir)

    utils.snapshot_src(".", logdir / "src", ".gitignore")

    message = "No GPU found. To actually train on CPU remove this assert."
    assert torch.cuda.is_available(), message  # FIXME
    assert config.precision in (16, 32), config.precision
    if config.precision == 16:
        wmlib.ENABLE_FP16 = True  # enable fp16 here since only cuda can use fp16
        print("setting fp16")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if device != "cpu":
        torch.set_num_threads(1)

    # reproducibility
    seed = config.seed
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.cuda.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True  # no apparent impact on speed
    torch.backends.cudnn.benchmark = True  # faster, increases memory though.., no impact on seed

    train_replay = wmlib.Replay(logdir / "train_episodes", seed=seed, **config.replay)
    eval_replay = wmlib.Replay(logdir / "eval_episodes", seed=seed, **dict(
        capacity=config.replay.capacity // 10,
        minlen=config.dataset.length,
        maxlen=config.dataset.length,
        save_video=config.eval_only,
        save_episode=False))
    step = utils.Counter(train_replay.stats["total_steps"])
    outputs = [
        utils.TerminalOutput(),
        utils.JSONLOutput(logdir),
        utils.TensorBoardOutputPytorch(logdir),
    ]
    logger = utils.Logger(step, outputs, multiplier=config.action_repeat)
    metrics = collections.defaultdict(list)

    should_train = utils.Every(config.train_every)
    should_log = utils.Every(config.log_every)
    should_video_train = utils.Every(config.eval_every)
    should_video_eval = utils.Every(config.eval_every)
    should_expl = utils.Until(config.expl_until // config.action_repeat)

    # save experiment used config
    with open(logdir / "used_config.yaml", "w") as f:
        f.write("## command line input:\n## " + " ".join(sys.argv) + "\n##########\n\n")
        yaml.dump(config, f)

    def make_env(mode, env_id):
        print(f"create {mode} env {env_id}")
        suite, task = config.task.split("_", 1)
        if suite == "dmc":
            env = envs.DMC(
                task, config.action_repeat, config.render_size, config.dmc_camera)
            env = envs.NormalizeAction(env)
        elif suite == "dmcnbg":
            env = envs.NaturalBackgroundDMC(
                task, config.action_repeat, config.render_size, config.dmc_camera)
            env = envs.NormalizeAction(env)
        elif suite == "metaworld":
            task = "-".join(task.split("_"))
            env = envs.MetaWorld(
                task,
                config.seed + env_id,
                config.action_repeat,
                config.render_size,
                config.camera,
            )
            env = envs.NormalizeAction(env)
        elif suite == "rlbench":
            env = envs.RLBench(
                task,
                config.render_size,
                config.action_repeat,
            )
            env = envs.NormalizeAction(env)
        elif suite == "dmcr":
            env = envs.DMCRemastered(task, config.action_repeat, config.render_size,
                                     config.dmc_camera, config.dmcr_vary)
            env = envs.NormalizeAction(env)
        elif suite == "minecraft":
            env = envs.Minecraft(task, config.seed, config.action_repeat, config.render_size, config.sim_size, config.eval_hard_reset_every if mode == "eval" else config.hard_reset_every)
            env = envs.MultiDiscreteAction(env)
        else:
            raise NotImplementedError(suite)
        env = envs.TimeLimit(env, config.time_limit)
        return env

    def per_episode(ep, mode, env_id=None):
        length = len(ep["reward"]) - 1
        score = float(ep["reward"].astype(np.float64).sum())
        if "metaworld" in config.task or 'rlbench' in config.task or 'minecraft' in config.task:
            success = float(np.sum(ep["success"]) >= 1.0)
            print(
                f"[{time.ctime()}/Env{env_id}] {mode.title()} episode has {float(success)} success, {float(ep['is_terminal'][-1])} terminal, {length} steps and return {score:.1f}."
            )
            logger.scalar(f"{mode}_success", float(success))
        else:
            print(f"{mode.title()} episode has {length} steps and return {score:.1f}.")
        logger.scalar(f"{mode}_return", score)
        logger.scalar(f"{mode}_length", length)
        for key, value in ep.items():
            if re.match(config.log_keys_sum, key):
                logger.scalar(f"sum_{mode}_{key}", ep[key].sum())
            if re.match(config.log_keys_mean, key):
                logger.scalar(f"mean_{mode}_{key}", ep[key].mean())
            if re.match(config.log_keys_max, key):
                logger.scalar(f"max_{mode}_{key}", ep[key].max(0).mean())
        should = {"train": should_video_train, "eval": should_video_eval}[mode]
        if should(step):
            for key in config.log_keys_video:
                logger.video(f"{mode}_policy_{key}", ep[key])
        replay = dict(train=train_replay, eval=eval_replay)[mode]
        logger.add(replay.stats, prefix=mode)
        if mode != 'eval':  # NOTE: to aggregate eval results at last
            logger.write()

    print("Create envs.")
    num_eval_envs = min(config.envs, config.eval_eps)
    if config.envs_parallel == "none":
        train_envs = [make_env("train", _) for _ in range(config.envs)]
        if config.eval_envs_parallel == "none":
            eval_envs = [make_env("eval", _) for _ in range(num_eval_envs)]
        else:
            make_async_env = lambda mode, env_id: envs.Async(
                functools.partial(make_env, mode, env_id), config.eval_envs_parallel)
            num_eval_envs = min(config.eval_envs, config.eval_eps)
            eval_envs = [make_async_env("eval", _) for _ in range(num_eval_envs)]
    else:
        make_async_env = lambda mode, env_id: envs.Async(
            functools.partial(make_env, mode, env_id), config.envs_parallel)
        train_envs = [make_async_env("train", _) for _ in range(config.envs)]
        eval_envs = [make_async_env("eval", _) for _ in range(num_eval_envs)]
    act_space = train_envs[0].act_space
    obs_space = train_envs[0].obs_space
    train_driver = wmlib.Driver(train_envs, device)
    train_driver.on_episode(lambda ep, env_id: per_episode(ep, mode="train", env_id=env_id))
    train_driver.on_step(lambda tran, worker: step.increment())
    train_driver.on_step(train_replay.add_step)
    train_driver.on_reset(train_replay.add_step)
    eval_driver = wmlib.EvalDriver(eval_envs, device, init_reset_blocking=('minecraft' in config.task))
    eval_driver.on_episode(lambda ep, env_id: per_episode(ep, mode="eval", env_id=env_id))
    eval_driver.on_episode(lambda ep, env_id: eval_replay.add_episode(ep))

    prefill = max(0, config.prefill - train_replay.stats["total_steps"])
    if prefill and not config.eval_only:
        print(f"Prefill dataset ({prefill} steps).")
        random_agent = agents.RandomAgent(act_space)
        train_driver(random_agent, steps=prefill, episodes=1)
        eval_driver(random_agent, episodes=1)
        train_driver.reset()
        eval_driver.reset()

    print("Create agent.")
    train_dataset = iter(train_replay.dataset(**config.dataset))
    report_dataset = iter(train_replay.dataset(**config.dataset))
    eval_dataset = iter(eval_replay.dataset(pin_memory=False, **config.dataset))

    def next_batch(iter, fp16=True):
        # casts to fp16 and cuda
        dtype = torch.float16 if wmlib.ENABLE_FP16 and fp16 else torch.float32  # only on cuda
        out = {k: v.to(device=device, dtype=dtype) for k, v in next(iter).items()}
        return out

    # the agent needs 1. init modules 2. go to device 3. set optimizer
    agnt = agents.DreamerV2(config, obs_space, act_space, step)
    agnt = agnt.to(device)
    agnt.init_optimizers()

    if not config.eval_only:
        train_agent = wmlib.CarryOverState(agnt.train)
        train_agent(next_batch(train_dataset))  # do initial benchmarking pass
        torch.cuda.empty_cache()  # clear cudnn bechmarking cache
    if (logdir / "variables.pt").exists():
        print("Load agent.")
        print(agnt.load_state_dict(torch.load(logdir / "variables.pt")))
    elif (load_logdir / "variables.pt").exists():
        print("Load agent.")
        print(agnt.load_state_dict(torch.load(load_logdir / "variables.pt")))
    else:
        print("Pretrain agent.")
        for _ in trange(config.pretrain, desc="pretrain"):
            train_agent(next_batch(train_dataset))
    train_replay._ongoing_eps.clear()
    train_policy = lambda *args: agnt.policy(
        *args, mode="explore" if should_expl(step) else "train")
    eval_policy = lambda *args: agnt.policy(*args, mode="eval")

    def train_step(tran, worker):
        if should_train(step):
            for _ in range(config.train_steps):
                mets = train_agent(next_batch(train_dataset))
                [metrics[key].append(value) for key, value in mets.items()]
        if should_log(step):
            for name, values in metrics.items():
                logger.scalar(name, np.array(values, np.float64).mean())
                metrics[name].clear()
            logger.add(agnt.report(next_batch(report_dataset)), prefix="train")
            logger.write(fps=True)
    train_driver.on_step(train_step)

    try:
        while step < config.steps:
            logger.write()
            print("Start evaluation.")
            eval_driver(eval_policy, episodes=config.eval_eps)
            if not config.eval_only:
                logger.add(agnt.report(next_batch(eval_dataset)), prefix="eval")
            eval_driver.reset()
            logger.write()  # NOTE: to aggregate eval results

            if config.eval_only:
                exit()

            if config.stop_steps != -1 and step >= config.stop_steps:
                break
            else:
                print("Start training.")
                train_driver(train_policy, steps=config.eval_every)
                torch.save(agnt.state_dict(), logdir / "variables.pt")
    except KeyboardInterrupt:
        print("Keyboard Interrupt - saving agent")
        if not config.eval_only:
            torch.save(agnt.state_dict(), logdir / "variables_interrupt.pt")
    except Exception as e:
        print("Training Error:", e)
        if not config.eval_only:
            torch.save(agnt.state_dict(), logdir / "variables_error.pt")
        raise e
    finally:
        for env in train_envs + eval_envs:
            try:
                env.finish()
            except Exception:
                try:
                    env.close()
                except Exception:
                    pass

    if not config.eval_only:
        torch.save(agnt.state_dict(), logdir / "variables_final.pt")


if __name__ == "__main__":
    main()
