import os
import threading
import time
import timeit
import pprint
from collections import deque

import torch
from torch import multiprocessing as mp
from torch import nn

from .file_writer import FileWriter
from .models import Model
from .utils import get_batch, log, create_env, create_buffers, create_optimizers, act
import client_helper
import bit_helper

mean_episode_return_buf = {p: deque(maxlen=100) for p in ['landlord', 'landlord_up', 'landlord_down']}
model_version = 0
models = []


def compute_loss(logits, targets):
    loss = ((logits.squeeze(-1) - targets) ** 2).mean()
    return loss


batches = []


def learn(position, actor_models, model, batch, optimizer, flags, lock):
    """Performs a learning (optimization) step."""
    global model_version, models, batches
    batches.append({
        "position": position,
        "batch": batch
    })
    return {
        'mean_episode_return_' + position: 0,
        'loss_' + position: 0,
    }
    if flags.training_device != "cpu":
        device = torch.device('cuda:' + str(flags.training_device))
    else:
        device = torch.device('cpu')
    obs_x_no_action = batch['obs_x_no_action'].to(device)
    obs_action = batch['obs_action'].to(device)
    obs_x = torch.cat((obs_x_no_action, obs_action), dim=2).float()
    obs_x = torch.flatten(obs_x, 0, 1)
    obs_z = torch.flatten(batch['obs_z'].to(device), 0, 1).float()
    target = torch.flatten(batch['target'].to(device), 0, 1)
    episode_returns = batch['episode_return'][batch['done']]
    mean_episode_return_buf[position].append(torch.mean(episode_returns).to(device))

    with lock:
        learner_outputs = model(obs_z, obs_x, return_value=True)
        loss = compute_loss(learner_outputs['values'], target)
        stats = {
            'mean_episode_return_' + position: torch.mean(
                torch.stack([_r for _r in mean_episode_return_buf[position]])).item(),
            'loss_' + position: loss.item(),
        }

        optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), flags.max_grad_norm)
        optimizer.step()

        for actor_model in actor_models:
            actor_model.get_model(position).load_state_dict(model.state_dict())
        return stats


def train(flags):
    """
    This is the main funtion for training. It will first
    initilize everything, such as buffers, optimizers, etc.
    Then it will start subprocesses as actors. Then, it will call
    learning function with  multiple threads.
    """
    plogger = FileWriter(
        xpid=flags.xpid,
        xp_args=flags.__dict__,
        rootdir=flags.savedir,
    )
    checkpointpath = os.path.expandvars(
        os.path.expanduser('%s/%s/%s' % (flags.savedir, flags.xpid, 'model.tar')))

    T = flags.unroll_length
    B = flags.batch_size

    def update_model(ver, url, force):
        global model_version, models
        if model_version != ver or force:
            print("检测到模型更新")
            st = time.time()
            model_version = ver
            weights = client_helper.download_pkl(url)
            for position in ["landlord", "landlord_up", "landlord_down"]:
                if flags.actor_device_cpu:
                    models[0].get_model(position).load_state_dict(weights[position])
                    torch.save(weights[position], "./models/" + position + ".ckpt")
                else:
                    for device in range(flags.num_actor_devices):
                        models[device].get_model(position).load_state_dict(weights[position])
                        torch.save(weights[position], "./models/" + position + ".ckpt")
            with open("./model_version.txt", "w") as f:
                f.write(str(model_version))
            print("更新模型成功！耗时: %.1f s" % (time.time() - st))

    def load_actor_models():
        global model_version, models
        if os.path.exists("./model_version.txt"):
            with open("./model_version.txt", "r") as f:
                model_version = int(f.read())
        print("版本比对中")
        model_info = client_helper.get_model_info()
        print("版本比对完成", model_info, model_info["model_version"])
        if model_info is not None:
            update_model(model_info["model_version"], model_info["model_url"], True)
        else:
            print("版本比对失败，更新模型失败")
        if not (os.path.exists("./models/landlord.ckpt") and os.path.exists(
                "./models/landlord_up.ckpt") and os.path.exists("./models/landlord_down.ckpt")):
            update_model(model_info["model_version"], model_info["model_url"], True)

    def check_update_model(force=False):
        global model_version, models
        if os.path.exists("./model_version.txt"):
            with open("./model_version.txt", "r") as f:
                model_version = int(f.read())
        print("版本比对中")
        model_info = client_helper.get_model_info()
        print("版本比对完成", model_info, model_info["model_version"])
        if model_info is not None:
            update_model(model_info["model_version"], model_info["model_url"], force)
        else:
            print("版本比对失败，更新模型失败")
        if not (os.path.exists("./models/landlord.ckpt") and os.path.exists(
                "./models/landlord_up.ckpt") and os.path.exists("./models/landlord_down.ckpt")):
            update_model(model_info["model_version"], model_info["model_url"], True)

    # Initialize actor models
    global models
    if not flags.actor_device_cpu:
        assert flags.num_actor_devices <= len(
            flags.gpu_devices.split(',')), 'The number of actor devices can not exceed the number of available devices'
        for device in range(flags.num_actor_devices):
            model = Model(device=device)
            model.share_memory()
            model.eval()
            models.append(model)
    else:
        model = Model(device="cpu")
        model.share_memory()
        model.eval()
        models.append(model)

    # Initialize buffers
    buffers = create_buffers(flags)

    # Initialize queues
    actor_processes = []
    ctx = mp.get_context('spawn')
    free_queue = []
    full_queue = []
    for device in range(flags.num_actor_devices):
        _free_queue = {'landlord': ctx.SimpleQueue(), 'landlord_up': ctx.SimpleQueue(),
                       'landlord_down': ctx.SimpleQueue()}
        _full_queue = {'landlord': ctx.SimpleQueue(), 'landlord_up': ctx.SimpleQueue(),
                       'landlord_down': ctx.SimpleQueue()}
        free_queue.append(_free_queue)
        full_queue.append(_full_queue)

    # Learner model for training
    learner_model = Model(device=flags.training_device)

    # Create optimizers
    optimizers = create_optimizers(flags, learner_model)

    # Stat Keys
    stat_keys = [
        'mean_episode_return_landlord',
        'loss_landlord',
        'mean_episode_return_landlord_up',
        'loss_landlord_up',
        'mean_episode_return_landlord_down',
        'loss_landlord_down',
    ]
    frames, stats = 0, {k: 0 for k in stat_keys}
    position_frames = {'landlord': 0, 'landlord_up': 0, 'landlord_down': 0}
    global model_version
    # Load models if any
    if flags.load_model:
        print("更新模型中，请稍后")
        load_actor_models()
    for position in ["landlord", "landlord_up", "landlord_down"]:
        if flags.actor_device_cpu:
            models[0].get_model(position).load_state_dict(torch.load("./models/" + position + ".ckpt", map_location="cpu"))
        else:
            for device in range(flags.num_actor_devices):
                models[device].get_model(position).load_state_dict(torch.load("./models/" + position + ".ckpt", map_location="cuda:"+str(device)))

    # Starting actor processes
    if flags.actor_device_cpu:
        flags.num_actor_devices = 1
    for device in range(flags.num_actor_devices):
        num_actors = flags.num_actors
        for i in range(flags.num_actors):
            actor = ctx.Process(
                target=act,
                args=(i, device, free_queue[device], full_queue[device], models[device], buffers[device], flags))
            actor.start()
            actor_processes.append(actor)

    def upload_batch_loop(flags):
        global model_version, models
        while True:
            if len(batches) > 0:
                my_batches = []
                my_batches.extend(batches)
                batches.clear()
                ver, url = client_helper.handle_batches(my_batches, model_version)
                st = time.time()
                if ver != model_version and url != "":
                    print("新模型:", ver)
                    update_model(ver, url, True)
                    print("更新完成！耗时: %.1f s" % (time.time() - st))
            time.sleep(15)

    def check_model_update_loop():
        info = client_helper.get_model_info()
        if info is not None:
            ver, url = info["model_version"], info["model_url"]
            update_model(ver, url, False)

    def batch_and_learn(i, device, position, local_lock, position_lock, lock=threading.Lock()):
        """Thread target for the learning process."""
        nonlocal frames, position_frames, stats
        while frames < flags.total_frames:
            batch = get_batch(free_queue[device][position], full_queue[device][position], buffers[device][position],
                              flags, local_lock)
            _stats = learn(position, models, learner_model.get_model(position), batch, optimizers[position], flags,
                           position_lock)
            with lock:
                for k in _stats:
                    stats[k] = _stats[k]
                to_log = dict(frames=frames)
                to_log.update({k: stats[k] for k in stat_keys})
                plogger.log(to_log)
                frames += T * B
                position_frames[position] += T * B

    for device in range(flags.num_actor_devices):
        for m in range(flags.num_buffers):
            free_queue[device]['landlord'].put(m)
            free_queue[device]['landlord_up'].put(m)
            free_queue[device]['landlord_down'].put(m)

    thread_upload = threading.Thread(target=upload_batch_loop, args=(flags,))
    thread_upload.setDaemon(True)
    thread_upload.start()
    thread_update_model = threading.Thread(target=check_model_update_loop)
    thread_update_model.setDaemon(True)
    thread_update_model.start()

    threads = []
    locks = [{'landlord': threading.Lock(), 'landlord_up': threading.Lock(), 'landlord_down': threading.Lock()} for _ in
             range(flags.num_actor_devices)]
    position_locks = {'landlord': threading.Lock(), 'landlord_up': threading.Lock(), 'landlord_down': threading.Lock()}

    for device in range(flags.num_actor_devices):
        for i in range(flags.num_threads):
            for position in ['landlord', 'landlord_up', 'landlord_down']:
                thread = threading.Thread(
                    target=batch_and_learn, name='batch-and-learn-%d' % i,
                    args=(i, device, position, locks[device][position], position_locks[position]))
                thread.start()
                threads.append(thread)

    def checkpoint(frames):
        if flags.disable_checkpoint:
            return
        # log.info('Saving checkpoint to %s', checkpointpath)
        # _models = learner_model.get_models()
        # torch.save({
        #     'model_state_dict': {k: _models[k].state_dict() for k in _models},
        #     'optimizer_state_dict': {k: optimizers[k].state_dict() for k in optimizers},
        #     "stats": stats,
        #     'flags': vars(flags),
        #     'frames': frames,
        #     'position_frames': position_frames
        # }, checkpointpath)

        # Save the weights for evaluation purpose
        # for position in ['landlord', 'landlord_up', 'landlord_down']:
        #     model_weights_dir = os.path.expandvars(os.path.expanduser(
        #         '%s/%s/%s' % (flags.savedir, flags.xpid, position + '_weights_' + str(frames) + '.ckpt')))
        #     torch.save(learner_model.get_model(position).state_dict(), model_weights_dir)

    fps_log = []
    timer = timeit.default_timer
    try:
        last_checkpoint_time = timer() - flags.save_interval * 60
        while frames < flags.total_frames:
            start_frames = frames
            position_start_frames = {k: position_frames[k] for k in position_frames}
            start_time = timer()
            time.sleep(5)

            if timer() - last_checkpoint_time > flags.save_interval * 60:
                checkpoint(frames)
                last_checkpoint_time = timer()

            end_time = timer()
            fps = (frames - start_frames) / (end_time - start_time)
            fps_avg = 0
            fps_log.append(fps)
            if len(fps_log) > 24:
                fps_log = fps_log[1:]
            for fps_record in fps_log:
                fps_avg += fps_record
            fps_avg = fps_avg / len(fps_log)
            position_fps = {k: (position_frames[k] - position_start_frames[k]) / (end_time - start_time) for k in
                            position_frames}
            log.info("本机速度 %.1f fps", fps_avg)
            # log.info('After %i (L:%i U:%i D:%i) frames: @ %.1f fps (avg@ %.1f fps) (L:%.1f U:%.1f D:%.1f) Stats:\n%s',
            #          frames,
            #          position_frames['landlord'],
            #          position_frames['landlord_up'],
            #          position_frames['landlord_down'],
            #          fps,
            #          fps_avg,
            #          position_fps['landlord'],
            #          position_fps['landlord_up'],
            #          position_fps['landlord_down'],
            #          pprint.pformat(stats))

    except KeyboardInterrupt:
        return
    else:
        for thread in threads:
            thread.join()
        log.info('Learning finished after %d frames.', frames)

    checkpoint(frames)
    plogger.close()
