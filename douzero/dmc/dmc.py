import os
import sys
import threading
import time
import timeit
import pprint
from collections import deque
import warnings

import torch
from torch import multiprocessing as mp
from torch import nn
import pickle
import random

from .file_writer import FileWriter
from .models import Model
from .utils import get_batch, log, create_env, create_optimizers, act
import client_helper
import bit_helper
import requests
from douzero.env.env import env_version



mean_episode_return_buf = {p: deque(maxlen=100) for p in ['landlord', 'landlord_up', 'landlord_down']}
model_version = 0
models = {}
warnings.filterwarnings("ignore", category=UserWarning)

def compute_loss(logits, targets):
    loss = ((logits.squeeze(-1) - targets) ** 2).mean()
    return loss


batches = []
program_version = "3.0.0"
updating = False

def learn(position, actor_models, model, batch, optimizer, flags, lock):
    global model_version, models, batches
    batches.append({
        "position": position,
        "batch": batch
    })

    return {
        'mean_episode_return_' + position: 0,
        'loss_' + position: 0,
    }


def train(flags):
    """
    This is the main funtion for training. It will first
    initilize everything, such as buffers, optimizers, etc.
    Then it will start subprocesses as actors. Then, it will call
    learning function with  multiple threads.
    """
    global models
    plogger = FileWriter(
        xpid=flags.xpid,
        xp_args=flags.__dict__,
        rootdir=flags.savedir,
    )
    checkpointpath = os.path.expandvars(
        os.path.expanduser('%s/%s/%s' % (flags.savedir, flags.xpid, 'model.tar')))

    T = flags.unroll_length
    B = flags.batch_size
    print(flags.actor_device_cpu)
    if flags.actor_device_cpu:
        device_iterator = ['cpu']
    else:
        device_iterator = range(flags.num_actor_devices)
        assert flags.num_actor_devices <= len(flags.gpu_devices.split(',')), 'The number of actor devices can not exceed the number of available devices'

    def update_model(ver, urls, force):
        global model_version, models, updating
        if updating:
            return
        updating = True
        if model_version != ver or force:
            print("检测到模型更新")
            if len(urls) > 0:
                url = urls[random.randint(0, len(urls)-1)]
            else:
                print("模型更新失败：没有有效的模型地址")
                updating = False
                return
            print("更新中，请耐心等待")
            st = time.time()
            weights = client_helper.download_pkl(url)
            if weights is not None:
                model_version = ver
                for position in ["landlord", "landlord_up", "landlord_down", "bidding"]:
                    if flags.actor_device_cpu:
                        models["cpu"].get_model(position).load_state_dict(weights[position])
                        torch.save(weights[position], "./models/" + position + ".ckpt")
                    else:
                        for device in range(flags.num_actor_devices):
                            models[device].get_model(position).load_state_dict(weights[position])
                            torch.save(weights[position], "./models/" + position + ".ckpt")
                with open("./model_version.txt", "w") as f:
                    f.write(str(model_version))
                print("更新模型成功！耗时: %.1f s" % (time.time() - st))
            else:
                print("更新模型失败！")
        updating = False

    def load_actor_models():
        global model_version, models
        if os.path.exists("./model_version.txt"):
            with open("./model_version.txt", "r") as f:
                model_version = int(f.read())
        print("初始化，正在获取服务器版本")
        model_info = client_helper.get_model_info()
        if model_info is not None:
            print("版本获取完成，服务器版本:", model_info["version"])
            update_model(model_info["version"], model_info["urls"], False)
        else:
            print("服务器版本获取失败，更新模型失败")
            return
        if not (os.path.exists("./models/landlord.ckpt") and os.path.exists(
                "./models/landlord_up.ckpt") and os.path.exists("./models/landlord_down.ckpt") and os.path.exists("./models/bidding.ckpt")):
            update_model(model_info["version"], model_info["urls"], True)

    # def check_update_model(force=False):
    #     global model_version, models
    #     if os.path.exists("./model_version.txt"):
    #         with open("./model_version.txt", "r") as f:
    #             model_version = int(f.read())
    #     print("版本比对中")
    #     model_info = client_helper.get_model_info()
    #     if model_info is not None:
    #         if model_info["program_version"] != program_version:
    #             print("客户端版本不正确！请从Github重新拉取！")
    #             return
    #         print("服务器版本:", model_info["version"])
    #         update_model(model_info["version"], model_info["urls"], force)
    #     else:
    #         print("版本比对失败，更新模型失败")
    #     if not (os.path.exists("./models/landlord.ckpt") and os.path.exists(
    #             "./models/landlord_up.ckpt") and os.path.exists("./models/landlord_down.ckpt")):
    #         update_model(model_info["version"], model_info["urls"], True)

    # Initialize actor models
    global models
    models = {}
    for device in device_iterator:
        model = Model(device="cpu")
        model.share_memory()
        model.eval()
        models[device] = model

    # Initialize queues
    actor_processes = []
    ctx = mp.get_context('spawn')
    batch_queues = {"landlord": ctx.SimpleQueue(), "landlord_up": ctx.SimpleQueue(), "landlord_down": ctx.SimpleQueue(), "bidding": ctx.SimpleQueue()}

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
        'mean_episode_return_bidding',
        'loss_bidding',
    ]
    frames, stats = 0, {k: 0 for k in stat_keys}
    position_frames = {'landlord': 0, 'landlord_up': 0, 'landlord_down': 0, 'bidding': 0}
    global model_version
    # Load models if any
    if flags.load_model:
        print("加载模型中，请稍后")
        load_actor_models()
    for position in ["landlord", "landlord_up", "landlord_down", 'bidding']:
        if flags.actor_device_cpu:
            models["cpu"].get_model(position).load_state_dict(torch.load("./models/" + position + ".ckpt", map_location="cpu"))
        else:
            for device in device_iterator:
                models[device].get_model(position).load_state_dict(torch.load("./models/" + position + ".ckpt", map_location="cuda:"+str(device)))

    # Starting actor processes
    if flags.actor_device_cpu:
        flags.num_actor_devices = 1
    for device in device_iterator:
        num_actors = flags.num_actors
        for i in range(flags.num_actors):
            actor = ctx.Process(
                target=act,
                args=(i, device, batch_queues, models[device], flags))
            actor.start()
            actor_processes.append(actor)

    def upload_batch_loop(flags):
        global model_version, models
        while True:
            try:
                if len(batches) > 0:
                    my_batches = []
                    my_batches.extend(batches)
                    batches.clear()
                    ver, urls = client_helper.handle_batches(my_batches, model_version, program_version)
                    st = time.time()
                    if len(urls) > 0:
                        if ver != model_version:
                            print("新模型:", ver)
                            update_model(ver, urls, True)
                            print("更新完成！耗时: %.1f s" % (time.time() - st))
                    else:
                        print("没有收到模型下载地址")
                else:
                    print("没有新Batch")
            except Exception as e:
                print("在处理Batch时出现错误:", repr(e))
            time.sleep(15)

    def update_env(env_ver, url, force=False):
        if env_ver != env_version or force:
            try:
                req = requests.get(url)
                data = req.content
                if len(data) > 10000:
                    with open("douzero/env/env.py", "wb") as f:
                        f.write(data)
                    print("更新Env文件，重启客户端")
                    os.execl(sys.executable, sys.executable, *sys.argv)
                    time.sleep(1)
                    exit()
                else:
                    print("更新Env文件时出错: ", data)
            except Exception as e:
                print("更新Env文件时出错: ", repr(e))

    def check_model_update_loop():
        while True:
            try:
                info = client_helper.get_model_info()
                if info is not None:
                    if "program_version" in info:
                        if info["program_version"] != program_version:
                            print("客户端版本过时，请从Github重新拉取")
                    # ver, urls = info["version"], info["urls"]
                    # update_model(ver, urls, False)
                    env_ver = info["env_version"]
                    update_env(env_ver, info["env_url"])
            except Exception as e:
                print("在检查模型更新时出现错误: ", repr(e))
            time.sleep(300)



    def batch_and_learn(i, device, position, local_lock, position_lock, lock=threading.Lock()):
        """Thread target for the learning process."""
        nonlocal frames, position_frames, stats
        while frames < flags.total_frames:
            batch = get_batch(batch_queues, position, flags, local_lock)
            _stats = learn(position, models, learner_model.get_model(position), batch,
                           optimizers[position], flags, position_lock)
            with lock:
                for k in _stats:
                    stats[k] = _stats[k]
                to_log = dict(frames=frames)
                to_log.update({k: stats[k] for k in stat_keys})
                plogger.log(to_log)
                frames += T * B
                position_frames[position] += T * B

    thread_upload = threading.Thread(target=upload_batch_loop, args=(flags,))
    thread_upload.setDaemon(True)
    thread_upload.start()
    thread_update_model = threading.Thread(target=check_model_update_loop)
    thread_update_model.setDaemon(True)
    thread_update_model.start()

    threads = []
    locks = {}
    for device in device_iterator:
        locks[device] = {'landlord': threading.Lock(), 'landlord_up': threading.Lock(), 'landlord_down': threading.Lock(), 'bidding': threading.Lock()}
    position_locks = {'landlord': threading.Lock(), 'landlord_up': threading.Lock(), 'landlord_down': threading.Lock(), 'bidding': threading.Lock()}

    for device in device_iterator:
        for i in range(flags.num_threads):
            for position in ['landlord', 'landlord_up', 'landlord_down', 'bidding']:
                thread = threading.Thread(
                    target=batch_and_learn, name='batch-and-learn-%d' % i, args=(i,device,position,locks[device][position],position_locks[position]))
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
            time.sleep(10)

            if timer() - last_checkpoint_time > flags.save_interval * 60:
                checkpoint(frames)
                last_checkpoint_time = timer()

            end_time = timer()
            fps = (frames - start_frames) / (end_time - start_time)
            fps_avg = 0
            fps_log.append(fps)
            if len(fps_log) > 30:
                fps_log = fps_log[1:]
            for fps_record in fps_log:
                fps_avg += fps_record
            fps_avg = fps_avg / len(fps_log)
            position_fps = {k: (position_frames[k] - position_start_frames[k]) / (end_time - start_time) for k in
                            position_frames}
            log.info("本机速度 %.1f fps", fps_avg)
            if fps_avg == 0:
                print("本机速度在训练的前几分钟为0是正常现象，请稍后")
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
