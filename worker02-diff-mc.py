#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jun 25 16:00:01 2020

@author: garima
"""

import os
import time
import numpy as np
import torch
import torch.nn as nn
import tqdm
from apex import amp
from deep_privacy import config_parser, logger, utils, torch_utils
from deep_privacy.data_tools.data_utils import denormalize_img
from deep_privacy.data_tools.dataloaders import load_dataset
from deep_privacy.metrics import fid
from deep_privacy.models import loss
from deep_privacy.models.generator import Generator
from deep_privacy.models.unet_model import init_model
from deep_privacy.torch_utils import to_cuda
from deep_privacy.utils import (amp_state_has_overflow, load_checkpoint,
                                save_checkpoint, wrap_models)
import ray

torch.manual_seed(0)
np.random.seed(0)

if False:
    torch.backends.cudnn.benchmark = True
    torch.backends.cudnn.deterministic = True
    torch.set_printoptions(precision=10)
else:
    torch.backends.cudnn.benchmark = True

class Trainer:

    def __init__(self, config):
        # Set Hyperparameters
        self.batch_size_schedule = config.train_config.batch_size_schedule
        self.dataset = config.dataset
        self.learning_rate = config.train_config.learning_rate
        self.running_average_generator_decay = config.models.generator.running_average_decay
        self.pose_size = config.models.pose_size
        self.discriminator_model = config.models.discriminator.structure
        self.full_validation = config.use_full_validation
        self.load_fraction_of_dataset = config.load_fraction_of_dataset

        # Image settings
        self.current_imsize = 4
        self.image_channels = 3
        self.max_imsize = config.max_imsize

        # Logging variables
        self.checkpoint_dir = config.checkpoint_dir
        self.model_name = self.checkpoint_dir.split("/")[-2]
        self.config_path = config.config_path
        self.global_step = 0
        self.logger = logger.Logger(config.summaries_dir,
                                    config.generated_data_dir)
        # Transition settings
        self.transition_variable = 1.
        self.transition_iters = config.train_config.transition_iters
        self.is_transitioning = False
        self.transition_step = 0
        self.start_channel_size = config.models.start_channel_size
        self.latest_switch = 0
        self.opt_level = config.train_config.amp_opt_level
        self.start_time = time.time()
        self.discriminator, self.generator = init_model(self.pose_size,
                                                        config.models.start_channel_size,
                                                        self.image_channels,
                                                        self.discriminator_model)
        self.init_running_average_generator()
        self.criterion = loss.WGANLoss(self.discriminator,
                                       self.generator,
                                       self.opt_level)
        if not self.load_checkpoint():
            print("Could not load checkpoint, so extending the models")
            self.extend_models()
            self.init_optimizers()

        self.batch_size = self.batch_size_schedule[self.current_imsize]
        self.update_running_average_beta()
        self.logger.log_variable("stats/batch_size", self.batch_size)

        self.num_ims_per_log = config.logging.num_ims_per_log
        self.next_log_point = self.global_step
        self.num_ims_per_save_image = config.logging.num_ims_per_save_image
        self.next_image_save_point = self.global_step
        self.num_ims_per_checkpoint = config.logging.num_ims_per_checkpoint
        self.next_validation_checkpoint = self.global_step

        self.dataloader_train, self.dataloader_val = load_dataset(
            self.dataset, self.batch_size, self.current_imsize, self.full_validation, self.pose_size, self.load_fraction_of_dataset)
        self.static_z = to_cuda(torch.randn((8, 32, 4, 4)))
        self.num_skipped_steps = 0

    def save_transition_checkpoint(self):
        dirname = os.path.dirname(self.config_path)
        filedir = os.path.join(dirname, "transition_checkpoints")
        os.makedirs(filedir, exist_ok=True)
        filepath = os.path.join(filedir, f"imsize{self.current_imsize}.ckpt")
        self.save_checkpoint(filepath)

    def update_running_average_beta(self):
        self.rae_beta = 0.5 ** (self.batch_size / (10*1000))
        self.logger.log_variable("stats/running_average_decay", self.rae_beta)

    def save_checkpoint(self, filepath=None):
        if filepath is None:
            filename = "step_{}.ckpt".format(self.global_step)
            filepath = os.path.join(self.checkpoint_dir, filename)
        state_dict = {
            "D": self.discriminator.state_dict(),
            "G": self.generator.state_dict(),
            'd_optimizer': self.d_optimizer.state_dict(),
            'g_optimizer': self.g_optimizer.state_dict(),
            "transition_step": self.transition_step,
            "is_transitioning": self.is_transitioning,
            "global_step": self.global_step,
            "total_time": self.total_time,
            "running_average_generator": self.running_average_generator.state_dict(),
            "latest_switch": self.latest_switch,
            "current_imsize": self.current_imsize,
            "transition_step": self.transition_step,
            "num_skipped_steps": self.num_skipped_steps
        }
        save_checkpoint(state_dict,
                        filepath,
                        max_keep=2 if not "validation_checkpoints" in filepath else 5)

    def load_checkpoint(self):
        try:
            map_location = "cuda:0" if torch.cuda.is_available() else "cpu"
            ckpt = load_checkpoint(self.checkpoint_dir,
                                   map_location=map_location)
            # Transition settings
            self.is_transitioning = ckpt["is_transitioning"]
            self.transition_step = ckpt["transition_step"]
            self.current_imsize = ckpt["current_imsize"]
            self.latest_switch = ckpt["latest_switch"]

            # Tracking stats
            self.global_step = ckpt["global_step"]
            self.start_time = time.time() - ckpt["total_time"] * 60
            self.num_skipped_steps = ckpt["num_skipped_steps"]

            # Models
            self.discriminator.load_state_dict(ckpt['D'])

            self.generator.load_state_dict(ckpt['G'])
            self.running_average_generator.load_state_dict(
                ckpt["running_average_generator"])
            to_cuda([self.generator, self.discriminator,
                     self.running_average_generator])
            self.running_average_generator = amp.initialize(self.running_average_generator,
                                                            None, opt_level=self.opt_level)
            self.init_optimizers()
            self.d_optimizer.load_state_dict(ckpt['d_optimizer'])
            self.g_optimizer.load_state_dict(ckpt['g_optimizer'])
            return True
        except FileNotFoundError as e:
            print(e)
            print(' [*] No checkpoint!')
            return False

    def init_running_average_generator(self):
        self.running_average_generator = Generator(self.pose_size,
                                                   self.start_channel_size,
                                                   self.image_channels)
        self.running_average_generator = wrap_models(
            self.running_average_generator)
        to_cuda(self.running_average_generator)
        self.running_average_generator = amp.initialize(self.running_average_generator,
                                                        None, opt_level=self.opt_level)

    def extend_running_average_generator(self):
        g = self.running_average_generator
        g.extend()

        for avg_param, cur_param in zip(g.new_parameters(), self.generator.new_parameters()):
            assert avg_param.data.shape == cur_param.data.shape, "AVG param: {}, cur_param: {}".format(
                avg_param.shape, cur_param.shape)
            avg_param.data = cur_param.data
        to_cuda(g)
        self.running_average_generator = amp.initialize(
            self.running_average_generator, None, opt_level=self.opt_level)

    def extend_models(self):
        self.discriminator.extend()
        self.generator.extend()
        self.extend_running_average_generator()

        self.current_imsize *= 2

        self.batch_size = self.batch_size_schedule[self.current_imsize]
        self.update_running_average_beta()
        self.transition_step += 1

    def update_running_average_generator(self):
        for avg_parameter, current_parameter in zip(
                self.running_average_generator.parameters(),
                self.generator.parameters()):

            avg_parameter.data = self.rae_beta*avg_parameter + \
                ((1-self.rae_beta) * current_parameter.float())

    def init_optimizers(self):
        self.d_optimizer = torch.optim.Adam(self.discriminator.parameters(),
                                            lr=self.learning_rate,
                                            betas=(0.0, 0.99))
        self.g_optimizer = torch.optim.Adam(self.generator.parameters(),
                                            lr=self.learning_rate,
                                            betas=(0.0, 0.99))
        self.initialize_amp()
        self.criterion.update_optimizers(self.d_optimizer, self.g_optimizer)

    def initialize_amp(self):
        to_cuda([self.generator, self.discriminator])
        [self.generator, self.discriminator], [self.g_optimizer, self.d_optimizer] = amp.initialize(
            [self.generator, self.discriminator],
            [self.g_optimizer, self.d_optimizer],
            opt_level=self.opt_level,
            num_losses=4)

    def save_transition_image(self, before):
        self.dataloader_val.update_next_transition_variable(
            self.transition_variable)
        real_image, condition, landmark = next(iter(self.dataloader_val))
        assert real_image.shape[0] >= 8
        real_data = real_image[:8]
        condition = condition[:8]
        landmark = landmark[:8]
        fake_data = self.generator(condition, landmark, self.static_z[:8])
        d_out_real = self.discriminator(real_data, condition, landmark)
        d_out_fake = self.discriminator(fake_data, condition, landmark)
        if before:
            self.d_out_real_before = d_out_real
            self.d_out_fake_before = d_out_fake
        fake_data = denormalize_img(fake_data.detach())
        real_data = denormalize_img(real_data)
        condition = denormalize_img(condition)

        to_save = torch.cat((real_data, condition, fake_data))
        tag = "before" if before else "after"
        torch.save(to_save, f".debug/{tag}.torch")
        imsize = self.current_imsize if before else self.current_imsize // 2
        imname = "transition/{}_{}_".format(tag, imsize)
        self.logger.save_images(imname, to_save, log_to_writer=False)

        if not before:
            im_before = torch.cat(
                [x for x in torch.load(".debug/before.torch")], dim=2)[None]
            im_after = torch.cat(
                [x for x in torch.load(".debug/after.torch")], dim=2)[None]
            im_before = torch.nn.functional.interpolate(
                im_before, scale_factor=2)

            diff = abs(im_after - im_before)
            diff = diff / diff.max()

            to_save = torch.cat((im_before[0], im_after[0], diff[0]), dim=1)
            self.logger.save_images(
                f"transition/to_imsize{self.current_imsize}_", to_save,
                log_to_writer=False)

            diff_real = (d_out_real - self.d_out_real_before).abs().sum()
            diff_fake = (d_out_fake - self.d_out_fake_before).abs().sum()
            self.logger.log_variable(
                "transition/discriminator_diff_real", diff_real)
            self.logger.log_variable(
                "transition/discriminator_diff_fake", diff_fake)

    def validate_model(self):
        real_scores = []
        fake_scores = []
        wasserstein_distances = []
        epsilon_penalties = []
        self.running_average_generator.eval()
        self.discriminator.eval()
        real_images = torch.zeros((len(self.dataloader_val)*self.batch_size,
                                   3,
                                   self.current_imsize,
                                   self.current_imsize))
        fake_images = torch.zeros((len(self.dataloader_val)*self.batch_size,
                                   3,
                                   self.current_imsize,
                                   self.current_imsize))
        with torch.no_grad():
            self.dataloader_val.update_next_transition_variable(
                self.transition_variable)
            for idx, (real_data, condition, landmarks) in enumerate(tqdm.tqdm(self.dataloader_val, desc="Validating model!")):
                fake_data = self.running_average_generator(condition,
                                                           landmarks)
                real_score = self.discriminator(
                    real_data, condition, landmarks)
                fake_score = self.discriminator(fake_data, condition,
                                                landmarks)
                wasserstein_distance = (real_score - fake_score).squeeze()
                epsilon_penalty = (real_score**2).squeeze()
                real_scores.append(real_score.mean().item())
                fake_scores.append(fake_score.mean().item())
                wasserstein_distances.append(
                    wasserstein_distance.mean().item())
                epsilon_penalties.append(
                    epsilon_penalty.mean().detach().item())

                start_idx = idx*self.batch_size
                end_idx = (idx+1)*self.batch_size
                real_images[start_idx:end_idx] = real_data.cpu().float()
                fake_images[start_idx:end_idx] = fake_data.cpu().float()
                del real_data, fake_data, real_score, fake_score, wasserstein_distance, epsilon_penalty
        real_images = torch_utils.image_to_numpy(real_images, to_uint8=False,
                                                 denormalize=True)
        fake_images_numpy = torch_utils.image_to_numpy(fake_images, to_uint8=False,
                                                 denormalize=True)
        fid_name = "{}_{}_{}".format(self.dataset,
                                     self.full_validation,
                                     self.current_imsize)
        if self.current_imsize >= 64:
            fid_val = fid.calculate_fid(real_images,
                                        fake_images_numpy, 
                                        False, 8, fid_name)
            self.logger.log_variable("stats/fid", np.mean(fid_val), True)
        self.logger.log_variable('discriminator/wasserstein-distance',
                                 np.mean(wasserstein_distances), True)
        self.logger.log_variable("discriminator/real-score",
                                 np.mean(real_scores), True)
        self.logger.log_variable("discriminator/fake-score",
                                 np.mean(fake_scores), True)
        self.logger.log_variable("discriminator/epsilon-penalty",
                                 np.mean(epsilon_penalties), True)
        self.logger.save_images("fakes", fake_images[:64],
                                log_to_validation=True)
        #self.discriminator.train()
        #self.generator.train()

    def log_loss_scales(self):
        self.logger.log_variable("amp/num_skipped_gradients", 
                                 self.num_skipped_steps)
        for loss_idx, loss_scaler in enumerate(amp._amp_state.loss_scalers):
            self.logger.log_variable("amp/loss_scale_{}".format(loss_idx),
                                     loss_scaler._loss_scale)

    def train_step(self, real_data, condition, landmarks):
        res = self.criterion.step(real_data, condition, landmarks)
        while res is None:
            res = self.criterion.step(real_data, condition, landmarks)
            self.num_skipped_steps += 1
            self.log_loss_scales()
        self.total_time = (time.time() - self.start_time) / 60
        wasserstein_distance, gradient_pen, real_scores, fake_scores, epsilon_penalty = res
        if self.global_step >= self.next_log_point and not amp_state_has_overflow():
            time_spent = time.time() - self.batch_start_time
            nsec_per_img = time_spent / (self.global_step - self.next_log_point + self.num_ims_per_log)

            self.logger.log_variable("stats/nsec_per_img", nsec_per_img)
            self.next_log_point = self.global_step + self.num_ims_per_log
            self.batch_start_time = time.time()
            self.log_loss_scales()
            self.logger.log_variable(
                'discriminator/wasserstein-distance',
                wasserstein_distance.item())
            self.logger.log_variable(
                'discriminator/gradient-penalty',
                gradient_pen.item())
            self.logger.log_variable("discriminator/real-score",
                                     real_scores.item())
            self.logger.log_variable("discriminator/fake-score",
                                     fake_scores.mean().item())
            self.logger.log_variable("discriminator/epsilon-penalty",
                                     epsilon_penalty.item())
            self.logger.log_variable("stats/transition-value",
                                     self.transition_variable)
            self.logger.log_variable("stats/batch_size", self.batch_size)
            self.logger.log_variable("stats/learning_rate", self.learning_rate)
            self.logger.log_variable(
                "stats/training_time_minutes", self.total_time)

    def update_transition_value(self):
        self.transition_variable = utils.compute_transition_value(
            self.global_step, self.is_transitioning, self.transition_iters,
            self.latest_switch
        )
        self.discriminator.update_transition_value(self.transition_variable)
        self.generator.update_transition_value(self.transition_variable)
        self.running_average_generator.update_transition_value(
            self.transition_variable)

    def maybe_save_validation_checkpoint(self):
        checkpoints = [100, 12*10**6, 20 * 10**6,
                       30 * 10**6, 40 * 10**6, 50 * 10**6]
        for validation_checkpoint in checkpoints:
            if self.global_step >= validation_checkpoint and (self.global_step - self.batch_size) < validation_checkpoint:
                print("Saving global checkpoint for validation")
                dirname = os.path.join("validation_checkpoints/{}".format(self.model_name))
                os.makedirs(dirname, exist_ok=True)
                fpath = os.path.join(dirname, 
                                     "step_{}.ckpt".format(self.global_step))
                self.save_checkpoint(fpath)

    def maybe_save_fake_data(self, real_data, condition, landmarks):
        if self.global_step >= self.next_image_save_point:
            self.next_image_save_point = self.global_step + self.num_ims_per_save_image
            self.generator.eval()
            with torch.no_grad():
                fake_data_sample = denormalize_img(
                    self.generator(condition, landmarks).data)
            self.logger.save_images("fakes", fake_data_sample[:64])
            # Save input images
            to_save = denormalize_img(real_data)
            self.logger.save_images("reals", to_save[:64], log_to_writer=False)
            to_save = denormalize_img(condition[:64, :3])
            self.logger.save_images("condition", to_save, log_to_writer=False)

    def maybe_validate_model(self):
        if self.global_step > self.next_validation_checkpoint:
            self.save_checkpoint()
            self.next_validation_checkpoint += self.num_ims_per_checkpoint
            self.validate_model()

    def transition_model(self):
        self.latest_switch += self.transition_iters
        print("train.py . transition_model . self.latest_switch : " + str(self.latest_switch))
        if self.is_transitioning:
            # Stop transitioning
            self.is_transitioning = False
            self.update_transition_value()
            print(f"Stopping transition. Global step: {self.global_step}, transition_variable: {self.transition_variable}, Current imsize: {self.current_imsize}")
            print("train.py . transition_model . self.is_transitioning, saving checkpoint")
            self.save_checkpoint()
        elif self.current_imsize < self.max_imsize:
            # Save image before transition
            self.save_transition_checkpoint()
            self.save_transition_image(True)
            self.extend_models()
            del self.dataloader_train, self.dataloader_val
            self.dataloader_train, self.dataloader_val = load_dataset(
                self.dataset, self.batch_size, self.current_imsize,
                self.full_validation, self.pose_size,
                self.load_fraction_of_dataset)
            self.is_transitioning = True
            print(f"Start transition. Global step: {self.global_step}, transition_variable: {self.transition_variable}, Current imsize: {self.current_imsize}")

            self.init_optimizers()
            self.update_transition_value()
            print(f"New transition value: {self.transition_variable}")

            # Save image after transition
            print("train.py . transition_model . self.current_imsize < self.max_imsize, saving transition_image")
            self.save_transition_image(False)

    def get_weights(self):
        ckpt = utils.load_checkpoint(self.checkpoint_dir, map_location="cuda:0")        
        """
        for key in list(ckpt['G'].keys()):
            if 'core_blocks_down' in key:
                key
                ckpt[key.replace('model.', '')] = ckpt[key]
                del ckpt[key]        
        """
        #self.generator.eval()
        #self.generator = torch.nn.DataParallel(self.generator, device_ids=[0])
        return self.generator.state_dict()

    def set_weights(self, ckpt):     
        #self.generator = torch.nn.DataParallel(self.generator, device_ids=[0])   
        print("In set_weights, printing state_dict of dictionary")
        for key in ckpt['parameters']:
            print(key)
        print("Done printing state_dict parameters!!!")
        self.generator.load_state_dict(ckpt)
        #self.g_optimizer.load_state_dict(weights['g_optimizer'])
        #self.generator.eval()

    def get_gradients(self):
        grads = []
        for p in self.generator.parameters():
            grad = None if p.grad is None else p.grad.data.cpu().numpy()
            grads.append(grad)
        return grads

    def set_gradients(self, gradients):
        for g, p in zip(gradients, self.generator.parameters()):
            if g is not None:
                print(type(g))
                print(type(p.grad))
                p.grad = torch.from_numpy(g).float().to("cuda:0")

@ray.remote(num_gpus=1)
class DataWorker(object):
    def __init__(self, config):
        print("DataWorker - I am called")
        self.trainer = Trainer(config)        

     def ip(self):
        return ray.services.get_node_ip_address()
    
    def train(self, weights=None):
        print("train.py - I am called")        
        self.trainer.batch_start_time = time.time()
        start_global_step = self.trainer.global_step
        end_global_step = start_global_step + self.trainer.transition_iters + 1
        print("train.py . train . Starting global step: " + str(start_global_step))
        while self.trainer.global_step < end_global_step:
            self.trainer.update_transition_value()
            self.trainer.dataloader_train.update_next_transition_variable(
                self.trainer.transition_variable)
            train_iter = iter(self.trainer.dataloader_train)
            next_transition_value = utils.compute_transition_value(
                self.trainer.global_step + self.trainer.batch_size, self.trainer.is_transitioning, self.trainer.transition_iters,
                self.trainer.latest_switch
            )
            self.trainer.dataloader_train.update_next_transition_variable(
                next_transition_value)

            #self.trainer.set_weights(weights)
            print("train.py . train . while . len(train_iter): " + str(len(train_iter)))
            for i, (real_data, condition, landmarks) in enumerate(train_iter):
                self.trainer.logger.update_global_step(self.trainer.global_step)
                if i % 4 == 0:
                    self.trainer.update_transition_value()
             
                self.trainer.train_step(real_data, condition, landmarks)

                # Log data
                self.trainer.update_running_average_generator()
                self.trainer.maybe_save_validation_checkpoint()
                self.trainer.maybe_validate_model()
                self.trainer.maybe_save_fake_data(real_data, condition, landmarks)

                self.trainer.global_step += self.trainer.batch_size
                print("train.py . train . transition_iters: " + str(self.trainer.transition_iters))
                if self.trainer.global_step >= (self.trainer.latest_switch + self.trainer.transition_iters):
                    self.trainer.transition_model()                    
                    break
                if (i + 1) % 4 == 0:
                    next_transition_value = utils.compute_transition_value(
                        self.trainer.global_step + self.trainer.batch_size, self.trainer.is_transitioning, self.trainer.transition_iters,
                        self.trainer.latest_switch
                    )
                    self.trainer.dataloader_train.update_next_transition_variable(
                        next_transition_value)
            print("End of for loop")
        print("train.py . train . Ending global step: " + str(self.trainer.global_step))
        return self.trainer.get_gradients()

@ray.remote(num_gpus=1)
class ParameterServer(object):
    def __init__(self, config):
         self.trainer = Trainer(config)
         self.g_optimizer = self.trainer.g_optimizer#torch.optim.Adam(self.trainer.generator.parameters(), lr=config.train_config.learning_rate, betas=(0.0, 0.99))

    def apply_gradients(self, *gradients):
        aggregated_generator_gradients = [
          np.stack(gradients_zip).mean(axis=0)
          for gradients_zip in zip(*gradients)
        ]
        self.g_optimizer.zero_grad()
        self.trainer.set_gradients(aggregated_generator_gradients)
        self.g_optimizer.step()
        print("*******************AGGREGATED_GRADIENTS*****************************")
        print(aggregated_generator_gradients)
        return self.trainer.get_weights()

    def get_weights(self):
        return self.trainer.get_weights()

     def ip(self):
        return ray.services.get_node_ip_address()

    def save_checkpoint(aggregated_generator_parameters):
        filename = "step_{}.ckpt".format(self.global_step)
        filepath = os.path.join(self.checkpoint_dir, filename)
        state_dict = {
            "D": self.trainer.discriminator.state_dict(),
            "G": self.trainer.generator.state_dict(),
            'd_optimizer': self.trainer.d_optimizer.state_dict(),
            'g_optimizer': self.trainer.g_optimizer.state_dict(),
            "transition_step": self.trainer.transition_step,
            "is_transitioning": self.trainer.is_transitioning,
            "global_step": self.trainer.global_step,
            "total_time": self.trainer.total_time,
            "running_average_generator": self.trainer.running_average_generator.state_dict(),
            "latest_switch": self.trainer.latest_switch,
            "current_imsize": self.trainer.current_imsize,
            "transition_step": self.trainer.transition_step,
            "num_skipped_steps": self.trainer.num_skipped_steps
        }
        save_checkpoint(state_dict,
                        filepath,
                        max_keep=2 if not "validation_checkpoints" in filepath else 5)


    
if __name__ == '__main__':
    num_workers = 2
    config = config_parser.initialize_and_validate_config()
    if args.redis_address is None:
        # Run everything locally.
        ray.init(num_gpus=args.num_parameter_servers +  2*args.num_workers)
    else:
        # Connect to a cluster.
        ray.init(redis_address=args.redis_address)
   # ray.init(ignore_reinit_error=True)  
    ps = ParameterServer.remote(config)
    workers = [DataWorker.remote(config) for i in range(num_workers)]
    
    # As a sanity check, make sure all workers and parameter servers are on
    # different machines.
    if args.redis_address is not None:
        all_ips = ray.get([ps.ip.remote() for ps in pss] +
                          [w.ip.remote() for w in workers])
        assert len(all_ips) == len(set(all_ips))

    current_weights = ps.get_weights.remote()
    gradients = {}
    for worker in workers:    
    	current_gradients = worker.train.remote(current_weights)
    	print("*********************************************************************")
    	print("*********************DONE TRAINING************************")
    	gradients[current_gradients] = worker
    	print("*********************************************************************")
    	print("*********************GRADIENTS************************")
    	print(gradients)
    	print("*********************************************************************")
    	print("********************RETRIEVING GRADIENTS***********************************")
    for i in range(num_workers):
    	ready_gradient_list, _ = ray.wait(list(gradients))
    	ready_gradient_id = ready_gradient_list[0]
    	worker = gradients.pop(ready_gradient_id)
    	print(ready_gradient_id)

    	# Compute and apply gradients.
    	current_weights = ps.apply_gradients.remote(*[ready_gradient_id])
    	print("APPLIED GRADIENTS ! ! ! ")
    	print("***********************RE-TRAINING WITH NEW GRADIENTS********************")
    	gradients[worker.train.remote(current_weights)] = worker

    #model.set_weights(ray.get(current_weights))
    #print("Final accuracy is {:.1f}.".format(accuracy))

    """
    ray.init(ignore_reinit_error=True)    
    ps = ParameterServer.remote(config, "cuda:0")  
    model = Trainer(config)
    trainers = [Trainer.remote(config) for i in range(num_workers)]
    print("Running asynchronous paramater server training using Ray")  
    current_weights = ps.get_weights.remote()
    gradients = [trainer.train.remote() for trainer in trainers]
    model.get_gradients()
    """
