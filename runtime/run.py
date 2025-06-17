# Copyright (c) 2022, NVIDIA CORPORATION. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import time

import horovod.tensorflow as hvd
import numpy as np
import tensorflow as tf
import tensorflow_addons as tfa
from tensorflow.python.compiler.tensorrt import trt_convert as trt

from runtime.checkpoint import CheckpointManager
from runtime.losses import DiceCELoss, WeightDecay, Joint_Dice_Weighted_BinaryCE_Loss, Joint_Dice_BinaryCE_Loss, Joint_Dice_Boundary_Weighted_CE_Loss, Joint_Dice_Focal_CE_Loss, Joint_Dice_Focal_Boundary_CE_Loss
from runtime.metrics import Dice, MetricAggregator, make_class_logger_metrics, Hard_Dice_Coef_Metric
from runtime.utils import is_main_process, make_empty_dir, progress_bar
from runtime.optimizers import SGDW
from runtime.schedulers import PolynomialDecayWithWarmup


def update_best_metrics(old, new, start_time, iteration, watch_metric=None):
    did_change = False
    for metric, value in new.items():
        if metric not in old or old[metric]["value"] < value:
            old[metric] = {"value": value, "timestamp": time.time() - start_time, "iter": int(iteration)}
            if watch_metric == metric:
                did_change = True
    return did_change


def get_scheduler(args, total_steps):
    scheduler = {
        "poly_warmup": PolynomialDecayWithWarmup(
            initial_learning_rate=args.learning_rate,
            factor_decrease_learning_rate=args.factor_decrease_learning_rate,
            total_steps=total_steps,
            warmup_iterations=args.warmup_iterations,
            power=args.power,
        ),
        "poly": tf.keras.optimizers.schedules.PolynomialDecay(
            initial_learning_rate=args.learning_rate,
            end_learning_rate=args.end_learning_rate,
            decay_steps=total_steps,
            power=args.power,
        ),
        "cosine": tf.keras.optimizers.schedules.CosineDecay(
            initial_learning_rate=args.learning_rate, decay_steps=total_steps
        ),
        "cosine_annealing": tf.keras.optimizers.schedules.CosineDecayRestarts(
            initial_learning_rate=args.learning_rate,
            first_decay_steps=args.cosine_annealing_first_cycle_steps,
            alpha=0.01,
            t_mul=2.0,
            m_mul=args.cosine_annealing_peak_decay
        ),
        "none": args.learning_rate,
    }[args.scheduler.lower()]
    return scheduler


def get_optimizer(args, scheduler):
    #choose optimizer: all optimizers have global gradient norm clipping to 12 (as per nnUnet)
    optimizer = {
        "sgdw": SGDW(learning_rate=scheduler, weight_decay=args.weight_decay, momentum=args.momentum, dampening=args.dampening, nesterov=args.nesterov, decoupled=args.decoupled, global_clipnorm=12.0, use_ema=args.use_ema, ema_momentum=args.ema_momentum, ema_overwrite_frequency=args.ema_overwrite_frequency),
        "sgd": tf.keras.optimizers.experimental.SGD(learning_rate=scheduler, momentum=args.momentum, nesterov=args.nesterov, global_clipnorm=12.0),
        "adam": tf.keras.optimizers.Adam(learning_rate=scheduler, global_clipnorm=12.0),
        "adamw": tf.keras.optimizers.experimental.AdamW(learning_rate=scheduler, weight_decay=args.weight_decay, beta_1=0.9, beta_2=0.999, epsilon=1e-08, amsgrad=False, global_clipnorm=12.0),
        "radam": tfa.optimizers.RectifiedAdam(learning_rate=scheduler, global_clipnorm=12.0),
    }[args.optimizer.lower()]
    if args.lookahead:
        optimizer = tfa.optimizers.Lookahead(optimizer)
    #Decide whether to exclude certain variables from weight decay
    if args.exclude_from_weight_decay == True:
        if (args.optimizer.lower() == 'sgdw') or (args.optimizer.lower() == 'adamw'):
            optimizer.exclude_from_weight_decay(var_names=['normalization','bias'])
    if args.amp:
        optimizer = tf.keras.mixed_precision.LossScaleOptimizer(optimizer, dynamic=True)
    return optimizer


def get_epoch_size(args, batch_size, dataset_size):
    if args.steps_per_epoch:
        return args.steps_per_epoch
    div = args.gpus * (batch_size if args.dim == 3 else args.nvol)
    return (dataset_size + div - 1) // div


def process_performance_stats(deltas, batch_size, mode):
    deltas_ms = 1000 * np.array(deltas)
    throughput_imgps = 1000.0 * batch_size / deltas_ms.mean()
    stats = {f"throughput_{mode}": throughput_imgps, f"latency_{mode}_mean": deltas_ms.mean()}
    for level in [90, 95, 99]:
        stats.update({f"latency_{mode}_{level}": np.percentile(deltas_ms, level)})

    return stats


def benchmark(args, step_fn, data, steps, warmup_steps, logger, mode="train"):
    assert steps > warmup_steps, "Number of benchmarked steps has to be greater then number of warmup steps"
    deltas = []
    wrapped_data = progress_bar(
        enumerate(data),
        quiet=args.quiet,
        desc=f"Benchmark ({mode})",
        unit="step",
        postfix={"phase": "warmup"},
        total=steps,
    )
    start = time.perf_counter()
    for step, (images, labels) in wrapped_data:
        _ = step_fn(images, labels, warmup_batch=step == 0)
        if step >= warmup_steps:
            deltas.append(time.perf_counter() - start)
            if step == warmup_steps and is_main_process() and not args.quiet:
                wrapped_data.set_postfix(phase="benchmark")
        start = time.perf_counter()
        if step >= steps:
            break

    stats = process_performance_stats(deltas, args.gpus * args.batch_size, mode=mode)
    logger.log_metrics(stats)


def train(args, model, dataset, logger):
    #
    train_data = dataset.train_dataset()
    #
    epochs = args.epochs
    batch_size = args.batch_size if args.dim == 3 else args.nvol
    steps_per_epoch = get_epoch_size(args, batch_size, dataset.train_size())
    total_steps = epochs * steps_per_epoch
    #
    scheduler = get_scheduler(args, total_steps)
    optimizer = get_optimizer(args, scheduler)
    #loss_fn = Joint_Dice_BinaryCE_Loss(reduce_batch=args.reduce_batch, include_background=args.include_background, batch_size=batch_size)
    loss_fn = Joint_Dice_Boundary_Weighted_CE_Loss(reduce_batch=args.reduce_batch, include_background=args.include_background, batch_size=batch_size)
    #loss_fn = Joint_Dice_Focal_CE_Loss(reduce_batch=args.reduce_batch, include_background=args.include_background, batch_size=batch_size)
    #loss_fn = Joint_Dice_Focal_Boundary_CE_Loss(reduce_batch=args.reduce_batch, include_background=args.include_background, batch_size=batch_size)
    wdecay = WeightDecay(factor=args.weight_decay)
    tstep = tf.Variable(0)
    
    @tf.function
    def train_step_fn(features, labels, dist_maps, warmup_batch=False):
        features, labels = model.adjust_batch(features, labels)
        with tf.GradientTape() as tape:
            #forward pass through network
            output_map = model(features)
            #compute loss
            loss = model.compute_loss(loss_fn, labels, dist_maps, output_map)
            #scale loss if using AMP
            if args.amp:
                loss = optimizer.get_scaled_loss(loss)
                #add in L2 weight decay if using standard adam or sgd
                if (args.optimizer.lower() == 'adam') or (args.optimizer.lower() == 'sgd'):
                    loss = loss + wdecay(model)
        #handles trainining across multiple GPUs (perhaps?)
        if args.use_hvd == True:
            tape = hvd.DistributedGradientTape(tape)
        #get gradients
        gradients = tape.gradient(loss, model.trainable_variables)
        if args.amp:
            #unscale gradients if using AMP
            gradients = optimizer.get_unscaled_gradients(gradients)
        ##gradient clipping to the same value of 12 that nnUnet uses
        #gradients, _ = tf.clip_by_global_norm(gradients, 12)
        #apply gradients to all trainable variables
        optimizer.apply_gradients(zip(gradients, model.trainable_variables))
        # Note: broadcast should be done after the first gradient step to ensure optimizer initialization.
        if warmup_batch:
            if args.use_hvd == True:
                hvd.broadcast_variables(model.variables, root_rank=0)
                #if using experimental optimizer, need to grab wrapped optimizers variables as such
                hvd.broadcast_variables(optimizer._optimizer.variables(), root_rank=0)
        return labels, output_map
    
    #dice metrics and aggregators for training/validation (validation metric uses batch size one since it is computed on full size images)
    dice_train = Hard_Dice_Coef_Metric(reduce_batch=args.reduce_batch, batch_size=batch_size, use_hvd=args.use_hvd)
    dice = Hard_Dice_Coef_Metric(reduce_batch=args.reduce_batch, batch_size=1, use_hvd=args.use_hvd)
    dice_metrics_train = MetricAggregator(name="dice_train")
    dice_metrics = MetricAggregator(name="dice_val")
    #checkpoint manager
    checkpoint = CheckpointManager(
        args.ckpt_dir,
        strategy=args.ckpt_strategy,
        resume_training=args.resume_training,
        resume_training_best=args.resume_training_best,
        variables={"model": model, "optimizer": optimizer, "step": tstep, **dice_metrics.checkpoint_metrics()},
    )
    if args.benchmark:
        benchmark(args, train_step_fn, train_data, args.bench_steps, args.warmup_steps, logger)
    else:
        epoch_counter = 0
        wrapped_data = progress_bar(
            train_data,
            quiet=args.quiet,
            desc="Train",
            postfix={"epoch": epoch_counter},
            unit="step",
            total=total_steps - int(tstep),
        )
        start_time = time.time()
        for images, labels, dist_maps in wrapped_data:
            #break training loop if trained for max number of steps
            if tstep >= total_steps:
                break
            #compute training step with current batch
            labels, output_map = train_step_fn(images, labels, dist_maps, warmup_batch=tstep == 0)
            #update training dice score
            if epoch_counter >= args.skip_train_eval:
                dice_train.update_state(output_map[0], labels)
            #check if end of training epoch
            if (tstep % steps_per_epoch == 0) and (tstep != 0):
                #log learning rate of current step
                lr = scheduler(tstep) if callable(scheduler) else scheduler
                metrics = {"learning_rate": float(lr)}
                #average training dice score over epoch
                if epoch_counter >= args.skip_train_eval:
                    hard_dice_train = dice_train.result()
                    hard_dice_train = tf.reduce_mean(hard_dice_train[1:])
                    _ = dice_metrics_train.update(hard_dice_train)
                    #add training dice score to metrics dictionary
                    metrics.update(dice_metrics_train.logger_metrics())
                    #reset training dice metrics for next epoch
                    dice_train.reset_state()
                #check if need to run model evaluation
                if epoch_counter >= args.skip_eval:
                    #run full sliding window inference to get accurate validation dice score
                    dice, hard_dice_val = evaluate(args, model, dataset, logger, dice)
                    #check if validation dice score has improved
                    hard_dice_val = tf.reduce_mean(hard_dice_val[1:])
                    did_improve = dice_metrics.update(hard_dice_val)
                    metrics.update(dice_metrics.logger_metrics())
                    if did_improve == True:
                        metrics["time_to_train"] = time.time() - start_time
                    #save new checkpoint if new best validation metric
                    checkpoint.update_best(float(hard_dice_val), epoch_counter)
                    #reset validation dice metrics for next epoch
                    dice.reset_state()
                #print out to screen and log file
                logger.log_metrics(metrics=metrics, step=int(tstep))
                logger.flush()
                #increment epoch counter
                epoch_counter = epoch_counter + 1
                if is_main_process() and not args.quiet:
                    wrapped_data.set_postfix(epoch = epoch_counter)
            #increment step counter
            tstep.assign_add(1)
        #finalize model weights if using ema
        if args.use_ema == True:
            optimizer._optimizer.finalize_variable_values(model.trainable_variables)
        #save last epoch if requested to do so
        checkpoint.update_last(epoch_counter)
        #final metrics to print out
        metrics = {
            "train_dice": round(float(dice_metrics_train.metrics["max"].result()), 5),
            "val_dice": round(float(dice_metrics.metrics["max"].result()), 5),
        }
        logger.log_metrics(metrics=metrics)
        logger.flush()


def evaluate(args, model, dataset, logger, dice=None):
    #if validation dice metric is none, instantiate metric
    if dice == None:
        dice = Hard_Dice_Coef_Metric(reduce_batch=args.reduce_batch, batch_size=1)
    #create validation dataset progress bar
    data_size = dataset.val_size()
    wrapped_data = progress_bar(
        enumerate(dataset.val_dataset()),
        quiet=args.quiet,
        desc="Validation",
        unit="step",
        total=data_size,
    )
    #iterate over validation dataset
    for i, (features, labels, dist_maps) in wrapped_data:
        if args.dim == 2:
            features, labels = features[0], labels[0]
        output_map = model.inference(features)

        dice.update_state(output_map, labels)
        if i + 1 == data_size:
            break
    #compute final dice score
    result = dice.result()
    if args.exec_mode == "evaluate":
        metrics = {
            "eval_dice": float(tf.reduce_mean(result)),
            "eval_dice_nobg": float(tf.reduce_mean(result[1:])),
        }
        logger.log_metrics(metrics)
        logger.flush()
    return dice, result


def predict(args, model, dataset, logger):
    if args.benchmark:

        @tf.function
        def predict_bench_fn(features, labels, warmup_batch):
            if args.dim == 2:
                features = features[0]
            output_map = model(features, training=False)
            return output_map

        benchmark(
            args,
            predict_bench_fn,
            dataset.test_dataset(),
            args.bench_steps,
            args.warmup_steps,
            logger,
            mode="predict",
        )
    else:
        if args.save_preds:
            dir_name = f"preds_fold_{args.fold}"
            if args.tta:
                dir_name += "_tta"
            save_dir = args.results / dir_name
            make_empty_dir(save_dir)

        data_size = dataset.test_size()
        wrapped_data = progress_bar(
            enumerate(dataset.test_dataset()),
            quiet=args.quiet,
            desc="Predict",
            unit="step",
            total=data_size,
        )

        for i, (images,) in wrapped_data:
            features, _ = model.adjust_batch(images, None)
            pred = model.inference(features, training=False)
            if args.save_preds:
                model.save_pred(pred, idx=i, data_module=dataset, save_dir=save_dir)
            if i + 1 == data_size:
                break


def export_model(args, model):
    checkpoint = tf.train.Checkpoint(model=model)
    checkpoint.restore(tf.train.latest_checkpoint(args.ckpt_dir)).expect_partial()

    input_shape = [1, *model.patch_size, model.n_class]
    dummy_input = tf.constant(tf.zeros(input_shape, dtype=tf.float32))
    _ = model(dummy_input, training=False)

    prec = "amp" if args.amp else "fp32"
    path = str(args.results / f"saved_model_task_{args.task}_dim_{args.dim}_{prec}")
    tf.keras.models.save_model(model, str(path))

    trt_prec = trt.TrtPrecisionMode.FP32 if prec == "fp32" else trt.TrtPrecisionMode.FP16
    converter = trt.TrtGraphConverterV2(
        input_saved_model_dir=path,
        conversion_params=trt.TrtConversionParams(precision_mode=trt_prec),
    )
    converter.convert()

    trt_path = str(args.results / f"trt_saved_model_task_{args.task}_dim_{args.dim}_{prec}")
    converter.save(trt_path)
