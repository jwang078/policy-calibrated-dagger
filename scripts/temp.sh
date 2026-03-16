lerobot-eval \
            --env.type=splatsim \
            --env.task=upright_small_engine_new \
            --env.camera_names='["base_rgb"]' \
            --env.image_resize_modes='["stretch"]' \
            --env.fps=30 \
            --policy.path=/home/jennyw2/code/lerobot/outputs/training/diffusion_approach_lever_test_nobox_base/checkpoints/050000/pretrained_model \
            --eval.n_episodes=5 \
            --output_dir=/home/jennyw2/code/lerobot/outputs/eval_output/2026-03-05-085018/diffusion_approach_lever_test_nobox_base_050000 \
            --eval.batch_size=1 \
            --eval.use_async_envs=false \
            --rename_map='{"observation.images.base_rgb_stretch": "observation.images.base_rgb"}'


sleep 10

lerobot-eval \
            --env.type=splatsim \
            --env.task=upright_small_engine_new \
            --env.camera_names='["base_rgb"]' \
            --env.image_resize_modes='["stretch"]' \
            --env.fps=30 \
            --policy.path=/home/jennyw2/code/lerobot/outputs/training/diffusion_approach_lever_test_nobox_lowres_base/checkpoints/050000/pretrained_model \
            --eval.n_episodes=5 \
            --output_dir=/home/jennyw2/code/lerobot/outputs/eval_output/2026-03-05-085307/diffusion_approach_lever_test_nobox_lowres_base_050000 \
            --eval.batch_size=1 \
            --eval.use_async_envs=false \
            --rename_map='{"observation.images.base_rgb_stretch": "observation.images.base_rgb"}'

sleep 10

lerobot-eval \
            --env.type=splatsim \
            --env.task=upright_small_engine_new \
            --env.camera_names='["base_rgb"]' \
            --env.image_resize_modes='["stretch"]' \
            --env.fps=30 \
            --policy.path=/home/jennyw2/code/lerobot/outputs/training/diffusion_approach_lever_test_nobox_lowres_jitter_base/checkpoints/050000/pretrained_model \
            --eval.n_episodes=5 \
            --output_dir=/home/jennyw2/code/lerobot/outputs/eval_output/2026-03-05-085307/diffusion_approach_lever_test_nobox_lowres_jitter_base_050000 \
            --eval.batch_size=1 \
            --eval.use_async_envs=false \
            --rename_map='{"observation.images.base_rgb_stretch": "observation.images.base_rgb"}'

# python /home/jennyw2/code/lerobot/src/lerobot/scripts/lerobot_train.py --dataset.repo_id=JennyWWW/splatsim_approach_lever_4_5path  --policy.type=diffusion --output_dir=./outputs/training/diffusion_approach_lever_4_5path_fixnorm_basewrist --job_name=diffusion_approach_lever_4_5path_fixnorm_basewrist --policy.repo_id=diffusion_approach_lever_4_5path_fixnorm_basewrist --eval_freq=25000 --save_freq=25000 --wandb.enable=true --steps=75000 --policy.device=cuda --batch_size=32 --policy.vision_backbone=resnet18 --policy.pretrained_backbone_weights=null --policy.use_group_norm=true "--policy.crop_shape=[224, 224]" --policy.crop_is_random=false --env.type=splatsim --env.task=upright_small_engine_new --env.camera_names='["base_rgb", "wrist_rgb"]' --env.fps=30 --eval.n_episodes=5 --eval.batch_size=1 --eval.use_async_envs=false --policy.optimizer_lr=1e-5 --env.image_resize_modes='["stretch"]' --policy.use_separate_rgb_encoder_per_camera=true --policy.input_features='{"observation.images.base_rgb": {"type": "VISUAL", "shape": [3, 224, 224]}, "observation.images.wrist_rgb": {"type": "VISUAL", "shape": [3, 224, 224]}, "observation.state": {"type": "STATE", "shape": [7]}}' --rename_map='{"observation.images.base_rgb_stretch": "observation.images.base_rgb", "observation.images.wrist_rgb_stretch": "observation.images.wrist_rgb"}'

# sleep 10

# python /home/jennyw2/code/lerobot/src/lerobot/scripts/lerobot_train.py --dataset.repo_id=JennyWWW/splatsim_approach_lever_4_5path  --policy.type=diffusion --output_dir=./outputs/training/diffusion_approach_lever_4_5path_fixnorm_base --job_name=diffusion_approach_lever_4_5path_fixnorm_base --policy.repo_id=diffusion_approach_lever_4_5path_fixnorm_base --eval_freq=25000 --save_freq=25000 --wandb.enable=true --steps=75000 --policy.device=cuda --batch_size=32 --policy.vision_backbone=resnet18 --policy.pretrained_backbone_weights=null --policy.use_group_norm=true "--policy.crop_shape=[224, 224]" --policy.crop_is_random=false --env.type=splatsim --env.task=upright_small_engine_new --env.camera_names='["base_rgb"]' --env.fps=30 --eval.n_episodes=5 --eval.batch_size=1 --eval.use_async_envs=false --policy.optimizer_lr=1e-5 --env.image_resize_modes='["stretch"]' --policy.use_separate_rgb_encoder_per_camera=true --policy.input_features='{"observation.images.base_rgb": {"type": "VISUAL", "shape": [3, 224, 224]}, "observation.state": {"type": "STATE", "shape": [7]}}' --rename_map='{"observation.images.base_rgb_stretch": "observation.images.base_rgb"}'

# sleep 10

# python /home/jennyw2/code/lerobot/src/lerobot/scripts/lerobot_train.py --dataset.repo_id=JennyWWW/splatsim_approach_lever_4_5path  --policy.type=diffusion --output_dir=./outputs/training/diffusion_approach_lever_4_5path_fixnorm_wrist --job_name=diffusion_approach_lever_4_5path_fixnorm_wrist --policy.repo_id=diffusion_approach_lever_4_5path_fixnorm_wrist --eval_freq=25000 --save_freq=25000 --wandb.enable=true --steps=75000 --policy.device=cuda --batch_size=32 --policy.vision_backbone=resnet18 --policy.pretrained_backbone_weights=null --policy.use_group_norm=true "--policy.crop_shape=[224, 224]" --policy.crop_is_random=false --env.type=splatsim --env.task=upright_small_engine_new --env.camera_names='["wrist_rgb"]' --env.fps=30 --eval.n_episodes=5 --eval.batch_size=1 --eval.use_async_envs=false --policy.optimizer_lr=1e-5 --env.image_resize_modes='["stretch"]' --policy.use_separate_rgb_encoder_per_camera=true --policy.input_features='{"observation.images.wrist_rgb": {"type": "VISUAL", "shape": [3, 224, 224]}, "observation.state": {"type": "STATE", "shape": [7]}}' --rename_map='{"observation.images.wrist_rgb_stretch": "observation.images.wrist_rgb"}'

# sleep 10

# PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True python /home/jennyw2/code/lerobot/src/lerobot/scripts/lerobot_train.py --dataset.repo_id=JennyWWW/splatsim_approach_lever_4_5path --policy.type=pi05 --output_dir=./outputs/training/pi05_approach_lever_4_5path_fixnorm_basewrist --job_name=pi05_approach_lever_4_5path_fixnorm_basewrist --policy.repo_id=pi05_approach_lever_4_5path_fixnorm_basewrist --policy.pretrained_path=lerobot/pi05_base --policy.compile_model=false --policy.gradient_checkpointing=true --wandb.enable=true --policy.dtype=bfloat16 --steps=3000 --policy.scheduler_decay_steps=3000 --policy.device=cuda --batch_size=16 --policy.train_expert_only=true --policy.use_amp=true --save_freq=1000 --env.type=splatsim --env.task=upright_small_engine_new --env.camera_names='["base_rgb", "wrist_rgb"]' --env.fps=30 --eval.n_episodes=5 --eval.batch_size=1 --eval.use_async_envs=false --eval_freq=1000 --env.image_resize_modes='["letterbox"]' --policy.input_features='{"observation.images.base_rgb": {"type": "VISUAL", "shape": [3, 224, 224]}, "observation.images.wrist_rgb": {"type": "VISUAL", "shape": [3, 224, 224]}, "observation.state": {"type": "STATE", "shape": [7]}}' --rename_map='{"observation.images.base_rgb_letterbox": "observation.images.base_rgb", "observation.images.wrist_rgb_letterbox": "observation.images.wrist_rgb"}'

# sleep 10

# python /home/jennyw2/code/lerobot/src/lerobot/scripts/lerobot_train.py --dataset.repo_id=JennyWWW/splatsim_approach_lever_4_5path --policy.type=pi05 --output_dir=./outputs/training/pi05_approach_lever_4_5path_fixnorm_base --job_name=pi05_approach_lever_4_5path_fixnorm_base --policy.repo_id=pi05_approach_lever_4_5path_fixnorm_base --policy.pretrained_path=lerobot/pi05_base --policy.compile_model=false --policy.gradient_checkpointing=true --wandb.enable=true --policy.dtype=bfloat16 --steps=3000 --policy.scheduler_decay_steps=3000 --policy.device=cuda --batch_size=16 --policy.train_expert_only=true --policy.use_amp=true --save_freq=1000 --env.type=splatsim --env.task=upright_small_engine_new --env.camera_names='["base_rgb"]' --env.fps=30 --eval.n_episodes=5 --eval.batch_size=1 --eval.use_async_envs=false --eval_freq=1000 --env.image_resize_modes='["letterbox"]' --policy.input_features='{"observation.images.base_rgb": {"type": "VISUAL", "shape": [3, 224, 224]}, "observation.state": {"type": "STATE", "shape": [7]}}' --rename_map='{"observation.images.base_rgb_letterbox": "observation.images.base_rgb"}'

# sleep 10

# python /home/jennyw2/code/lerobot/src/lerobot/scripts/lerobot_train.py --dataset.repo_id=JennyWWW/splatsim_approach_lever_4_5path --policy.type=pi05 --output_dir=./outputs/training/pi05_approach_lever_4_5path_fixnorm_wrist --job_name=pi05_approach_lever_4_5path_fixnorm_wrist --policy.repo_id=pi05_approach_lever_4_5path_fixnorm_wrist --policy.pretrained_path=lerobot/pi05_base --policy.compile_model=false --policy.gradient_checkpointing=true --wandb.enable=true --policy.dtype=bfloat16 --steps=3000 --policy.scheduler_decay_steps=3000 --policy.device=cuda --batch_size=16 --policy.train_expert_only=true --policy.use_amp=true --save_freq=1000 --env.type=splatsim --env.task=upright_small_engine_new --env.camera_names='["wrist_rgb"]' --env.fps=30 --eval.n_episodes=5 --eval.batch_size=1 --eval.use_async_envs=false --eval_freq=1000 --env.image_resize_modes='["letterbox"]' --policy.input_features='{"observation.images.wrist_rgb": {"type": "VISUAL", "shape": [3, 224, 224]}, "observation.state": {"type": "STATE", "shape": [7]}}' --rename_map='{"observation.images.wrist_rgb_letterbox": "observation.images.wrist_rgb"}'
