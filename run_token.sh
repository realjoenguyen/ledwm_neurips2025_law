sh scripts/run_messenger_s1.sh 0,1,2,3 --jax.train_devices 0,1,2,3 --jax.policy_devices 0  --run.server dgx --configs reward_grain large_encoder_token large_decoder_token dense_image_token --envs.amount 50 --batch_size 108
sh scripts/run_messenger_s1.sh 0,1,2,3 --jax.train_devices 0,1,2,3 --jax.policy_devices 0  --run.server dgx --configs reward_grain large_encoder_token large_decoder_token dense_image_token --envs.amount 50 --batch_size 100

