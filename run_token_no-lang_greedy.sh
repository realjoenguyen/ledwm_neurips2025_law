
sh scripts/run_messenger_s1.sh 0,1,2,3 --jax.train_devices 0,1,2,3 --jax.policy_devices 0  --run.server dgx --configs reward_grain large_encoder_token large_decoder_token dense_image_token no_lang_token image_loss_only  --envs.amount 50 --batch_size 160
echo 'FAIL'
sh scripts/run_messenger_s1.sh 0,1,2,3 --jax.train_devices 0,1,2,3 --jax.policy_devices 0  --run.server dgx --configs reward_grain large_encoder_token large_decoder_token dense_image_token no_lang_token image_loss_only  --envs.amount 50 --batch_size 140
echo 'FAIL'
sh scripts/run_messenger_s1.sh 0,1,2,3 --jax.train_devices 0,1,2,3 --jax.policy_devices 0  --run.server dgx --configs reward_grain large_encoder_token large_decoder_token dense_image_token no_lang_token image_loss_only  --envs.amount 50 --batch_size 120
echo 'FAIL'
sh scripts/run_messenger_s1.sh 0,1,2,3 --jax.train_devices 0,1,2,3 --jax.policy_devices 0  --run.server dgx --configs reward_grain large_encoder_token large_decoder_token dense_image_token no_lang_token image_loss_only  --envs.amount 50 --batch_size 108
echo 'FAIL'
sh scripts/run_messenger_s1.sh 0,1,2,3 --jax.train_devices 0,1,2,3 --jax.policy_devices 0  --run.server dgx --configs reward_grain large_encoder_token large_decoder_token dense_image_token no_lang_token image_loss_only  --envs.amount 50 --batch_size 100

