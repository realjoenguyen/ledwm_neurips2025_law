sh scripts/run_messenger_s1.sh 1,2,3 --jax.train_devices 0,1,2 --jax.policy_devices 0  --run.server opt --configs reward_grain large_encoder_token large_decoder_token dense_image_token no_lang_token image_loss_only random_policy --envs.amount 50 --batch_size 120
echo 'FAIL'
sh scripts/run_messenger_s1.sh 1,2,3 --jax.train_devices 0,1,2 --jax.policy_devices 0  --run.server opt --configs reward_grain large_encoder_token large_decoder_token dense_image_token no_lang_token image_loss_only random_policy --envs.amount 50 --batch_size 111
echo "FAIL"
sh scripts/run_messenger_s1.sh 1,2,3 --jax.train_devices 0,1,2 --jax.policy_devices 0  --run.server opt --configs reward_grain large_encoder_token large_decoder_token dense_image_token no_lang_token image_loss_only random_policy --envs.amount 50 --batch_size 105

