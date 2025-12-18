import os
import modal

####### initialize environment hyperparameters ######
env_name = "MountainCar-v0"
has_continuous_action_space = False  # continuous action space; else discrete

max_ep_len = 1000                   # max timesteps in one episode
max_training_timesteps = int(3e6)   # break training loop if timeteps > max_training_timesteps

save_model_freq = int(1e5)          # save model frequency (in num timesteps)

action_std = 0.6                    # starting std for action distribution (Multivariate Normal)
action_std_decay_rate = 0.05        # linearly decay action_std (action_std = action_std - action_std_decay_rate)
min_action_std = 0.1                # minimum action_std (stop decay after action_std <= min_action_std)
action_std_decay_freq = int(2.5e5)  # action_std decay frequency (in num timesteps)
#####################################################

## Note : print/log frequencies should be > than max_ep_len

################ PPO hyperparameters ################
update_timestep = max_ep_len * 4      # update policy every n timesteps
K_epochs = 80               # update policy for K epochs in one PPO update

eps_clip = 0.2          # clip parameter for PPO
gamma = 0.99            # discount factor

lr_actor = 0.00045       # learning rate for actor network
lr_critic = 0.0015       # learning rate for critic network

random_seed = 42         # set random seed if required (0 = no random seed)
#####################################################

################### Modal configuration ###################
run_num_pretrained = 1      #### change this to prevent overwriting weights in same env_name folder
volumne_name = 'PPO_preTrained'
volume_mount_path = "/root/models"
#####################################################

image = modal.Image.debian_slim(python_version="3.10").pip_install(
                                                            "torch",
                                                            "numpy",
                                                            "gymnasium",
                                                            "clickhouse-connect",
                                                            # "langfuse"
                                                        ).add_local_dir(
                                                            os.getcwd(), 
                                                            "/root/workspace", 
                                                            copy=True
                                                        )

app = modal.App("ppo-training",
                image=image
                )

volume = modal.Volume.from_name(volumne_name, create_if_missing=True)

@app.function(
    gpu=None,
    timeout=86400,  # 24 hours timeout
    secrets=[modal.Secret.from_dotenv()],
    volumes={volume_mount_path: volume},
)
def train():
    """Execute the training loop. This runs inside Modal."""
    import os
    import gymnasium as gym
    from gymnasium.spaces.utils import flatdim
    import torch
    import numpy as np
    from datetime import datetime

    # from langfuse import get_client, propagate_attributes

    from workspace.PPO import PPO
    from workspace.clickhouse_logger import ClickHouseLogger

    print("training environment name : " + env_name)

    ############# print all hyperparameters #############
    print("--------------------------------------------------------------------------------------------")
    print("max training timesteps : ", max_training_timesteps)
    print("max timesteps per episode : ", max_ep_len)
    print("model saving frequency : " + str(save_model_freq) + " timesteps")
    print("--------------------------------------------------------------------------------------------")
    if has_continuous_action_space:
        print("Initializing a continuous action space policy")
        print("--------------------------------------------------------------------------------------------")
        print("starting std of action distribution : ", action_std)
        print("decay rate of std of action distribution : ", action_std_decay_rate)
        print("minimum std of action distribution : ", min_action_std)
        print("decay frequency of std of action distribution : " + str(action_std_decay_freq) + " timesteps")
    else:
        print("Initializing a discrete action space policy")
    print("--------------------------------------------------------------------------------------------")
    print("PPO update frequency : " + str(update_timestep) + " timesteps")
    print("PPO K epochs : ", K_epochs)
    print("PPO epsilon clip : ", eps_clip)
    print("discount factor (gamma) : ", gamma)
    print("--------------------------------------------------------------------------------------------")
    print("optimizer learning rate actor : ", lr_actor)
    print("optimizer learning rate critic : ", lr_critic)
    if random_seed:
        print("--------------------------------------------------------------------------------------------")
        print("setting random seed to ", random_seed)
    print("============================================================================================")
    #####################################################

    # Set random seed if specified
    if random_seed:
        torch.manual_seed(random_seed)
        np.random.seed(random_seed)


    # Create environment
    env = gym.make(env_name)

    # Get state and action dimensions
    state_dim = flatdim(env.observation_space)
    if has_continuous_action_space:
        action_dim = flatdim(env.action_space)
    else:
        action_dim = env.action_space.n

    # Setup checkpointing - save to volume for persistence
    checkpoint_dir = os.path.join(volume_mount_path, env_name)
    os.makedirs(checkpoint_dir, exist_ok=True)
    checkpoint_path = os.path.join(checkpoint_dir, f"{env_name}_{random_seed}_{run_num_pretrained}.pth")

    # Initialize PPO agent
    ppo_agent = PPO(
        state_dim,
        action_dim,
        lr_actor,
        lr_critic,
        gamma,
        K_epochs,
        eps_clip,
        has_continuous_action_space,
        action_std if action_std else 0.6
    )

    # Track training
    start_time = datetime.now().replace(microsecond=0)
    print("Started training at (GMT) : ", start_time)


    # Initialize ClickHouse logger
    logger = ClickHouseLogger()
    run_id = f"{env_name}_{random_seed}_{run_num_pretrained}_{start_time.strftime('%Y%m%d_%H%M%S')}"

    # lf = get_client()

    time_step = 0
    i_episode = 0

    # Training loop
    while time_step <= max_training_timesteps:
        # Only pass seed on first reset to maintain randomness across episodes
        reset_seed = random_seed if (random_seed and i_episode == 0) else None
        state, info = env.reset(seed=reset_seed)
        current_ep_reward = 0

        # with lf.start_as_current_observation(as_type="span", name=f'epsiode_{i_episode}', session_id=run_id) as root:
            # root.update_trace(input={"initial_state": state.tolist()})
        for t in range(1, max_ep_len + 1):
            # with lf.start_as_current_observation(as_type="span", name=f"step_{t}") as step_span:
            # Select action with policy
            action = ppo_agent.select_action(state)
            state, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated

            # Save reward and terminal flags
            ppo_agent.buffer.rewards.append(reward)
            ppo_agent.buffer.is_terminals.append(terminated)

            # step_span.score(name="reward", value=float(reward), data_type="NUMERIC")
            # step_span.score(name="timestep", value=int(t), data_type="NUMERIC")
            # step_span.score(name="action", value=float(action), data_type="NUMERIC")

            # step_span.update(
            #     input={"state": state.tolist()},
            #     output={"action": float(action), "reward": float(reward)}
            # )

            time_step += 1
            current_ep_reward += reward

            # Update PPO agent
            if time_step % update_timestep == 0:
                ppo_agent.update()

            # Decay action std if continuous action space
            if (has_continuous_action_space and
                action_std_decay_freq and
                time_step % action_std_decay_freq == 0):
                ppo_agent.decay_action_std(
                    action_std_decay_rate,
                    min_action_std
                )

            # Save model weights
            if time_step % save_model_freq == 0:
                print("--------------------------------------------------------------------------------------------")
                print(f"saving model at : {checkpoint_path}")
                ppo_agent.save(checkpoint_path)
                
                # Verify file exists before committing
                if os.path.exists(checkpoint_path):
                    file_size = os.path.getsize(checkpoint_path)
                    print(f"Model file saved successfully. Size: {file_size} bytes")
                    
                    # Commit volume to persist the model
                    try:
                        volume.commit()
                        print("Volume committed successfully - model persisted to volume")
                    except Exception as e:
                        print(f"Error committing volume: {e}")
                else:
                    print(f"ERROR: Model file not found at {checkpoint_path} after save operation!")
                
                print("Elapsed Time  : ", datetime.now().replace(microsecond=0) - start_time)
                print("--------------------------------------------------------------------------------------------")

            # Break if episode is over
            if done:
                break
            
            ##### Logging episode data #####

        # Get current action_std if continuous action space
        current_action_std = None
        if has_continuous_action_space:
            current_action_std = ppo_agent.action_std

        # Log episode to ClickHouse
        try:
            logger.log_step(
                episode=i_episode,
                timestep=time_step,
                reward=current_ep_reward,
                action_std=current_action_std,
                env_name=env_name,
                run_id=run_id
            )
        except Exception as e:
            print("Error logging to ClickHouse. Silently passing: ", e)
            pass

        # root.update_trace(
        #     output={"episode_reward": float(current_ep_reward), "episode_length": int(t), "final_state": state.tolist(), "end_reason": "terminated" if terminated else "truncated"}
        # )

        # Flush Langfuse buffer after each 100 episodes
        # if i_episode % 100:
        #     try:
        #         lf.flush()
        #     except Exception as e:
        #         print(f"Warning: Failed to flush Langfuse buffer. Silently passing: {e}")

        i_episode += 1

        if i_episode % 100 == 0:
            print("Episode : {} \t\t Timestep : {} \t\t Reward : {}".format(i_episode, time_step, round(current_ep_reward, 2)))

    env.close()

    # Flush and close ClickHouse logger
    try:
        logger.flush()
        logger.close()
    except Exception as e:
        print(f"Warning: Failed to close ClickHouse logger: {e}")

    # try:
    #     lf.flush()
    #     lf.shutdown()
    # except Exception as e:
    #     print(f"Warning: Failed to shutdown Langfuse client: {e}")

    end_time = datetime.now().replace(microsecond=0)
    print("============================================================================================")
    print("Started training at (GMT) : ", start_time)
    print("Finished training at (GMT) : ", end_time)
    print("Total training time  : ", end_time - start_time)
    print("============================================================================================")

    return {
        "checkpoint_path": checkpoint_path,
        "total_episodes": i_episode,
        "total_timesteps": time_step
    }


if __name__ == '__main__':
    # modal run train.py
    train.call()
    