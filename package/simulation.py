# Simulation.py
import os
import librosa
import soundfile as sf
import numpy as np
import random
import pandas as pd
import config_new as config


def normalize(audio, target_level=-20.0):

    rms = np.sqrt(np.mean(audio**2))  # Compute RMS
    if rms > 0:  # Avoid division by zero
        scalar = 10 ** (target_level / 20) / rms
        audio *= scalar
    return np.clip(audio, -1.0, 1.0)

def get_mosquito_species(mosquito_dirs):
    all_species_files = {}

    for mosquito_dir in mosquito_dirs:
        if not os.path.exists(mosquito_dir):
            continue  

        for f in os.listdir(mosquito_dir):
            if f.endswith(".wav") and not f.startswith("An.Minimus"):
                species = f.split("_")[0]  
                if species not in all_species_files:
                    all_species_files[species] = []
                all_species_files[species].append(os.path.join(mosquito_dir, f))

    if not all_species_files:
        raise ValueError("No mosquito files found!")

    # ใช้ weighted sampling เพื่อให้ทุก species มีโอกาสถูกเลือกเท่ากัน
    species = random.choice(list(all_species_files.keys()))
    selected_file = random.choice(all_species_files[species])

    return selected_file, species


def get_train_noise_files(directory):
    """
    Retrieve and categorize mixed noise files in the train dataset based on filename.
    Stores both file paths and file names separately.
    """
    if not os.path.exists(directory):
        raise ValueError(f"Train noise directory not found: {directory}")

    noise_data = {
        "urban": {"file_paths": [], "file_names": []},
        "forest": {"file_paths": [], "file_names": []}
    }

    # List all noise files (exclude directories)
    noise_files = [f for f in os.listdir(directory) if os.path.isfile(os.path.join(directory, f)) and f.endswith(".wav")]

    if not noise_files:
        raise ValueError(f"No noise files found in {directory}")

    # Categorize by environment based on filename (case insensitive)
    for f in noise_files:
        file_path = os.path.join(directory, f)
        fname = f.lower()  # Normalize case

        if "urban" in fname:
            noise_data["urban"]["file_paths"].append(file_path)
            noise_data["urban"]["file_names"].append(f)
        elif "forest" in fname:
            noise_data["forest"]["file_paths"].append(file_path)
            noise_data["forest"]["file_names"].append(f)
        else:
            print(f" Warning: Unknown environment in file {f}, skipping!")

    return noise_data

def get_valtest_noise_files(directory):
   
    if not os.path.exists(directory):
        raise ValueError(f"Validation/Test noise directory not found: {directory}")

    noise_data = {
        "urban": {"file_paths": [], "file_names": []},
        "forest": {"file_paths": [], "file_names": []}
    }

    for env in os.listdir(directory):
        env_path = os.path.join(directory, env)

        if os.path.isdir(env_path):
            noise_files = [f for f in os.listdir(env_path) if os.path.isfile(os.path.join(env_path, f)) and f.endswith(".wav")]
            
            if not noise_files:
                print(f"Warning: No noise files found in {env_path}")

            # ปรับให้ตรงกับโครงสร้าง `ENVIRONMENTS`
            if "urban" in env.lower():
                noise_data["urban"]["file_paths"].extend([os.path.join(env_path, f) for f in noise_files])
                noise_data["urban"]["file_names"].extend(noise_files)
            elif "forest" in env.lower():
                noise_data["forest"]["file_paths"].extend([os.path.join(env_path, f) for f in noise_files])
                noise_data["forest"]["file_names"].extend(noise_files)
            else:
                print(f"Warning: Unknown environment in folder {env}, skipping!")

    return noise_data



def get_environment_type(species):

    # print(f"Checking environment for species: {species}")

    for env, settings in config.ENVIRONMENTS.items():
        if species in settings["mosquito_species"]:
            # print(f" Matched Environment: {env} for {species}")
            return env  

    # print("No matching environment found, defaulting to 'urban'")
    return "urban"


def get_noise_files(noise_dir, environment_type):
    """
    Retrieve available noise files based on environment type.
    """
    noise_subdir = config.ENVIRONMENTS[environment_type]["noise_subdir"]
    noise_path = os.path.join(noise_dir, noise_subdir)

    if not os.path.exists(noise_path):
        raise ValueError(f"Missing noise directory: {noise_path}")
    
    noise_files = [f for f in os.listdir(noise_path) if f.endswith(".wav")]
    
    if not noise_files:
        raise ValueError(f"No noise files found in {noise_path}")
    
    return noise_files, noise_path



def apply_smooth_volume(audio, sr, fade_in_time=1.0, fade_out_time=1.0, method="sigmoid"):
    fade_in_samples = min(max(int(fade_in_time * sr), 1), len(audio) // 2)
    fade_out_samples = min(max(int(fade_out_time * sr), 1), len(audio) // 2)

    if len(audio) < 2 * max(fade_in_samples, fade_out_samples):
        print(f" Warning: Audio is too short for requested fade-in/out times. Skipping fade.")
        return audio  

    fade_in_curve = np.ones(fade_in_samples)
    fade_out_curve = np.ones(fade_out_samples)

    if method == "linear":
        fade_in_curve = np.linspace(0, 1, fade_in_samples)
        fade_out_curve = np.linspace(1, 0, fade_out_samples)

    elif method == "exponential":
        fade_in_curve = (np.exp(np.linspace(0, 4, fade_in_samples)) - 1) / (np.exp(4) - 1)
        fade_out_curve = np.flip(fade_in_curve)

    elif method == "sigmoid":
        x = np.linspace(-6, 6, fade_in_samples)
        fade_in_curve = 1 / (1 + np.exp(-x))
        fade_out_curve = np.flip(fade_in_curve)

    # Apply fade-in and fade-out
    dynamic_audio = audio.copy()
    dynamic_audio[:fade_in_samples] *= fade_in_curve
    dynamic_audio[-fade_out_samples:] *= fade_out_curve

    return dynamic_audio

# def prep_env(noise_dir, mosquito_dirs, env_target,dataset_type=None):
#     """
#     ปรับ prep_env เพื่อบังคับให้เลือก environment ตาม env_target
#     """
#     # กรอง mosquito files ที่ตรงกับ env_target โดยใช้ config.ENVIRONMENTS
#     matching_files = []
#     for mosquito_dir in mosquito_dirs:
#         if not os.path.exists(mosquito_dir):
#             continue
#         for f in os.listdir(mosquito_dir):
#             if f.endswith(".wav") and not f.startswith("An.Minimus"):
#                 species = f.split("_")[0]
#                 # ตรวจสอบว่า species นี้อยู่ในกลุ่มของ env_target หรือไม่
#                 if species in config.ENVIRONMENTS[env_target]["mosquito_species"]:
#                     matching_files.append(os.path.join(mosquito_dir, f))
#     if not matching_files:
#         raise ValueError(f"No mosquito files found for environment: {env_target}")
#     selected_file = random.choice(matching_files)
#     species = os.path.basename(selected_file).split("_")[0]
#     env_type = env_target  

#     # # ดึง noise files สำหรับ env_target
#     # noise_files, noise_path = get_noise_files(noise_dir, env_target)
#     # noise_file = random.choice(noise_files)
#     # noise_filepath = os.path.join(noise_path, noise_file)
#     if dataset_type == "train":
#         noise_data = get_train_noise_files(noise_dir)
#     else:
#         noise_data = get_valtest_noise_files(noise_dir)

#     noise_files = noise_data[env_target]["file_paths"]
#     noise_file = random.choice(noise_files)



#     background_noise, _ = librosa.load(noise_file, sr=config.SAMPLING_RATE, mono=True)
#     total_samples = int(config.AUDIO_DURATION * config.SAMPLING_RATE)
#     if len(background_noise) > total_samples:
#         start_idx = random.randint(0, len(background_noise) - total_samples)
#         background_noise = background_noise[start_idx:start_idx + total_samples]
#     else:
#         background_noise = np.pad(background_noise, (0, total_samples - len(background_noise)), mode='wrap')

#     subenv = noise_file.split('_')[2]  
#     noise_level = config.ENVIRONMENTS[env_target]["noise_level"]
#     background_noise *= noise_level  
#     background_noise = np.sign(background_noise) * np.abs(background_noise) ** 0.8
#     background_noise /= np.max(np.abs(background_noise))
    
#     return background_noise, env_type, noise_file, subenv


def prep_env(noise_dir, mosquito_dirs, env_target, dataset_type=None, use_gaussian_for_outdoor=False):
    # ---------- (1) สุ่มไฟล์ยุงก่อน ----------
    matching_files = []
    for mosquito_dir in mosquito_dirs:
        if not os.path.exists(mosquito_dir):
            continue
        for f in os.listdir(mosquito_dir):
            if f.endswith(".wav") and not f.startswith("An.Minimus"):
                species = f.split("_")[0]
                if species in config.ENVIRONMENTS[env_target]["mosquito_species"]:
                    matching_files.append((os.path.join(mosquito_dir, f), mosquito_dir))

    if not matching_files:
        raise ValueError(f"No mosquito files found for environment: {env_target}")

    selected_file, source_dir = random.choice(matching_files)
    species = os.path.basename(selected_file).split("_")[0]
    env_type = env_target

    # ---------- (2) ตรวจว่าไฟล์มาจาก OUTDOOR ----------
    is_outdoor = config.OUTDOOR_DIR in os.path.abspath(source_dir)

    # ---------- (3) เลือก background noise ----------
    total_samples = int(config.AUDIO_DURATION * config.SAMPLING_RATE)
    if use_gaussian_for_outdoor and is_outdoor:
        std = 0.03  
        background_noise = np.random.normal(0, std, total_samples)
        background_noise = np.clip(background_noise, -1.0, 1.0)
        noise_file = "gaussian_noise"
        subenv = "synthetic"
    else:
        if dataset_type == "train":
            noise_data = get_train_noise_files(noise_dir)
        else:
            noise_data = get_valtest_noise_files(noise_dir)

        noise_files = noise_data[env_target]["file_paths"]
        noise_file = random.choice(noise_files)
        background_noise, _ = librosa.load(noise_file, sr=config.SAMPLING_RATE, mono=True)

        if len(background_noise) > total_samples:
            start_idx = random.randint(0, len(background_noise) - total_samples)
            background_noise = background_noise[start_idx:start_idx + total_samples]
        else:
            background_noise = np.pad(background_noise, (0, total_samples - len(background_noise)), mode='wrap')

        subenv = noise_file.split('_')[2]
        noise_level = config.ENVIRONMENTS[env_target]["noise_level"]
        background_noise *= noise_level
        background_noise = np.sign(background_noise) * np.abs(background_noise) ** 0.8
        background_noise /= np.max(np.abs(background_noise))

    # ---------- ส่งคืนเพิ่ม selected_file ----------
    return background_noise, env_type, noise_file, subenv, selected_file,source_dir


def generate_random_pos(duration, min_events=2, min_interval=0.2, mean_interval=1.0, 
                                 noise_prob=0.5, mosquito_clump_prob=0.1):
    intervals = []
    current_time = 0.0

    while current_time < duration:
        first_event = "noise" if random.random() < noise_prob else "mosquito"

        interval_length = max(min_interval, random.expovariate(1.0 / mean_interval))
        if current_time + interval_length > duration:
            break

        end_time = current_time + interval_length

        if first_event == "noise":
            intervals.append(("Noise", current_time, end_time))
            current_time = end_time  # ให้ Noise จบแล้ว Mosquito ต่อเนื่องทันที
            
            mos_start = current_time  # ไม่มีช่วงว่าง
            if mos_start < duration:
                mos_end = mos_start + max(min_interval, random.expovariate(1.0 / mean_interval))
                if mos_end > duration:
                    mos_end = duration
                intervals.append(("Mosquito", mos_start, mos_end))
                current_time = mos_end  # ให้ Mosquito จบแล้วไป Noise ต่อเนื่องทันที

                # มีโอกาสให้ Mosquito Events ติดกัน
                while random.random() < mosquito_clump_prob and current_time < duration:
                    next_start = current_time  # ให้ Mosquito ถัดไปเริ่มต่อเนื่อง
                    next_end = next_start + max(min_interval, random.expovariate(1.0 / mean_interval))
                    if next_end > duration:
                        next_end = duration
                    intervals.append(("Mosquito", next_start, next_end))
                    current_time = next_end  # ไป Mosquito ต่อเนื่อง

        else:
            intervals.append(("Mosquito", current_time, end_time))
            current_time = end_time  # ให้ Mosquito จบแล้ว Noise ต่อเนื่องทันที

            noise_start = current_time  # ไม่มีช่องว่างระหว่าง Mosquito -> Noise
            if noise_start < duration:
                noise_end = noise_start + max(min_interval, random.expovariate(1.0 / mean_interval))
                if noise_end > duration:
                    noise_end = duration
                intervals.append(("Noise", noise_start, noise_end))
                current_time = noise_end  # ไป Mosquito ต่อเนื่อง

    # ตรวจสอบให้แน่ใจว่ามีอย่างน้อย `min_events` ของ Mosquito
    mosquito_events = [event for event in intervals if event[0] == "Mosquito"]
    while len(mosquito_events) < min_events:
        last_end_time = intervals[-1][2] if intervals else 0.0
        new_start = last_end_time
        if new_start + min_interval > duration:
            break
        new_end = new_start + random.uniform(min_interval, mean_interval)
        intervals.append(("Mosquito", new_start, min(duration, new_end)))
        mosquito_events.append(("Mosquito", new_start, new_end))

    return intervals  



def prepare_mos(environment_type, mosquito_dirs, background_noise, event_intervals):
    """
    Generate mosquito audio signals with randomized events and metadata.
    """
    env_settings = config.ENVIRONMENTS[environment_type]
    allowed_species = env_settings["mosquito_species"]
    snr_range = env_settings["snr_range"]

    random.seed(None)  
    np.random.seed(None)

    mosquito_files = []
    for mosquito_dir in mosquito_dirs:
        for f in os.listdir(mosquito_dir):
            if f.endswith(".wav") and f.split("_")[0] in allowed_species and f.split("_")[1] != 'nan':
                mosquito_files.append(os.path.join(mosquito_dir, f))

    total_samples = int(config.AUDIO_DURATION * config.SAMPLING_RATE)
    random.shuffle(mosquito_files)

    mosquito_audio = np.zeros(total_samples)
    metadata = {"audio_labels": [], "event_intervals": []}
    snr_values = np.random.uniform(snr_range[0], snr_range[1], size=len(event_intervals))

    previous_end = 0  # Track last event end time

    for (event_type, start, end), snr in zip(event_intervals, snr_values):
        start_sample, end_sample = int(start * config.SAMPLING_RATE), int(end * config.SAMPLING_RATE)

        # กรณีเป็น Noise Event → แค่บันทึก metadata (ไม่ต้องเพิ่มเสียง)
        if event_type == "Noise":
            metadata["audio_labels"].append({
                "event_type": "Noise",
                "start_time": start,
                "end_time": end,
                "species": None,
                "sex": None,
                "label": 'Noise',
                "snr": None,
                "environment": environment_type
            })
            previous_end = end  # อัปเดตตำแหน่งล่าสุด
            continue  

        # ไม่ใช้เสียงยุงซ้ำ (ใช้ pop())
        if not mosquito_files:
            print("Refreshing mosquito list...")
            mosquito_files = [os.path.join(mosquito_dir, f) for mosquito_dir in mosquito_dirs
                              for f in os.listdir(mosquito_dir)
                              if f.endswith(".wav") and f.split("_")[0] in allowed_species]
            random.shuffle(mosquito_files)  # รีเซ็ตและสับใหม่

        mosquito_file = mosquito_files.pop()  # ใช้ pop() เพื่อไม่ให้ซ้ำ
        filename = os.path.basename(mosquito_file)
        
        parts = filename.split("_")
        species_name = parts[0] if len(parts) > 0 else "Unknown"
        if len(parts[1]) == 2 :
            sex = parts[1][1] 
        else :
            continue

        mosquito_sound, _ = librosa.load(mosquito_file, sr=config.SAMPLING_RATE, mono=True)

        mosquito_duration = end_sample - start_sample
        mosquito_sound = np.pad(mosquito_sound[:mosquito_duration], (0, max(0, mosquito_duration - len(mosquito_sound))), 'constant')

        
        mosquito_sound = apply_smooth_volume(mosquito_sound, config.SAMPLING_RATE, fade_in_time=0.7, fade_out_time=0.7, method="exponential")

        
        
        # คำนวณ Scaling Factor ให้ SNR อยู่ในช่วงที่กำหนด
        noise_power = np.mean(background_noise[start_sample:end_sample] ** 2)
        mosquito_power = np.mean(mosquito_sound ** 2)
        snr_linear = 10 ** (snr / 10)
        scaling_factor = np.sqrt(noise_power * snr_linear / (mosquito_power + 1e-9))

        scaling_factor = np.sqrt(noise_power * snr_linear / (mosquito_power + 1e-9))
        mosquito_sound *= scaling_factor * np.random.uniform(0.9, 1.1)


        mosquito_audio[start_sample:end_sample] += mosquito_sound


        # บันทึก Mosquito Event
        metadata["audio_labels"].append({
            "event_type": "Mosquito",
            "start_time": start,
            "end_time": end,
            "species": species_name,
            "sex": sex,
            "label":f"{species_name}_{sex}",
            "snr": round(snr, 2),
            "environment": environment_type
        })

        previous_end = end  # อัปเดตตำแหน่งล่าสุด

    return mosquito_audio, metadata





def process_simulation(mosquito_dirs, noise_dir, output_dir, num_simulations,env_target,dataset_type):
    """
    Generate simulated mosquito sound data.
    """
    os.makedirs(output_dir, exist_ok=True)

    metadata_list = []
    random.seed(None)
    np.random.seed(None)

    if isinstance(mosquito_dirs, set):
        mosquito_dirs = list(mosquito_dirs)

    for i in range(num_simulations):
        background_noise, env_type, noise_file, subenv, selected_file, source_dir = prep_env(
    noise_dir, mosquito_dirs, env_target, dataset_type, use_gaussian_for_outdoor=True)
        mosquito_intervals = generate_random_pos(config.AUDIO_DURATION)
        mosquito_audio, metadata = prepare_mos(env_type, mosquito_dirs, background_noise, mosquito_intervals)

        # รวมเสียงยุง + Background Noise
        simulated_audio = background_noise + mosquito_audio


        # ปรับความดังให้คงที่ (Normalize)
        simulated_audio = normalize(simulated_audio)

        # บันทึกไฟล์เสียงที่ถูก Generate

        # เดิม
# audio_filename = f"{env_type}_simulated_{dataset_type}_{i+1}.wav"

    # แก้ใหม่
        is_outdoor = 'outdoor' in os.path.basename(source_dir).lower()

        audio_filename = f"{env_type}_{'outdoor_' if is_outdoor else ''}simulated_{dataset_type}_{i+1}.wav"
        audio_filepath = os.path.join(output_dir, audio_filename)
        sf.write(audio_filepath, simulated_audio, config.SAMPLING_RATE)
        
        
        if mosquito_intervals:
            last_mosquito = mosquito_intervals[-1]

            # ตรวจสอบว่า Mosquito ตัวสุดท้ายไปจนถึง AUDIO_DURATION หรือไม่
            if last_mosquito[2] < config.AUDIO_DURATION:
                last_noise_exists = any(
                    entry["event_type"] == "Noise" and entry["start_time"] == last_mosquito[2] 
                    for entry in metadata["audio_labels"]
                )

                if not last_noise_exists:
                    metadata["audio_labels"].append({
                        "event_type": "Noise",
                        "start_time": last_mosquito[2],  # Noise เริ่มที่เวลาสิ้นสุดของ Mosquito สุดท้าย
                        "end_time": config.AUDIO_DURATION,
                        "species": None,
                        
                        "sex": None,
                        "label":"Noise",
                        "snr": None,
                        "environment": env_type
                    })


        for entry in metadata["audio_labels"]:
            entry["file_path"]= audio_filepath
            entry["file_name"] = audio_filename
            entry["duration"] = librosa.get_duration(y=simulated_audio, sr=config.SAMPLING_RATE)
            entry["noise_file"] = noise_file
            entry["subenv"] = subenv
            entry["mos_filepath"] =  selected_file
            entry["mos_dataset"] = os.path.basename(source_dir)  # 👈 เพิ่มตรงนี้

            metadata_list.append(entry)

    metadata_df = pd.DataFrame(metadata_list)

    # Save Metadata

    # ตรวจสอบว่าไฟล์ยุงมาจาก outdoor หรือไม่
    if 'outdoor' in os.path.basename(source_dir).lower():
        metadata_filename = f"simulation_metadata_{env_target}_outdoor_{dataset_type}.csv"
    else:
        metadata_filename = f"simulation_metadata_{env_target}_{dataset_type}.csv"

    metadata_csv_path = os.path.join(config.METADATA_DIR, metadata_filename)


    # metadata_csv_path = os.path.join(config.METADATA_DIR, f"simulation_metadata_{env_target}_{dataset_type}.csv")
    metadata_df.to_csv(metadata_csv_path, index=False)

    print(f"{dataset_type} {env_type} Metadata Preview:")
    print(metadata_df.head(20))  

    return simulated_audio, metadata_df