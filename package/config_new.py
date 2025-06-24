import os

# กำหนด Paths หลักสำหรับ Dataset
BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "data"))

# ───────────────────────────────────────────────────────────
# Paths ของ Mosquito และ Noise Data
# ───────────────────────────────────────────────────────────
INDOOR_DIR = os.path.join(BASE_DIR, "indoor_final_8kHz_16bits_normalize")
OUTDOOR_DIR = os.path.join(BASE_DIR, "outdoor_96kHz_16bits")
HUMBUGDB_DIR = os.path.join(BASE_DIR, "humbugDB_8kHz_16bits")
MOSQUITO_DIRS = [INDOOR_DIR, HUMBUGDB_DIR]
OUTDOOR_MOSQUITO_DIRS = [OUTDOOR_DIR] 
MOSQUITO_DIR = os.path.join(BASE_DIR, "mosquito_sounds")  # ไดเรกทอรียุง
NOISE_DIR = os.path.join(BASE_DIR, "Environmental_Noise")  # ไดเรกทอรีเสียงพื้นหลัง

# ───────────────────────────────────────────────────────────
# Train & Validation Noise Data
# ───────────────────────────────────────────────────────────
NOISE_TRAIN_DIR = os.path.join(NOISE_DIR, "Env_Train_Norm")  
NOISE_VALTEST_DIR = os.path.join(NOISE_DIR, "Env_Val_test_Norm")  # SubFolder -> Env_Urban, Env_Forest

# ───────────────────────────────────────────────────────────
# Paths สำหรับ Simulation
# ───────────────────────────────────────────────────────────
SIMULATED_DIR = os.path.join(BASE_DIR, "simulated_audio")  # ที่เก็บไฟล์เสียงที่จำลอง
DATASET_DIR = os.path.join(BASE_DIR, "simulated_dataset")  # โฟลเดอร์หลักของ Dataset

SIMULATED_TRAIN_DIR = os.path.join(DATASET_DIR, "train")  # ที่เก็บไฟล์ Train
SIMULATED_VAL_DIR = os.path.join(DATASET_DIR, "val")  # ที่เก็บไฟล์ Validation
SIMULATED_TEST_DIR = os.path.join(DATASET_DIR, "test")  # ที่เก็บไฟล์ Test

# จำนวน Simulation Audio
TOTAL_SIMULATIONS = 1500
NUM_SIMULATIONS = {
    "train": round(TOTAL_SIMULATIONS * 0.8),  # 80% สำหรับ Training
    "val": round(TOTAL_SIMULATIONS * 0.1),    # 10% สำหรับ Validation
    "test": round(TOTAL_SIMULATIONS * 0.1)    # 10% สำหรับ Testing
}
output_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "output", "evaluation"))

# ───────────────────────────────────────────────────────────
# Paths สำหรับ Metadata ของ Simulation
# ───────────────────────────────────────────────────────────
METADATA_DIR = os.path.join(DATASET_DIR, "metadata")

# ───────────────────────────────────────────────────────────
# Paths สำหรับ Segmented Audio Metadata (แบ่งตาม Environment)
# ───────────────────────────────────────────────────────────
SEGMENT_DIR = os.path.join(DATASET_DIR, "segments")

SEGMENT_TRAIN = os.path.join(SEGMENT_DIR, "train")
SEGMENT_VAL = os.path.join(SEGMENT_DIR, "val")
SEGMENT_TEST = os.path.join(SEGMENT_DIR, "test")

# Path สำหรับ Segment Metadata 
SEGMENT_METADATA_TRAIN_URBAN = os.path.join(SEGMENT_TRAIN, "train_segment_metadata_urban.csv")
SEGMENT_METADATA_TRAIN_FOREST = os.path.join(SEGMENT_TRAIN, "train_segment_metadata_forest.csv")

SEGMENT_METADATA_VAL_URBAN = os.path.join(SEGMENT_VAL, "val_segment_metadata_urban.csv")
SEGMENT_METADATA_VAL_FOREST = os.path.join(SEGMENT_VAL, "val_segment_metadata_forest.csv")

SEGMENT_METADATA_TEST_URBAN = os.path.join(SEGMENT_TEST, "test_segment_metadata_urban.csv")
SEGMENT_METADATA_TEST_FOREST = os.path.join(SEGMENT_TEST, "test_segment_metadata_forest.csv")

# สร้างไดเรกทอรีทั้งหมดถ้ายังไม่มี
for path in [SEGMENT_TRAIN, SEGMENT_VAL, SEGMENT_TEST]:
    os.makedirs(path, exist_ok=True)

# ───────────────────────────────────────────────────────────
# Sampling Parameters
# ───────────────────────────────────────────────────────────
SAMPLING_RATE = 8000
AUDIO_DURATION = 10  # วินาที
TARGET_LEN = int(0.3 * SAMPLING_RATE)
BIT_DEPTH = 16
WIN = 256
HOP = 64
NFFT = 512

# ───────────────────────────────────────────────────────────
# SNR และ Environment Configuration
# ───────────────────────────────────────────────────────────
ENVIRONMENTS = {
    "urban": {
        "noise_subdir": "Env_Urban",
        "mosquito_species": ["Ae.Aegypti", "Cx.Quin"],
        "noise_level": 1.5,
        "snr_range": (5, 15)
    },
    "forest": {
        "noise_subdir": "Env_Forest",
        "mosquito_species": ["An.Dirus", "Ae.Albopictus"],
        "noise_level": 0.5,
        "snr_range": (15, 25)
    }
}
# ───────────────────────────────────────────────────────────
# Output สำหรับแต่ละ Environment (แยกเก็บผลลัพธ์)
# ───────────────────────────────────────────────────────────

def set_env_scope(env):
    global ENV_SCOPE, OUTPUT_DIR, MODEL_DIR, EVAL_DIR, TENSORBOARD_RUN_DIR
    ENV_SCOPE = env.lower()

    OUTPUT_BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "output"))
    OUTPUT_DIR = os.path.join(OUTPUT_BASE_DIR, ENV_SCOPE)

    MODEL_DIR = os.path.join(OUTPUT_DIR, "models")
    EVAL_DIR = os.path.join(OUTPUT_DIR, "evaluation")
    TENSORBOARD_RUN_DIR = os.path.join(OUTPUT_DIR, "runs")

    for path in [MODEL_DIR, EVAL_DIR, TENSORBOARD_RUN_DIR]:
        os.makedirs(path, exist_ok=True)


# ───────────────────────────────────────────────────────────
# สร้างไดเรกทอรีหลักถ้ายังไม่มี
# ───────────────────────────────────────────────────────────
for path in [DATASET_DIR, METADATA_DIR]:
    os.makedirs(path, exist_ok=True)

print("Configuration paths are set up!")
