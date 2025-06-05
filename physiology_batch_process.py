import sys
import os
import gc
from physiologycheck import physiologyprocess
import psutil

def memory_debug(tag=""):
    mem = psutil.Process(os.getpid()).memory_info().rss / (1024 ** 2)
    print(f"[{tag}] 💾 {mem:.1f}MB", flush=True)

data_root = 'D:/LabRoom/Projects/SD Physiology/Processed/main'
output_dir = 'D:/LabRoom/Projects/SD Physiology/Processed/processed_individual'
log_dir = './logs'

participant_list = sys.argv[1:]

print(f"[🔧 Subprocess] Received {len(participant_list)} participants\n")

for i, pid in enumerate(participant_list):
    print(f"[Sub-Batch] ▶ Processing {pid} ({i + 1}/{len(participant_list)})")
    memory_debug("Before")

    physiologyprocess(
        data_root=data_root,
        output_dir=output_dir,
        log_dir=log_dir,
        selected_participants=[pid],
        figure=False,
    )

    gc.collect()
    memory_debug("After")

print("✅ Sub-Batch complete.\n")
