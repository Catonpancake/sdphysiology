import os
import subprocess

# === Configuration ===
data_root = 'D:/LabRoom/Projects/SD Physiology/Processed/main'
batch_size = 30

all_participants = sorted(os.listdir(data_root))
total_batches = (len(all_participants) + batch_size - 1) // batch_size

print(f"🔧 Total Participants: {len(all_participants)} | Batch Size: {batch_size} | Total Batches: {total_batches}\n")

for batch_index in range(total_batches):
    batch_start = batch_index * batch_size
    batch_end = min((batch_index + 1) * batch_size, len(all_participants))
    batch_participants = all_participants[batch_start:batch_end]

    print(f"\n🚀 Running Batch {batch_index + 1}/{total_batches} ({batch_start + 1} to {batch_end})")
    print(f"▶ Selected Participants: {batch_participants}\n")

    try:
        subprocess.run(
            ["python", "physiology_batch_process.py"] + batch_participants,
            check=True
        )
    except subprocess.CalledProcessError as e:
        print(f"❌ Error in Batch {batch_index + 1}: {e}")
        continue

    print(f"✅ Finished Batch {batch_index + 1}\n")

print("🎉 All batches completed!")
