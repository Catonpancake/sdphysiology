# import sys
# import os
# import gc
# from physiologycheck import physiologyprocess
# import psutil

# ## This script process physiolgy data for each participant in a sub-batch, due to memory issue.


# def memory_debug(tag=""):
#     mem = psutil.Process(os.getpid()).memory_info().rss / (1024 ** 2)
#     print(f"[{tag}] 💾 {mem:.1f}MB", flush=True)

# data_root = 'D:/LabRoom/Projects/SD Physiology/Processed/main'
# output_dir = 'D:/LabRoom/Projects/SD Physiology/Processed/processed_individual'
# log_dir = './logs'

# participant_list = sys.argv[1:]

# print(f"[🔧 Subprocess] Received {len(participant_list)} participants\n")

# for i, pid in enumerate(participant_list):
#     print(f"[Sub-Batch] ▶ Processing {pid} ({i + 1}/{len(participant_list)})")
#     memory_debug("Before")

#     physiologyprocess(
#         data_root=data_root,
#         output_dir=output_dir,
#         log_dir=log_dir,
#         selected_participants=[pid],
#         figure=False,
#         process_ppg=False, 
#         process_eda=False, 
#         process_rsp=True
#     )

#     gc.collect()
#     memory_debug("After")

# print("✅ Sub-Batch complete.\n")


import sys
import os
import gc
from physiologycheck import physiologyprocess, process_rsp_only_all_channels
import psutil
import pandas as pd
import bioread
import matplotlib.pyplot as plt
# ======= 경로 설정 =======
data_root = 'D:/LabRoom/Projects/SD Physiology/Processed/main'
rsp_raw_dir = 'C:/Users/Jiyoon/OneDrive/HubRoom/Projects/SD physiology/Data/Preprocessing/RSP'
output_dir = 'D:/LabRoom/Projects/SD Physiology/Processed/processed_individual'
log_dir = './logs'
log_path = os.path.join(log_dir, 'rsp_filter_check.csv')
fig_save_dir = './rsp_filter_figs'
os.makedirs(fig_save_dir, exist_ok=True)
# ======= 유틸 =======
def memory_debug(tag=""):
    mem = psutil.Process(os.getpid()).memory_info().rss / (1024 ** 2)
    print(f"[{tag}] 💾 {mem:.1f}MB", flush=True)

# ======= 설정 =======
STD_THRESHOLD = 2.0  # 필터링 안된 경우 std가 이 이상일 것으로 추정됨

# ======= 필터 이슈 로그용 리스트 =======
rsp_filter_issues = []

# ======= 받은 participant 목록 =======
participant_list = sys.argv[1:]
print(f"[🔧 Subprocess] Received {len(participant_list)} participants\n")

for i, pid in enumerate(participant_list):
    print(f"[Sub-Batch] ▶ Processing {pid} ({i + 1}/{len(participant_list)})")
    memory_debug("Before")

    # (1) RSP 필터 여부 확인
    acq_filename = [f for f in os.listdir(rsp_raw_dir) if f.startswith(pid) and f.endswith(".acq")]
    if acq_filename:
        acq_path = os.path.join(rsp_raw_dir, acq_filename[0])
        try:
            data = bioread.read_file(acq_path)
            rsp_channels = [ch for ch in data.channels if "RSP" in ch.name]

            if len(rsp_channels) >= 2:
                rsp2 = rsp_channels[1].data
                std2 = rsp2[:5000].std()

                if std2 > STD_THRESHOLD:
                    rsp_filter_issues.append({
                        "participant": pid,
                        "channel2_std": std2,
                        "note": f"Channel 2 std > {STD_THRESHOLD} (가능성: 필터 미적용)"
                    })
                    
                plt.figure(figsize=(10, 2))
                plt.plot(rsp2)
                plt.title(f'{pid} - RSP Channel 2 (first 5000 samples)\nSTD={std2:.3f}')
                plt.xlabel('Samples')
                plt.ylabel('Amplitude')
                plt.tight_layout()
                plt.savefig(os.path.join(fig_save_dir, f'{pid}.png'))
                plt.close()

        except Exception as e:
            rsp_filter_issues.append({
                "participant": pid,
                "error": str(e)
            })

    # (2) physiology 처리
    process_rsp_only_all_channels(
    data_root=rsp_raw_dir,
    output_dir=output_dir,
    log_dir=log_dir,
    selected_participants=[pid],
    figure=False
)

    gc.collect()
    memory_debug("After")

# ======= 로그 저장 =======
if rsp_filter_issues:
    os.makedirs(log_dir, exist_ok=True)
    pd.DataFrame(rsp_filter_issues).to_csv(log_path, index=False, encoding='utf-8-sig')
    print(f"⚠️ Channel 2 필터 이슈 {len(rsp_filter_issues)}건 → {log_path} 저장됨")
else:
    print("✅ 모든 participant의 Channel 2에서 이상 없음")

print("✅ Sub-Batch complete.\n")
