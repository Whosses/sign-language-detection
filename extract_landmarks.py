import cv2, os, mediapipe as mp, pandas as pd

hands = mp.solutions.hands.Hands(static_image_mode=True, max_num_hands=1,
                                 min_detection_confidence=0.5)
rows, failed = [], []
for label in sorted(os.listdir("data/raw")):
    for f in os.listdir(f"data/raw/{label}"):
        img = cv2.imread(f"data/raw/{label}/{f}")
        if img is None:
            failed.append((label, f, "read_error")); continue
        res = hands.process(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
        if not res.multi_hand_landmarks:
            failed.append((label, f, "no_hand")); continue
        lm = res.multi_hand_landmarks[0].landmark
        rows.append([v for p in lm for v in (p.x, p.y, p.z)] + [label])

cols = [f"{a}{i}" for i in range(1, 22) for a in "xyz"] + ["label"]
pd.DataFrame(rows, columns=cols).to_csv("dataset_raw.csv", index=False)
pd.DataFrame(failed, columns=["label", "file", "reason"]).to_csv("failed_log.csv", index=False)
print(f"OK: {len(rows)} | loi: {len(failed)}")