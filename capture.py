import cv2, os, sys
label = sys.argv[1]
n = int(sys.argv[2]) if len(sys.argv) > 2 else 300
session = sys.argv[3] if len(sys.argv) > 3 else "s1"
folder = f"data/raw/{label}"
os.makedirs(folder, exist_ok=True)
count = len(os.listdir(folder))
target = count + n
saving, tick = False, 0
cap = cv2.VideoCapture(0)
while count < target:
    ok, frame = cap.read()
    if not ok:
        break
    show = frame.copy()
    cv2.putText(show, f"{label}: {count}/{target} | SPACE chup | Q thoat",
                (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
    cv2.imshow("capture", show)
    k = cv2.waitKey(1) & 0xFF
    if k == ord(" "):
        saving = not saving
    if k == ord("q"):
        break
    tick += 1
    if saving and tick % 3 == 0:
        cv2.imwrite(f"{folder}/{label}_{session}_{count:04d}.jpg", frame)
        count += 1
cap.release()
cv2.destroyAllWindows()