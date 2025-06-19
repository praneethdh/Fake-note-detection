import cv2
import os
import numpy as np
import re

def load_real_notes(folder):
    references = {}
    for file in os.listdir(folder):
        if file.lower().endswith(('.jpg', '.jpeg', '.png')):
            label = os.path.splitext(file)[0].strip()
            img = cv2.imread(os.path.join(folder, file), 0)
            if img is not None:
                references[label] = img
    return references

def get_orb_match_score(img1, img2):
    orb = cv2.ORB_create(nfeatures=1000)
    kp1, des1 = orb.detectAndCompute(cv2.resize(img1, (800, 400)), None)
    kp2, des2 = orb.detectAndCompute(cv2.resize(img2, (800, 400)), None)

    if des1 is None or des2 is None:
        return 0, 0

    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
    matches = bf.match(des1, des2)
    good_matches = [m for m in matches if m.distance < 40]

    match_ratio = len(good_matches) / max(len(kp1), 1) * 100
    return len(good_matches), match_ratio

def label_image(img, text):
    if img is None:
        img = np.zeros((250, 400, 3), dtype=np.uint8)
    labeled = cv2.resize(img, (400, 250))
    cv2.rectangle(labeled, (0, 0), (400, 30), (255, 255, 255), -1)
    cv2.putText(labeled, text, (10, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
    return labeled

def extract_note_value(filename):
    match = re.search(r'(\d+)', filename)
    return match.group(1) if match else None

def detect_and_display(real_folder="real_notes", test_folder="test_notes", min_match_percent=90):
    real_notes = load_real_notes(real_folder)
    if not real_notes:
        print("No real notes found in", real_folder)
        return

    for file in os.listdir(test_folder):
        if not file.lower().endswith(('.jpg', '.jpeg', '.png')):
            continue

        test_path = os.path.join(test_folder, file)
        test_gray = cv2.imread(test_path, 0)
        test_color = cv2.imread(test_path)

        if test_gray is None or test_color is None:
            print(f"Could not read {file}")
            continue

        test_value = extract_note_value(file)
        if test_value is None:
            print(f"Skipping {file}: could not extract denomination")
            continue

        reference_candidates = {k: v for k, v in real_notes.items() if extract_note_value(k) == test_value}
        if not reference_candidates:
            print(f"No reference image found for Rs.{test_value}")
            continue

        match_scores = {}
        match_ratios = {}
        for label, real_img in reference_candidates.items():
            score, ratio = get_orb_match_score(test_gray, real_img)
            match_scores[label] = score
            match_ratios[label] = ratio

        best_label = max(match_scores, key=match_scores.get)
        best_score = match_scores[best_label]
        best_ratio = match_ratios[best_label]

        is_real = best_ratio >= min_match_percent
        result_text = "Real" if is_real else "Fake"

        ref_img_path = os.path.join(real_folder, f"{best_label}.jpg")
        ref_img_color = cv2.imread(ref_img_path)

        labeled_ref = label_image(ref_img_color, f"Real Note: Rs.{test_value}")
        labeled_test = label_image(test_color, f"Result: {result_text}")

        combined = cv2.hconcat([labeled_ref, labeled_test])
        cv2.imshow(f"Checking: {file}", combined)
        print(f"{file} → Best Score: {best_score}, Match: {best_ratio:.2f}% → {result_text}")

        cv2.waitKey(0)
        cv2.destroyAllWindows()

if __name__ == "__main__":
    detect_and_display()
