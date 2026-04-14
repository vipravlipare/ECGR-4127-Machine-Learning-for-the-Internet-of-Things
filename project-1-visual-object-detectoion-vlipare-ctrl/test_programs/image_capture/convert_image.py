import re
import os

# -------- SETTINGS --------
INPUT_FILE = r"C:\Github_School\project-1-visual-object-detectoion-vlipare-ctrl\imageTxtFiles\NoShoeSession5.txt"

LABEL_PREFIX = "NoShoe"   # or "Shoe"

if LABEL_PREFIX == "Shoe":
    OUTPUT_DIR = r"C:\Github_School\project-1-visual-object-detectoion-vlipare-ctrl\dataset\ShoeImages"
else:
    OUTPUT_DIR = r"C:\Github_School\project-1-visual-object-detectoion-vlipare-ctrl\dataset\NoShoeImages"

FILE_ENCODING = "utf-16-le"
# --------------------------


def get_next_index(output_dir, prefix):
    os.makedirs(output_dir, exist_ok=True)

    pattern = re.compile(rf"^{re.escape(prefix)}(\d+)\.jpg$", re.IGNORECASE)
    max_index = 0

    for filename in os.listdir(output_dir):
        match = pattern.match(filename)
        if match:
            num = int(match.group(1))
            if num > max_index:
                max_index = num

    return max_index + 1


def extract_all_image_blocks(filename):
    with open(filename, "r", encoding=FILE_ENCODING, errors="ignore") as f:
        text = f.read()

    blocks = re.findall(r"IMAGE_START(.*?)IMAGE_END", text, re.DOTALL)

    images = []
    for block in blocks:
        numbers = re.findall(r"\d+", block)
        if not numbers:
            continue
        img_bytes = bytes(int(n) for n in numbers)
        images.append(img_bytes)

    return images


def convert_one_file():
    if not os.path.exists(INPUT_FILE):
        print(f"File not found: {INPUT_FILE}")
        return

    next_index = get_next_index(OUTPUT_DIR, LABEL_PREFIX)
    images = extract_all_image_blocks(INPUT_FILE)

    if not images:
        print("No images found in file.")
        return

    for img_bytes in images:
        output_name = f"{LABEL_PREFIX}{next_index}.jpg"
        output_path = os.path.join(OUTPUT_DIR, output_name)

        with open(output_path, "wb") as f:
            f.write(img_bytes)

        print(f"Saved {output_path}")
        next_index += 1

    print(f"\nDone. Saved {len(images)} image(s) from:")
    print(INPUT_FILE)


if __name__ == "__main__":
    convert_one_file()