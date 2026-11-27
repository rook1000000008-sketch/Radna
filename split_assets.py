import os
import numpy as np
from PIL import Image
from scipy.ndimage import label, find_objects, binary_dilation

INPUT_IMAGE = "Radna_cooking.png"
OUTPUT_DIR = "Asset_Output"

# รายชื่อไฟล์เรียงตามลำดับจาก แถวบนลงล่าง -> ซ้ายไปขวา (รวม 19 ชิ้น)
FILE_NAMES = [
    # แถวที่ 1 (7 ชิ้น)
    "meat_quail_a.png",      # 1. ไข่นก 1 ฟอง
    "meat_quail.png",        # 2. ไข่นก 8 ฟอง
    "meat_liver.png",        # 3. ตับ
    "meat_beef.png",         # 4. เนื้อวัว
    "meat_crabstick.png",    # 5. ปูอัด
    "meat_crispy.png",       # 6. หมูกรอบชุด
    "meat_crispy_a.png",     # 7. หมูกรอบ 1 ชิ้น

    # แถวที่ 2 (5 ชิ้น)
    "meat_squid_a.png",      # 8. หมึก 1 ชิ้น
    "meat_squid.png",        # 9. หมึกชุด
    "meat_shrimp.png",       # 10. กุ้ง 3 ตัว
    "meat_seafood.png",      # 11. ทะเลรวม
    "meat_mixed.png",        # 12. รวมมิตร

    # แถวที่ 3 (5 ชิ้น)
    "meat_friedegg.png",     # 13. ไข่ดาว
    "veg_kale.png",          # 14. ผักคะน้า
    "meat_pork.png",         # 15. หมูหมัก
    "noodle_senmee.png",     # 16. เส้นหมี่
    "noodle_senyai.png",     # 17. เส้นใหญ่

    # แถวที่ 4 (2 ชิ้น)
    "bowl_empty.png",        # 18. ชามสเตนเลส
    "noodle_meekrob.png"     # 19. หมี่กรอบ
]

def split_sprites():
    if not os.path.exists(INPUT_IMAGE):
        print(f"❌ ไม่พบไฟล์ {INPUT_IMAGE} กรุณาตรวจสอบชื่อไฟล์")
        return

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    img = Image.open(INPUT_IMAGE).convert("RGBA")
    data = np.array(img, dtype=np.float32)

    # คำนวณความสว่างของแต่ละพิกเซล เพื่อแยกวัตถุออกจากพื้นหลังสีดำ
    brightness = np.max(data[:, :, :3], axis=2)
    is_not_transparent = data[:, :, 3] > 20
    
    # เจาะพื้นหลังสีดำให้โปร่งใส (Alpha = 0) และไล่ระดับขอบให้เนียน
    new_alpha = np.clip((brightness - 12) * 15, 0, 255)
    final_alpha = np.where(is_not_transparent, np.minimum(data[:, :, 3], new_alpha), 0)
    
    clean_data = data.astype(np.uint8)
    clean_data[:, :, 3] = final_alpha.astype(np.uint8)
    clean_img = Image.fromarray(clean_data, "RGBA")

    # สร้าง Mask ตรวจจับวัตถุ
    mask = final_alpha > 25

    # ขยายขอบ Mask 65px เพื่อรวมชิ้นที่อยู่กลุ่มเดียวกัน
    structure = np.ones((65, 65))
    dilated_mask = binary_dilation(mask, structure=structure)

    labeled_array, num_features = label(dilated_mask)
    slices = find_objects(labeled_array)

    boxes = []
    for sl in slices:
        y1, y2 = sl[0].start, sl[0].stop
        x1, x2 = sl[1].start, sl[1].stop
        if (x2 - x1) > 40 and (y2 - y1) > 40:
            boxes.append((x1, y1, x2, y2))

    # ใช้จุดกึ่งกลางแนวตั้ง (y_center) แบ่งแถวทีละ 550px แล้วเรียงจากซ้ายไปขวา
    boxes.sort(key=lambda b: (((b[1] + b[3]) // 2) // 550, b[0]))

    print(f"🔍 ตรวจพบและลบพื้นหลังดำสำเร็จทั้งหมด {len(boxes)} ชิ้น")

    for idx, (x1, y1, x2, y2) in enumerate(boxes):
        pad = 15
        crop_box = (
            max(0, x1 - pad),
            max(0, y1 - pad),
            min(clean_img.width, x2 + pad),
            min(clean_img.height, y2 + pad)
        )
        cropped = clean_img.crop(crop_box)

        filename = FILE_NAMES[idx] if idx < len(FILE_NAMES) else f"extra_{idx+1}.png"
        out_path = os.path.join(OUTPUT_DIR, filename)
        cropped.save(out_path, "PNG")
        print(f"✅ บันทึก (พื้นหลังโปร่งใส): {filename} ({cropped.width}x{cropped.height} px)")

if __name__ == "__main__":
    split_sprites()