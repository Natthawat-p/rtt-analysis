# 5G NR RTT Dataset Analysis

โปรเจกต์วิเคราะห์ชุดข้อมูล Round Trip Time (RTT) จากเปเปอร์ Mundlamuri et al. (2024)
เพื่อเป็นพื้นฐานสำหรับงานพัฒนาระบบ AoA Estimation บน 5G NR ของทีม

## 📋 ภาพรวม

โปรเจกต์นี้ port โค้ด MATLAB (`main.m`) ของผู้พัฒนา dataset มาเป็น Python pipeline ที่:
- อ่านไฟล์ Q15 binary ของ OpenAirInterface
- คำนวณ SNR, Range estimate, และ Coherent combining (PD/MF)
- สร้างกราฟวิเคราะห์ทั้ง dataset

**ผลลัพธ์ที่ verify ได้ตรงเปเปอร์:**
- SNR estimate: **25.6 dB** (เปเปอร์ระบุ 25 dB)
- Range estimate: **12.04 m** ที่ระยะจริง 10 m (เปเปอร์รายงาน 13.02 m)
- Sawtooth artifact: ค้นพบคาบ 14 frames = 1 slot ของ 5G NR

## 📚 อ้างอิงเปเปอร์

- **เปเปอร์หลัก**: Mundlamuri et al., "5G NR Positioning with OpenAirInterface: Tools and Methodologies" (arXiv:2407.20463v2)
- **Novel RTT scheme**: Mundlamuri et al., "Novel Round Trip Time Estimation in 5G NR" (arXiv:2404.19618)
- **Dataset**: https://hdl.handle.net/2047/D20666241

## 🗂 โครงสร้างโปรเจกต์

```
.
├── dataset_loader.py        # อ่าน Q15 binary, parse folder, fftshift
├── dataset_analysis.py      # SNR (Eq.10), range (Eq.11), PD/MF (Eq.12-13)
├── dataset_plots.py         # ฟังก์ชัน plot ทั้งหมด
├── explore_dataset.py       # Main driver: walk → summarise → inspect
│
├── check_drift.py           # Diagnostic: peak position over time
├── check_period.py          # Diagnostic: periodicity analysis (14-frame sawtooth)
├── diagnose_peak.py         # Diagnostic: peak location reference (N/2 = 768)
│
├── reference/
│   └── main.m               # MATLAB reference จากผู้พัฒนา dataset
│
├── docs/
│   └── RTT_AoA_Technical_Report_Formal.docx
│
├── README.md
├── .gitignore
└── requirements.txt
```

## 🚀 วิธีติดตั้ง

### 1. Clone repo

```bash
git clone <repo-url>
cd <repo-name>
```

### 2. สร้าง Python virtual environment

```bash
python3 -m venv .venv
source .venv/bin/activate          # macOS / Linux
```

### 3. ติดตั้ง dependencies

```bash
pip install -r requirements.txt
```

### 4. ดาวน์โหลด Dataset

Dataset ขนาดใหญ่ (~GB) ไม่ได้รวมใน repo — ดาวน์โหลดจาก:

```
https://hdl.handle.net/2047/D20666241
```

แตกไฟล์แล้วให้โครงสร้างเป็น:

```
./ssb_measurements/
├── scenario_ssb_10m_ue_att_0_20240405T133333/
│   ├── srs_chT.raw
│   ├── srs_chF.raw
│   ├── srs_chF_lin_interp.raw
│   └── srsrxdataF.raw
└── ... (24 folders รวม)
```

## 📊 วิธีใช้งาน

### รัน analysis pipeline แบบสมบูรณ์

```bash
python3 explore_dataset.py
```

ผลลัพธ์:
- ตารางสรุปทุก folder (SNR, range estimate)
- กราฟวิเคราะห์ folder ตัวอย่าง (`analysis_output/*.png`)
- JSON summary ของทั้ง dataset

### รัน diagnostic scripts

```bash
python3 diagnose_peak.py     # peak location
python3 check_drift.py        # clock drift
python3 check_period.py       # sawtooth period
```

### ใช้เป็น library

```python
from dataset_loader import load_measurement
from dataset_analysis import estimate_snr_db, range_peak_detector
from dataset_plots import plot_impulse_response

meas = load_measurement("./ssb_measurements/scenario_ssb_10m_ue_att_0_...")
snr = estimate_snr_db(meas.srsrxdataF)
print(f"SNR: {snr['snr_db']:.2f} dB")

distance = range_peak_detector(meas.srs_chT)
print(f"Range: {distance:.2f} m")

fig = plot_impulse_response(meas.srs_chT[:, 0])
fig.savefig("impulse.png")
```

## 🔑 พารามิเตอร์ระบบ (ตาม main.m)

| Parameter | Value |
|---|---|
| FFT size N | 1536 |
| Sampling rate fs | 46.08 MHz |
| Subcarrier spacing | 30 kHz |
| Centre frequency | 3.6192 GHz |
| PRB | 106 (1272 active subcarriers) |
| SRS comb size | 2 |
| Reference time-zero | array index N/2 = 768 |

## ⚠ ข้อควรระวัง

1. **Frame size**: ทุกไฟล์ `.raw` เป็น `(N=1536, n_frames)` ใน column-major (MATLAB-style)
2. **Q15 format**: int16 I + int16 Q interleaved → แปลงเป็น float ด้วย `/2**15`
3. **First 499 frames**: invalid → drop ตาม `main.m`
4. **fftshift**: ต้องทำกับ `srs_chF` และ `srsrxdataF` แต่**ไม่ทำ**กับ `srs_chT` และ `srs_chF_lin_interp`
5. **Reference index**: peak ที่ index N/2 = 768 หมายถึง delay = 0
6. **Range bias**: estimate มี bias 2-3 m จาก cable/processing delay

## 📈 ผลลัพธ์หลัก

### SNR vs TX Gain (linear region)
```
TX gain 89.5 dB → SNR ~25 dB
TX gain 79.5 dB → SNR ~18 dB    (-7 dB per -10 dB gain)
TX gain 69.5 dB → SNR ~8 dB
```

### Range Estimate ที่ 10 m (high SNR)
- Single frame: jitter 9-16 m
- After 3500 frames (PD): 12.04 m
- After 3500 frames (MF): 12.09 m

### ค้นพบเพิ่ม
- **Sawtooth pattern** ใน peak position มี period = 14 frames
- 14 frames = 1 slot ของ 5G NR ที่ 30 kHz SCS
- Long-term clock drift = 0 (ระบบ sync ดี)

## 🔮 งานในเฟสถัดไป

- [ ] Setup OAI 2-RX antenna บน USRP B210
- [ ] แก้ logger ใน srsRAN ดึง `mean_lse` แยกตาม RX port
- [ ] เก็บ AoA dataset ของเราเอง
- [ ] AoA estimator (Bartlett + MUSIC) — กำลังปรับปรุง

## 👥 ทีม

Internal research project — Private repo สำหรับทีมวิจัย
