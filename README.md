# Retrolens - Hand Tracking Filter & Portal

Retrolens adalah aplikasi kamera real-time berbasis Python yang menggunakan **YOLO Hand Pose Detection** untuk mendeteksi tangan dan membuat efek filter interaktif berdasarkan posisi jari.

Aplikasi ini dibuat untuk menghasilkan efek **portal/filter berbasis hand tracking** tanpa menggunakan MediaPipe sebagai runtime.

---

## ✨ Features

- Real-time hand tracking
- Deteksi hingga 2 tangan
- Deteksi 21 keypoints pada tangan
- Hand gesture untuk mengganti filter
- Portal effect menggunakan posisi jari
- Berbagai macam visual filter
- Particle effect pada portal
- Image segmentation untuk efek Galaxy
- Temporal tracking agar pergerakan tangan lebih stabil
- Optimasi untuk CPU
- Tidak membutuhkan GPU NVIDIA
- Tidak menggunakan MediaPipe runtime

---

## 🎨 Available Filters

Retrolens memiliki beberapa filter yang dapat digunakan:

| Filter | Description |
|---|---|
| MONO | Mengubah gambar menjadi hitam putih |
| DUAL-TONE | Efek dua warna |
| PIXELATE | Efek pixel / mosaic |
| INVERT | Membalik warna gambar |
| SEPIA | Efek warna vintage |
| BLUR | Efek blur |
| THERMAL | Efek thermal / heatmap |
| SKETCH | Efek seperti gambar sketsa |
| GLITCH | Efek glitch digital |
| NEON | Efek neon |
| GALAXY | Efek galaxy dengan segmentation |

---

## 🖐️ Hand Tracking

Model hand pose mendeteksi **21 titik pada tangan**.

Struktur keypoint:

```text
0  = Wrist

Thumb
1  = Thumb CMC
2  = Thumb MCP
3  = Thumb IP
4  = Thumb Tip

Index
5  = Index MCP
6  = Index PIP
7  = Index DIP
8  = Index Tip

Middle
9  = Middle MCP
10 = Middle PIP
11 = Middle DIP
12 = Middle Tip

Ring
13 = Ring MCP
14 = Ring PIP
15 = Ring DIP
16 = Ring Tip

Pinky
17 = Pinky MCP
18 = Pinky PIP
19 = Pinky DIP
20 = Pinky Tip
```

Keypoints yang paling penting untuk aplikasi ini:

```text
4  = Thumb Tip
8  = Index Tip
20 = Pinky Tip
```

---

## 🌀 Portal Effect

Portal membutuhkan **2 tangan**.

Aplikasi mengambil:

```text
Thumb Tip
Index Tip
```

dari masing-masing tangan.

Dengan 2 tangan akan didapatkan 4 titik:

```text
Hand 1
├── Thumb Tip
└── Index Tip

Hand 2
├── Thumb Tip
└── Index Tip
```

Keempat titik tersebut digunakan untuk membentuk area portal.

Filter yang sedang aktif kemudian diterapkan ke area di dalam portal.

Portal juga memiliki:

- Border
- Particle effect
- Filter effect
- Portal label

---

## ☝️ Gesture Control

Untuk mengganti filter, cukup menggunakan **1 tangan**.

Gesture yang digunakan adalah mendekatkan:

```text
Thumb Tip
+
Pinky Tip
```

Ketika jaraknya cukup dekat, filter akan berpindah ke filter berikutnya.

Urutan filter:

```text
MONO
↓
DUAL-TONE
↓
PIXELATE
↓
INVERT
↓
SEPIA
↓
BLUR
↓
THERMAL
↓
SKETCH
↓
GLITCH
↓
NEON
↓
GALAXY
↓
MONO
```

Gesture menggunakan 1 tangan sehingga tidak perlu membuka 2 tangan hanya untuk mengganti filter.

---

## 🖥️ Requirements

Recommended:

- Python 3.10+
- Linux / Windows
- Webcam
- CPU
- RAM minimal 4 GB
- Internet untuk menginstall dependencies dan mengunduh model

GPU NVIDIA **tidak diperlukan**.

Retrolens dapat dijalankan menggunakan CPU.

---

## 📦 Installation

Clone repository:

```bash
git clone https://github.com/GedeAnanda/python-handtrack.git
cd python-handtrack
```

Buat virtual environment:

```bash
python3 -m venv venv
```

Aktifkan virtual environment:

### Linux / macOS

```bash
source venv/bin/activate
```

### Windows

```powershell
venv\Scripts\activate
```

---

## 🔥 Install PyTorch CPU

Retrolens dapat menggunakan PyTorch CPU-only.

Install:

```bash
python -m pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu
```

---

## 📚 Install Dependencies

Install dependency lainnya:

```bash
python -m pip install -r requirements.txt
```

Isi `requirements.txt`:

```text
ultralytics
opencv-python
numpy
requests
```

---

## 🧠 Model Files

Retrolens membutuhkan beberapa model.

Struktur file model:

```text
python-handtrack/
├── hand_pose.pt
├── segmentation_model.pt
├── main.py
├── requirements.txt
└── README.md
```

### Hand Pose Model

File:

```text
hand_pose.pt
```

Model ini digunakan untuk mendeteksi tangan dan 21 keypoints.

Model menghasilkan koordinat:

```text
x1
y1
x2
y2
confidence
class
21 hand keypoints
```

Keypoint tersebut digunakan untuk:

- Hand tracking
- Gesture detection
- Portal generation

### Segmentation Model

File:

```text
segmentation_model.pt
```

Model segmentation digunakan terutama untuk filter:

```text
GALAXY
```

Model segmentation membantu memisahkan foreground dari background.

---

## ▶️ Run

Pastikan virtual environment sudah aktif:

```bash
source venv/bin/activate
```

Kemudian jalankan:

```bash
python main.py
```

Jika webcam tersedia, aplikasi akan membuka kamera secara otomatis.

---

## 🎮 Controls

Saat aplikasi berjalan:

| Input | Action |
|---|---|
| Thumb + Pinky | Ganti filter |
| 2 Hands | Aktifkan portal |
| Q | Keluar dari aplikasi |

---

## ⚙️ Configuration

Beberapa konfigurasi dapat diubah langsung di `main.py`.

Contoh:

```python
CAMERA_WIDTH = 640
CAMERA_HEIGHT = 480

HAND_IMGSZ = 416
SEG_IMGSZ = 256

HAND_CONF = 0.01
KEYPOINT_CONF = 0.03
SEG_CONF = 0.25

HAND_MEMORY_FRAMES = 36
GESTURE_DISTANCE = 60

SMOOTHING_FACTOR = 0.30
```

### Camera Resolution

```python
CAMERA_WIDTH = 640
CAMERA_HEIGHT = 480
```

Resolusi yang lebih tinggi dapat meningkatkan kualitas gambar tetapi juga meningkatkan beban CPU.

Untuk CPU yang lebih lemah, gunakan:

```python
CAMERA_WIDTH = 640
CAMERA_HEIGHT = 480
```

---

## 🧠 Hand Detection Resolution

```python
HAND_IMGSZ = 416
```

Nilai ini menentukan ukuran input model hand detection.

Semakin tinggi nilainya:

- Detail tangan lebih baik
- Potensi deteksi lebih stabil
- CPU usage meningkat
- FPS dapat menurun

Contoh:

```text
320 = lebih ringan
416 = balance
640 = lebih berat
```

Untuk CPU-only, `416` merupakan pilihan yang cukup seimbang.

---

## 🎯 Detection Confidence

```python
HAND_CONF = 0.01
```

Nilai confidence menentukan seberapa rendah confidence detection yang masih diterima.

Nilai terlalu tinggi dapat menyebabkan tangan lebih sering hilang.

Nilai terlalu rendah dapat menghasilkan false detection.

Jika detection terlalu sering hilang, nilai dapat diturunkan.

Jika terlalu banyak objek salah terdeteksi, nilai dapat dinaikkan.

---

## 🌀 Temporal Hand Tracking

Retrolens menggunakan memory terhadap hasil deteksi sebelumnya.

```python
HAND_MEMORY_FRAMES = 36
```

Tujuannya agar tracking tidak langsung hilang ketika model gagal mendeteksi tangan selama beberapa frame.

Dengan pendekatan ini, pergerakan tangan dapat terasa lebih stabil dibandingkan hanya mengandalkan hasil YOLO pada setiap frame.

---

## 🧮 Gesture Distance

Gesture menggunakan jarak antara:

```text
Thumb Tip
Pinky Tip
```

Konfigurasinya:

```python
GESTURE_DISTANCE = 60
```

Jika tangan berada terlalu jauh dari kamera, nilai ini mungkin perlu disesuaikan.

---

## 🖼️ Camera Orientation

Retrolens menggunakan vertical flip untuk memperbaiki orientasi kamera tanpa membuat tampilan menjadi mirror secara horizontal.

Bagian yang digunakan:

```python
frame = cv2.flip(frame, 0)
```

---

## 🚀 Performance

Retrolens dirancang agar dapat berjalan pada CPU.

Namun performa sangat bergantung pada:

- CPU
- Resolusi webcam
- Hand detection resolution
- Jumlah tangan
- Filter yang digunakan
- Segmentation
- Background complexity

Filter seperti `GALAXY` membutuhkan proses tambahan karena menggunakan segmentation.

Jika FPS rendah, beberapa hal yang dapat dicoba:

```python
CAMERA_WIDTH = 640
CAMERA_HEIGHT = 480
```

dan:

```python
HAND_IMGSZ = 320
```

Untuk CPU yang lebih kuat:

```python
HAND_IMGSZ = 416
```

dapat memberikan hasil deteksi yang lebih baik.

---

## 🧪 CPU Only

Retrolens tidak membutuhkan NVIDIA CUDA.

Untuk memastikan PyTorch berjalan tanpa CUDA:

```bash
python -c "import torch; print('Torch:', torch.__version__); print('CUDA available:', torch.cuda.is_available()); print('CUDA:', torch.version.cuda)"
```

Output yang diharapkan:

```text
CUDA available: False
CUDA: None
```

Hal tersebut normal untuk instalasi CPU-only.

---

## ⚠️ NNPACK Warning

Pada beberapa CPU, PyTorch dapat menampilkan warning seperti:

```text
NNPACK unsupported hardware
```

Warning tersebut bukan error aplikasi.

PyTorch hanya memberitahukan bahwa salah satu optimasi CPU NNPACK tidak tersedia pada hardware yang digunakan.

Program tetap dapat berjalan menggunakan CPU.

---

## 🛠️ Troubleshooting

### Camera tidak terbuka

Pastikan webcam terdeteksi:

```bash
ls /dev/video*
```

Kemudian coba jalankan kembali:

```bash
python main.py
```

Jika memiliki lebih dari satu kamera, ubah camera index pada `main.py`.

Contoh:

```python
cap = cv2.VideoCapture(0)
```

Menjadi:

```python
cap = cv2.VideoCapture(1)
```

---

### Hand tidak terdeteksi

Pastikan:

- Tangan berada di dalam frame
- Pencahayaan cukup
- Tangan tidak terlalu kecil
- Tidak terlalu banyak objek di background
- Model `hand_pose.pt` tersedia

Coba menggunakan:

```python
HAND_IMGSZ = 416
```

dan:

```python
HAND_CONF = 0.01
```

---

### FPS terlalu rendah

Turunkan:

```python
HAND_IMGSZ = 320
```

Gunakan resolusi kamera:

```python
CAMERA_WIDTH = 640
CAMERA_HEIGHT = 480
```

Filter `GALAXY` juga lebih berat dibandingkan filter lainnya karena menggunakan segmentation.

---

### Portal tidak muncul

Portal membutuhkan **2 tangan**.

Pastikan kedua tangan berhasil terdeteksi.

Setiap tangan harus memiliki:

```text
Thumb Tip
Index Tip
```

yang dapat dilacak.

---

### Gesture sulit mengganti filter

Gesture menggunakan:

```text
Thumb Tip
+
Pinky Tip
```

Coba dekatkan ibu jari dan kelingking.

Jika masih sulit, parameter:

```python
GESTURE_DISTANCE = 60
```

dapat dinaikkan.

Contoh:

```python
GESTURE_DISTANCE = 70
```

atau:

```python
GESTURE_DISTANCE = 80
```

---

## 📁 Project Structure

```text
python-handtrack/
│
├── main.py
├── requirements.txt
├── README.md
│
├── hand_pose.pt
└── segmentation_model.pt
```

---

## 🔄 How It Works

Secara sederhana, alur Retrolens:

```text
Webcam
   │
   ▼
OpenCV
   │
   ▼
YOLO Hand Pose
   │
   ├── Hand Detection
   │
   └── 21 Hand Keypoints
   │
   ▼
Hand Tracking
   │
   ├── Gesture Detection
   │       │
   │       ▼
   │   Change Filter
   │
   └── 2 Hand Detection
           │
           ▼
      Portal Generation
           │
           ▼
      Apply Filter
           │
           ▼
       Display Frame
```

---

## 🧩 Technologies

Retrolens menggunakan:

- Python
- OpenCV
- Ultralytics YOLO
- PyTorch
- NumPy
- Requests

MediaPipe tidak digunakan sebagai runtime aplikasi.

---

## 📌 Important Notes

Model `hand_pose.pt` merupakan model hand pose detection yang digunakan untuk menghasilkan 21 keypoints tangan.

Dataset hand keypoint yang digunakan untuk melatih model memiliki anotasi yang berasal dari pipeline hand landmark, tetapi aplikasi Retrolens sendiri menjalankan inference menggunakan model YOLO melalui Ultralytics dan tidak membutuhkan package MediaPipe saat runtime.

---

## 🔒 Privacy

Retrolens memproses kamera secara lokal pada komputer.

Frame kamera digunakan oleh aplikasi untuk melakukan:

- Hand detection
- Hand tracking
- Filter processing
- Portal rendering

Tidak ada kebutuhan untuk mengupload video kamera ke server eksternal agar fitur utama dapat berjalan.

---

## 📜 License

Project ini dibuat untuk tujuan pembelajaran, eksperimen computer vision, dan pengembangan efek kamera real-time.

Pastikan untuk memeriksa license dari model atau dependency pihak ketiga yang digunakan sebelum menggunakan project ini untuk tujuan komersial.

---

## 👨‍💻 Author

**GedeAnanda**

Retrolens - Hand Tracking Filter & Portal

Built with Python, OpenCV, Ultralytics YOLO, and PyTorch.
