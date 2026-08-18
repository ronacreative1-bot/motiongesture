# Fluid Camera Hand V2

Versi pengembangan dari prototype `fluid-camera-hand` untuk GitHub Pages.

## Perubahan utama

- Dua tangan (`numHands: 2`).
- Gesture classifier ringan berbasis landmark:
  - point/move = mengalirkan partikel,
  - pinch = mengecilkan radius interaksi,
  - open palm = pulse/burst,
  - fist = vortex.
- Smoothing posisi tangan agar gerakan lebih stabil.
- Inference MediaPipe dibatasi sekitar 24 FPS agar render tidak terlalu terganggu.
- Adaptive particle budget berdasarkan FPS.
- GPU delegate dengan fallback CPU.
- Preset fluid: Silk, Neon, Storm, Calm.
- Camera preview bisa disembunyikan.
- UI mobile-first dan safe-area friendly untuk iPhone.
- Tetap tanpa build step: upload folder ke GitHub Pages dan langsung jalan.

## Menjalankan lokal

```bash
python3 -m http.server 8080
```

Buka `http://localhost:8080` di komputer. Untuk kamera di perangkat lain, paling aman gunakan hosting HTTPS.

## Deploy ke GitHub Pages

1. Salin `index.html`, `styles.css`, dan `app.js` ke root repository/branch Pages.
2. Commit dan push.
3. Buka URL GitHub Pages melalui HTTPS.
4. Izinkan kamera ketika diminta.

## Catatan teknis

MediaPipe `detectForVideo()` berjalan sinkron pada browser. V2 membatasi frekuensi inference untuk menjaga frame rate. Tahap V3 yang lebih serius dapat memindahkan inference ke Web Worker dan mengganti particle field dengan GPU fluid solver (WebGL2/WebGPU).
