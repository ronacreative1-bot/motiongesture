# Analisis Prototype Asli

## Arsitektur saat ini

Prototype awal adalah aplikasi statis satu file (`index.html`) yang menggabungkan:

- HTML UI,
- CSS,
- Canvas 2D particle renderer,
- akses webcam,
- MediaPipe Hand Landmarker,
- gesture logic,
- interaction loop.

Pendekatan ini sangat mudah dideploy ke GitHub Pages, tetapi makin sulit dipelihara ketika fitur bertambah.

## Yang sudah bagus

1. Tidak membutuhkan backend.
2. Mobile-first dan memakai `playsinline` untuk video.
3. Kamera hanya dibuka setelah aksi pengguna.
4. Ada fallback interaction lewat pointer/touch.
5. Particle field cukup ringan untuk prototype visual.
6. MediaPipe sudah menggunakan mode `VIDEO`.

## Masalah utama

### 1. Ini particle field, bukan fluid solver
Partikel menerima gaya lokal lalu digambar sebagai garis/trail. Tidak ada velocity grid, pressure projection, advection, divergence correction, atau dye field seperti solver Navier-Stokes.

### 2. Pinch tergantung ukuran layar
Versi awal mengubah jarak thumb-index ke piksel layar lalu membandingkan dengan konstanta `110`. Ambang gesture jadi tidak konsisten antara iPhone kecil, tablet, dan desktop.

### 3. Open-palm belum mengukur keterbukaan jari
Nilai `openScore` berasal dari jarak landmark 5 dan 17. Itu lebih dekat ke ukuran/lebar telapak daripada jumlah jari yang terbuka.

### 4. Tracking dan rendering berada di main thread
Inference hand landmark sinkron dapat berbenturan dengan render Canvas dan event UI.

### 5. Belum ada temporal smoothing
Koordinat landmark langsung dipakai untuk menghitung velocity. Noise kecil pada landmark dapat menghasilkan velocity spike.

### 6. Performance budget statis
Jumlah partikel tidak menyesuaikan kemampuan perangkat. Variabel `MAX` ada tetapi tidak menjadi mekanisme adaptasi nyata.

### 7. UI state belum sepenuhnya sinkron
Tombol dan status tidak selalu merefleksikan apakah kamera benar-benar aktif atau sudah dihentikan.

## Arah V2

V2 mempertahankan arsitektur tanpa build-step agar mudah di-host, tetapi memisahkan HTML/CSS/JS dan memperbaiki gesture serta performance budget.

### Gesture mapping

| Gesture | Aksi |
|---|---|
| Point / Move | flow mengikuti kecepatan telunjuk |
| Pinch | radius mengecil, kontrol lebih presisi |
| Open palm | radial burst/pulse |
| Fist | vortex + sedikit inward pull |
| Two hands | dua sumber interaksi independen |

### Performance

- Kamera diminta maksimal 30 FPS.
- Inference ditargetkan sekitar 24 FPS.
- Render tetap `requestAnimationFrame`.
- Jumlah partikel turun jika FPS rendah dan naik perlahan ketika performa longgar.
- Device pixel ratio dibatasi 2.

## Roadmap V3

Jika targetnya ingin benar-benar menyerupai aplikasi fluid profesional:

1. WebGL2/WebGPU grid-based fluid solver.
2. Velocity + pressure + divergence + curl textures.
3. Dye injection berdasarkan gesture.
4. Bloom post-processing GPU.
5. Web Worker/worker-friendly inference path.
6. PWA/offline shell dan model hosting lokal.
7. Gesture calibration per pengguna.
8. Quality presets berdasarkan GPU/device.
