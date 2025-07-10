# CycleGAN for Satellite Smoke Detection and Image Enhancement

This project applies CycleGAN to detect smoke emissions from thermal power plants in satellite images. The goal is to enhance the resolution of low-quality images and improve visibility of smoke regions using advanced GAN-based techniques.

## Project Objective

- Enhance satellite images using CycleGAN.
- Improve resolution of input images using SRGAN.
- Detect smoke emissions from power plants using CV techniques.
- Support environmental monitoring and pollution analysis.

## Technologies Used

- Python (Jupyter Notebook)
- CycleGAN (Image-to-Image Translation)
- SRGAN (Super-Resolution)
- OpenCV
- TensorFlow / PyTorch
- Google Colab
- Satellite Imagery from Sentinel-2

## Dataset

- Source: Sentinel Hub EO Browser
- States covered: Maharashtra, Odisha, Uttar Pradesh, Madhya Pradesh, Telangana, Andhra Pradesh
- Format: RGB satellite images (before and after smoke)

## Workflow

1. **Data Collection:** Satellite images extracted from Sentinel Hub.
2. **Preprocessing:** Normalization, resizing, and augmentation.
3. **Image Enhancement:** CycleGAN for style transfer; SRGAN for super-resolution.
4. **Smoke Detection:** Thresholding and contour detection to highlight emissions.
5. **Evaluation Metrics:** PSNR, SSIM, and LPIPS used for assessing image quality.

## Results

- Image enhancement resulted in better visibility of smoke plumes.
- Quantitative improvements in SSIM and PSNR scores.
- Visual improvement in smoke region identification.

## Repository Structure

```
Cycle_GAN_Smoke_Detection/
├── Cycle_GAN_.ipynb          # Main Notebook
├── dataset/                  # Raw and processed satellite images
├── results/                  # Output samples after GAN and detection
├── models/                   # CycleGAN and SRGAN architectures
└── README.md
```

## How to Run

1. Clone the repository:
   ```bash
   git clone https://github.com/yourusername/CycleGAN-Smoke-Detection.git
   cd CycleGAN-Smoke-Detection
   ```
2. Open `Cycle_GAN_.ipynb` in Google Colab.
3. Upload your dataset or link it via Google Drive.
4. Run all cells sequentially.

## Future Scope

- Integrate real-time smoke alert system using satellite API.
- Apply attention-based GANs for better localization.
- Automate smoke region segmentation using deep learning models.

## 🤝 Acknowledgments

- Sentinel Hub for EO data access.
- PyTorch/TensorFlow GAN implementations.
- Academic inspiration from image-to-image translation and environmental monitoring literature.
