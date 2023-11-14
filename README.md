# Driver-Identification-and-Verification-From-Smartphone-Accelerometers-Using-Deep-Neural-Networks

### Smartphone 데이터 비교실험 논문구현
### [2020 IEEE TITS] Driver Identification and Verification From Smartphone Accelerometers Using Deep Neural Networks
** IMU센서 데이터 => 2D transform => DNN => Labeling
1) Spectograms: Short-Time Fourier Transform using a DFT (1D to 2D)
2) Feature map extractor from a CNN (1D to 2D)

1),2)를 통해서 추출한 센서데이터에 대한 2D 이미지 데이터를 입력값으로 pre-trained ResNet-50 network에 적용하여 운전자 식별함(자세한 CNN architecture는 논문 참고)

#### 실험결과: https://docs.google.com/spreadsheets/d/1wCc0gsfeFkKPOzhDx_54-1VQd9EBFUNFCVxIKFvzKmo/edit?usp=sharing
