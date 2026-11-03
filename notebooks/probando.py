import torch
print("¿CUDA disponible en PyTorch?:", torch.cuda.is_available())
if torch.cuda.is_available():
    print("Dispositivo detectado:", torch.cuda.get_device_name(0))
else:
    print("ALERTA: PyTorch está corriendo en CPU.")