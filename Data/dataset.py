import os
import torch
import glob
from torch.utils.data import Dataset
from tqdm import tqdm  # İlerleme çubuğu için


class SlakhChunkedDataset(Dataset):
    def __init__(self, root_dir, split="train", sequence_length=128):
        self.root_dir = root_dir
        self.sequence_length = sequence_length
        self.hop_length = 512

        # İşlenmiş .pt dosyalarını bul
        self.file_paths = sorted(glob.glob(os.path.join(root_dir, "*.pt")))

        # --- CHUNKING (Parçalama) HARİTASI ---
        # Her bir indeksin hangi dosyaya ve hangi başlangıç noktasına
        # denk geldiğini tutan liste: [(dosya_yolu, baslangic_frame), ...]
        self.chunks = []

        print(f"📊 Dataset İndeksleniyor ({split})... Lütfen bekleyin.")

        # Dosyaları tek tek açıp ne kadar uzun olduklarına bakmamız lazım
        # Bu işlem __init__ aşamasında biraz vakit alabilir ama eğitimde hız kazandırır.
        for path in tqdm(self.file_paths):
            try:
                # Sadece shape bilgisini almak için map_location kullanıyoruz
                # Not: PyTorch tam yükleme yapmadan header okumayı desteklemez,
                # bu yüzden dosyayı yüklüyoruz.
                data = torch.load(path, map_location="cpu")
                total_frames = data["target"].shape[-1]  # Örn: 2000 frame

                # Şarkıyı sequence_length (128) boyutunda dilimlere böl
                # step = sequence_length (Örtüşmesiz - Non-overlapping)
                # Eğer örtüşme (overlap) istersen step değerini düşürebilirsin (örn: 64)
                for start_idx in range(0, total_frames, self.sequence_length):
                    self.chunks.append((path, start_idx))

            except Exception as e:
                print(f"⚠️ Dosya okunamadı veya bozuk: {path} - {e}")

        print(
            f"✅ İndeksleme Tamamlandı: Toplam {len(self.chunks)} parça (chunk) oluşturuldu."
        )

    def __len__(self):
        # Artık dosya sayısı değil, toplam parça sayısı döndürüyoruz
        return len(self.chunks)

    def __getitem__(self, idx):
        # 1. Hangi dosya ve hangi başlangıç noktası olduğunu al
        path, start_frame = self.chunks[idx]

        # Hedeflenen Boyutlar
        req_frames = self.sequence_length
        req_samples = req_frames * self.hop_length

        try:
            # 2. Dosyayı yükle
            data = torch.load(path)
            waveform = data["waveform"].float()  # [1, Total_Samples]
            target = data["target"].float()  # [1, 88, Total_Frames]

            # 3. Kesme (Slicing) Koordinatlarını Hesapla
            end_frame = start_frame + req_frames

            start_sample = start_frame * self.hop_length
            end_sample = end_frame * self.hop_length

            # 4. Veriyi Kes
            # Not: Eğer end_frame, şarkının sonundan büyükse PyTorch hata vermez,
            # sadece alabildiği kadarını alır (kısa gelir).
            chunk_target = target[:, :, start_frame:end_frame]
            chunk_waveform = waveform[:, start_sample:end_sample]

            # 5. Boyut Kontrolü ve Padding (Doldurma)
            # Eğer şarkının son parçasıysa (kısa geldiyse), sonunu 0 ile doldur.
            current_frames = chunk_target.shape[2]
            current_samples = chunk_waveform.shape[1]

            if current_frames < req_frames:
                pad_amount = req_frames - current_frames
                chunk_target = torch.nn.functional.pad(chunk_target, (0, pad_amount))

            if current_samples < req_samples:
                pad_amount = req_samples - current_samples
                chunk_waveform = torch.nn.functional.pad(
                    chunk_waveform, (0, pad_amount)
                )

            return chunk_waveform, chunk_target

        except Exception as e:
            print(f"⚠️ Chunk yükleme hatası: {path} (Idx: {start_frame}) - {e}")
            # Hata durumunda boş tensor döndür
            return torch.zeros(1, req_samples), torch.zeros(1, 88, req_frames)
