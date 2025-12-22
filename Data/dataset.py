import os
import torch
import glob
from torch.utils.data import Dataset
from tqdm import tqdm


class SlakhChunkedDataset(Dataset):
    def __init__(self, root_dir, file_list=None, sequence_length=128):
        """
        Args:
            root_dir (str): Verilerin bulunduğu klasör.
            file_list (list, optional): İşlenecek özel dosya listesi (.pt yolları).
                                      Eğer None verilirse, root_dir içindeki hepsini alır.
            sequence_length (int): Modelin zaman eksenindeki girdi boyutu.
        """
        self.root_dir = root_dir
        self.sequence_length = sequence_length
        self.hop_length = 512

        # --- MODIFICATION START ---
        # Eğer dışarıdan özel bir liste gelirse onu kullan, gelmezse klasörü tara
        if file_list is not None:
            self.file_paths = file_list
            print(f"📂 Özel dosya listesi kullanılıyor: {len(self.file_paths)} dosya.")
        else:
            self.file_paths = sorted(glob.glob(os.path.join(root_dir, "*.pt")))
            print(f"📂 Klasör tarandı: {len(self.file_paths)} dosya bulundu.")
        # --- MODIFICATION END ---

        # --- CHUNKING HARİTASI ---
        self.chunks = []

        # Eğer dosya listesi boşsa hata vermesin, sadece uyarsın
        if len(self.file_paths) == 0:
            print(f"⚠️ UYARI: '{root_dir}' konumunda hiç .pt dosyası bulunamadı!")
            return

        print(f"📊 Dataset İndeksleniyor... Lütfen bekleyin.")

        for path in tqdm(self.file_paths):
            try:
                # Sadece metadata/header okumak için map_location kullanıyoruz
                # Not: .pt dosyalarında tüm dosyayı okumadan shape almak zordur,
                # ancak bu işlem eğitim öncesi sadece 1 kez yapılır.
                data = torch.load(path, map_location="cpu")

                # Hedefin (Piano Roll) uzunluğunu al: [1, 88, Time] -> Time
                total_frames = data["target"].shape[-1]

                # Şarkıyı sequence_length boyutunda dilimlere böl
                # (Non-overlapping / Örtüşmesiz)
                for start_idx in range(0, total_frames, self.sequence_length):
                    self.chunks.append((path, start_idx))

            except Exception as e:
                print(f"⚠️ Dosya okunamadı veya bozuk: {path} - {e}")

        print(
            f"✅ İndeksleme Tamamlandı: {len(self.file_paths)} dosyadan toplam {len(self.chunks)} parça (chunk) oluşturuldu."
        )

    def __len__(self):
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
            chunk_target = target[:, :, start_frame:end_frame]

            # Waveform bazen target'tan frame hesaplaması yüzünden birkaç sample kısa kalabilir
            # Bu yüzden güvenli slicing yapıyoruz
            curr_wav_len = waveform.shape[1]
            if end_sample > curr_wav_len:
                # Eğer sample yetmiyorsa alabileceğimizi alalım, padding aşağıda halledecek
                chunk_waveform = waveform[:, start_sample:]
            else:
                chunk_waveform = waveform[:, start_sample:end_sample]

            # 5. Boyut Kontrolü ve Padding (Doldurma)
            current_frames = chunk_target.shape[2]
            current_samples = chunk_waveform.shape[1]

            # Target Padding (Sağ tarafa 0 ekle)
            if current_frames < req_frames:
                pad_amount = req_frames - current_frames
                chunk_target = torch.nn.functional.pad(chunk_target, (0, pad_amount))

            # Waveform Padding (Sağ tarafa 0 ekle)
            if current_samples < req_samples:
                pad_amount = req_samples - current_samples
                chunk_waveform = torch.nn.functional.pad(
                    chunk_waveform, (0, pad_amount)
                )

            return chunk_waveform, chunk_target

        except Exception as e:
            print(f"⚠️ Chunk yükleme hatası: {path} (Idx: {start_frame}) - {e}")
            # Hata durumunda boş tensor döndür (Batch'i patlatmamak için)
            return torch.zeros(1, req_samples), torch.zeros(1, 88, req_frames)
