import torch
import torch.nn as nn
import torch.nn.functional as F
# torch.autograd.set_detect_anomaly(True)  # 注释掉这行，因为不是所有PyTorch版本都支持

class GlobalMaskCalculator:
    def __init__(self, args, device):
        self.args = args
        self.device = device

    def compute_global_statistics(self, train_loader):
        num_channels = self.args.enc_in
        freq_length = self.args.seq_len // 2 + 1  # FFT后的频率长度

        amplitude_sum = torch.zeros(freq_length, num_channels, dtype=torch.float32).to(self.device)
        amplitude_squared_sum = torch.zeros(freq_length, num_channels, dtype=torch.float32).to(self.device)
        count = 0

        with torch.no_grad():
            for data in train_loader:
                #print("data:",data.shape)
                lookback_window = data[0].float().to(self.device)
                #print("lookback:",lookback_window.shape) #[batch, length, channel]
                xf = torch.fft.rfft(lookback_window, dim=1)
                amplitude = torch.abs(xf)

                amplitude_sum += amplitude.sum(dim=0)
                amplitude_squared_sum += (amplitude ** 2).sum(dim=0)
                count += lookback_window.size(0)

        mean_amplitude = amplitude_sum / count
        mean_amplitude_squared = amplitude_squared_sum / count
        variance_amplitude = mean_amplitude_squared - (mean_amplitude ** 2)
        std_amplitude = torch.sqrt(variance_amplitude + 1e-5)

        global_mask_amp = mean_amplitude / (std_amplitude + 1e-5)

        return global_mask_amp

def run_filter(args, train_loader):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    calculator = GlobalMaskCalculator(args, device)
    global_mask_amp = calculator.compute_global_statistics(train_loader)
    return global_mask_amp

class FrequencyDomainFilter(nn.Module):
    def __init__(self, args, global_mask_amp, hidden_dim=64, dropout_rate=0.3):
        super(FrequencyDomainFilter, self).__init__()
        self.args = args
        num_channels = args.enc_in
        self.output_length = args.pred_len
        self.mask = global_mask_amp
        self.hidden_dim = hidden_dim
        self.dropout_rate = dropout_rate
        
        # Global filter linear layers with L2 regularization and Dropout
        self.linear_r = nn.Sequential(
            nn.Linear(global_mask_amp.size(0), hidden_dim),  # bottleneck structure
            nn.RReLU(),
            nn.Dropout(p=dropout_rate),
            nn.Linear(hidden_dim, global_mask_amp.size(0)),
        )
        
        self.linear_i = nn.Sequential(
            nn.Linear(global_mask_amp.size(0), hidden_dim),  # bottleneck structure
            nn.RReLU(),
            nn.Dropout(p=dropout_rate),
            nn.Linear(hidden_dim, global_mask_amp.size(0)),
        )

    def apply_global_filter(self, data):
        data_fft = torch.fft.rfft(data, dim=1)
        mask_r = self.linear_r(self.mask.permute(1, 0)).permute(1, 0)
        mask_i = self.linear_i(self.mask.permute(1, 0)).permute(1, 0)
        masked_fft_real = data_fft.real * mask_r
        masked_fft_imag = data_fft.imag * mask_i
        masked_fft = torch.complex(masked_fft_real, masked_fft_imag)
        filtered_data = torch.fft.irfft(masked_fft, dim=1, n=data.size(1))#.real
        return filtered_data

    def forward(self, batch_x):
        x = self.apply_global_filter(batch_x)
        return x
