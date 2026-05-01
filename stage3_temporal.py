import torch
import torch.nn as nn

# ==========================================
# 1. THE CONVLSTM CELL (Blueprint Sec 2.4)
# ==========================================
class ConvLSTMCell(nn.Module):
    def __init__(self, input_dim=256, hidden_dim=128, kernel_size=3):
        super(ConvLSTMCell, self).__init__()
        self.hidden_dim = hidden_dim
        padding = kernel_size // 2
        
        # We combine the 4 math gates (Input, Forget, Cell, Output) into ONE convolution 
        # for extreme efficiency. It outputs 4 * hidden_dim channels.
        self.conv = nn.Conv2d(in_channels=input_dim + hidden_dim,
                              out_channels=4 * hidden_dim,
                              kernel_size=kernel_size,
                              padding=padding)

    def forward(self, x_t, cur_state):
        h_cur, c_cur = cur_state
        
        # Concatenate the current input with the past memory
        combined = torch.cat([x_t, h_cur], dim=1) 
        combined_conv = self.conv(combined)
        
        # Split the convolution back into the 4 LSTM gates
        cc_i, cc_f, cc_o, cc_g = torch.split(combined_conv, self.hidden_dim, dim=1)
        
        i = torch.sigmoid(cc_i) # Input gate
        f = torch.sigmoid(cc_f) # Forget gate
        o = torch.sigmoid(cc_o) # Output gate
        g = torch.tanh(cc_g)    # Cell state update
        
        c_next = f * c_cur + i * g
        h_next = o * torch.tanh(c_next)
        
        return h_next, c_next

# ==========================================
# 2. THE 10-YEAR SEQUENCE LOOP
# ==========================================
class TemporalModule(nn.Module):
    def __init__(self, input_dim=256, hidden_dim=128):
        super(TemporalModule, self).__init__()
        self.hidden_dim = hidden_dim
        self.cell = ConvLSTMCell(input_dim, hidden_dim)

    def forward(self, x_seq):
        # x_seq shape: [Batch, Time, Channels, Height, Width]
        b, seq_len, c, h, w = x_seq.size()
        
        # Start with a blank memory for Year 1
        h_t = torch.zeros(b, self.hidden_dim, h, w, device=x_seq.device)
        c_t = torch.zeros(b, self.hidden_dim, h, w, device=x_seq.device)
        
        outputs =[]
        
        # Loop through Time (Year 1 to Year T)
        for t in range(seq_len):
            h_t, c_t = self.cell(x_seq[:, t, :, :, :], (h_t, c_t))
            outputs.append(h_t)
            
        # Stack the outputs back together along the Time dimension
        # Result shape: [Batch, Time, Hidden_Dim, Height, Width]
        outputs = torch.stack(outputs, dim=1)
        return outputs

# ==========================================
# 3. TEMPORAL SMOOTHNESS LOSS (Blueprint Sec 2.5)
# ==========================================
class TemporalLoss(nn.Module):
    def __init__(self):
        super(TemporalLoss, self).__init__()

    def forward(self, y_hat_seq, M_nochange):
        """
        y_hat_seq: Predictions across time[Batch, Time, 1, 512, 512]
        M_nochange: Binary SAR mask (1 = no construction, 0 = construction detected)
        """
        # We calculate the difference between Year 2 and Year 1, Year 3 and Year 2, etc.
        diff = y_hat_seq[:, 1:] - y_hat_seq[:, :-1]
        
        # Square the difference
        squared_diff = diff ** 2
        
        # Multiply by the SAR change mask.
        # If SAR says "No Change" (1), but the AI predicted a change, the AI is severely punished!
        # If SAR says "Construction" (0), the punishment is wiped out (0), allowing the prediction to change.
        l_temp = (squared_diff * M_nochange).sum(dim=1).mean()
        
        return l_temp

# --- LET'S TEST TIME TRAVEL! ---
if __name__ == "__main__":
    print("⏳ Initializing 2D ConvLSTM...")
    conv_lstm = TemporalModule(input_dim=256, hidden_dim=128)
    calc_temp_loss = TemporalLoss()
    
    # Simulate a "3-Year" sequence coming out of Stage 2 Cross-Attention
    #[Batch=1, Time=3, Channels=256, Height=64, Width=64]
    dummy_seq = torch.rand(1, 3, 256, 64, 64)
    
    print("\nPushing 3 years of data through ConvLSTM...")
    out_seq = conv_lstm(dummy_seq)
    
    print(f"✅ Output Sequence Shape: {out_seq.shape}")
    print("   -> (Expected: [1, 3, 128, 64, 64])")
    
    # ---------------------------------------------
    # Testing the SAR Temporal Loss
    # ---------------------------------------------
    # Simulate the final Decoded predictions[Batch=1, Time=3, Channels=1, Height=512, Width=512]
    dummy_preds = torch.rand(1, 3, 1, 512, 512)
    
    # Simulate the SAR "No Change" Mask for the transitions (Time-1 length)
    #[Batch=1, Time=2, Channels=1, Height=512, Width=512]
    dummy_sar_mask = torch.randint(0, 2, (1, 2, 1, 512, 512)).float()
    
    print("\nCalculating SAR-Masked Temporal Loss...")
    loss_val = calc_temp_loss(dummy_preds, dummy_sar_mask)
    print(f"✅ Temporal Loss value: {loss_val.item():.4f}")