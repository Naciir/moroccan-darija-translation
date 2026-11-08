import torch
import torch.nn as nn

class VanillaLSTMCell(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int):
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim

        # Input-to-gate linear transformations
        self.W_xi = nn.Linear(input_dim, hidden_dim, bias=True)
        self.W_xf = nn.Linear(input_dim, hidden_dim, bias=True)
        self.W_xo = nn.Linear(input_dim, hidden_dim, bias=True)
        self.W_xg = nn.Linear(input_dim, hidden_dim, bias=True)

        # Hidden-to-gate linear transformations (bias=False to avoid duplicate bias terms)
        self.W_hi = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.W_hf = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.W_ho = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.W_hg = nn.Linear(hidden_dim, hidden_dim, bias=False)

    def forward(self, x_t: torch.Tensor, state: tuple[torch.Tensor, torch.Tensor]):
        h_prev, c_prev = state
        
        i_t = torch.sigmoid(self.W_xi(x_t) + self.W_hi(h_prev))
        f_t = torch.sigmoid(self.W_xf(x_t) + self.W_hf(h_prev))
        o_t = torch.sigmoid(self.W_xo(x_t) + self.W_ho(h_prev))
        g_t = torch.tanh(self.W_xg(x_t) + self.W_hg(h_prev))
        
        c_t = f_t * c_prev + i_t * g_t
        h_t = o_t * torch.tanh(c_t)
        
        return h_t, c_t
# --- Append to cells.py ---

class CIFGCell(nn.Module):
    """
    Coupled Input-Forget Gate LSTM (CIFG)
    Appendix Equations (13)-(18):
    f_t = sigmoid(W_hf * h_{t-1} + W_xf * x_t + b_f)
    i_t = 1 - f_t
    o_t = sigmoid(W_ho * h_{t-1} + W_xo * x_t + b_o)
    g_t = tanh(W_hg * h_{t-1} + W_xg * x_t + b_g)
    c_t = f_t * c_{t-1} + i_t * g_t
    h_t = o_t * tanh(c_t)
    """
    def __init__(self, input_dim: int, hidden_dim: int):
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim

        # Input-to-gate linear layers (no W_xi since i_t = 1 - f_t)
        self.W_xf = nn.Linear(input_dim, hidden_dim, bias=True)
        self.W_xo = nn.Linear(input_dim, hidden_dim, bias=True)
        self.W_xg = nn.Linear(input_dim, hidden_dim, bias=True)

        # Hidden-to-gate linear layers (no W_hi)
        self.W_hf = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.W_ho = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.W_hg = nn.Linear(hidden_dim, hidden_dim, bias=False)

    def forward(self, x_t: torch.Tensor, state: tuple[torch.Tensor, torch.Tensor]):
        h_prev, c_prev = state

        f_t = torch.sigmoid(self.W_xf(x_t) + self.W_hf(h_prev))
        i_t = 1.0 - f_t  # Coupled input gate
        o_t = torch.sigmoid(self.W_xo(x_t) + self.W_ho(h_prev))
        g_t = torch.tanh(self.W_xg(x_t) + self.W_hg(h_prev))

        c_t = f_t * c_prev + i_t * g_t
        h_t = o_t * torch.tanh(c_t)

        return h_t, c_t


class LSTMPCell(nn.Module):
    """
    LSTM with Recurrent Projection Layer (LSTMP)
    Appendix Equations (19)-(25):
    c_t, g_t, m_t in R^H
    h_t in R^P (P < H)
    W_p in R^{P x H} (Projection matrix)
    """
    def __init__(self, input_dim: int, hidden_dim: int, proj_dim: int = 128):
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim  # H
        self.proj_dim = proj_dim      # P

        # Input-to-gate linear layers: input_dim -> H
        self.W_xi = nn.Linear(input_dim, hidden_dim, bias=True)
        self.W_xf = nn.Linear(input_dim, hidden_dim, bias=True)
        self.W_xo = nn.Linear(input_dim, hidden_dim, bias=True)
        self.W_xg = nn.Linear(input_dim, hidden_dim, bias=True)

        # Hidden-to-gate linear layers: P -> H
        self.W_hi = nn.Linear(proj_dim, hidden_dim, bias=False)
        self.W_hf = nn.Linear(proj_dim, hidden_dim, bias=False)
        self.W_ho = nn.Linear(proj_dim, hidden_dim, bias=False)
        self.W_hg = nn.Linear(proj_dim, hidden_dim, bias=False)

        # Recurrent projection layer: H -> P
        self.W_p = nn.Linear(hidden_dim, proj_dim, bias=True)

    def forward(self, x_t: torch.Tensor, state: tuple[torch.Tensor, torch.Tensor]):
        h_prev, c_prev = state  # h_prev: (B, P), c_prev: (B, H)

        i_t = torch.sigmoid(self.W_xi(x_t) + self.W_hi(h_prev))
        f_t = torch.sigmoid(self.W_xf(x_t) + self.W_hf(h_prev))
        o_t = torch.sigmoid(self.W_xo(x_t) + self.W_ho(h_prev))
        g_t = torch.tanh(self.W_xg(x_t) + self.W_hg(h_prev))

        c_t = f_t * c_prev + i_t * g_t
        m_t = o_t * torch.tanh(c_t)
        h_t = self.W_p(m_t)  # Project down to R^P

        return h_t, c_t