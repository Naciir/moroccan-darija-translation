import torch
import torch.nn as nn

# Each cell computes its gate pre-activations with TWO fused matmuls instead of 8 (CIFG: 6) small
# ones:  W_x = [W_xi; W_xf; W_xo; W_xg]  (input_dim -> k*H, one bias per gate)
#        W_h = [W_hi; W_hf; W_ho; W_hg]  (rec_dim   -> k*H, bias=False to avoid duplicate bias terms)
# The sum is .chunk()-ed back into the named gates, so the equations below are exactly the
# appendix equations and the parameter counts are identical to the per-gate version.
# `n_gates` tells init_weights (models.py) to initialise each gate block on its own.

def _fused(in_dim, hidden_dim, k, bias):
    lin = nn.Linear(in_dim, k * hidden_dim, bias=bias)
    lin.n_gates = k
    return lin


class VanillaLSTMCell(nn.Module):
    """
    Vanilla LSTM
    i_t = sigmoid(W_xi x_t + W_hi h_{t-1} + b_i)
    f_t = sigmoid(W_xf x_t + W_hf h_{t-1} + b_f)
    o_t = sigmoid(W_xo x_t + W_ho h_{t-1} + b_o)
    g_t = tanh(W_xg x_t + W_hg h_{t-1} + b_g)
    c_t = f_t * c_{t-1} + i_t * g_t
    h_t = o_t * tanh(c_t)
    """
    def __init__(self, input_dim: int, hidden_dim: int):
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.W_x = _fused(input_dim, hidden_dim, 4, bias=True)    # [W_xi; W_xf; W_xo; W_xg]
        self.W_h = _fused(hidden_dim, hidden_dim, 4, bias=False)  # [W_hi; W_hf; W_ho; W_hg]

    def forward(self, x_t: torch.Tensor, state: tuple[torch.Tensor, torch.Tensor]):
        h_prev, c_prev = state
        z_i, z_f, z_o, z_g = (self.W_x(x_t) + self.W_h(h_prev)).chunk(4, dim=-1)

        i_t = torch.sigmoid(z_i)
        f_t = torch.sigmoid(z_f)
        o_t = torch.sigmoid(z_o)
        g_t = torch.tanh(z_g)

        c_t = f_t * c_prev + i_t * g_t
        h_t = o_t * torch.tanh(c_t)

        return h_t, c_t


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
        # no W_xi / W_hi since i_t = 1 - f_t  -> only 3 gate blocks
        self.W_x = _fused(input_dim, hidden_dim, 3, bias=True)    # [W_xf; W_xo; W_xg]
        self.W_h = _fused(hidden_dim, hidden_dim, 3, bias=False)  # [W_hf; W_ho; W_hg]

    def forward(self, x_t: torch.Tensor, state: tuple[torch.Tensor, torch.Tensor]):
        h_prev, c_prev = state
        z_f, z_o, z_g = (self.W_x(x_t) + self.W_h(h_prev)).chunk(3, dim=-1)

        f_t = torch.sigmoid(z_f)
        i_t = 1.0 - f_t  # Coupled input gate
        o_t = torch.sigmoid(z_o)
        g_t = torch.tanh(z_g)

        c_t = f_t * c_prev + i_t * g_t
        h_t = o_t * torch.tanh(c_t)

        return h_t, c_t


class LSTMPCell(nn.Module):
    """
    LSTM with Recurrent Projection Layer (LSTMP)
    Appendix Equations (19)-(25):
    i_t = sigmoid(W_xi x_t + W_hi h_{t-1} + b_i)
    f_t = sigmoid(W_xf x_t + W_hf h_{t-1} + b_f)
    o_t = sigmoid(W_xo x_t + W_ho h_{t-1} + b_o)
    g_t = tanh(W_xg x_t + W_hg h_{t-1} + b_g)
    c_t = f_t * c_{t-1} + i_t * g_t
    m_t = o_t * tanh(c_t)
    h_t = W_p m_t + b_p
    c_t, g_t, m_t in R^H ; h_t in R^P (P < H) ; W_p in R^{P x H} (projection matrix)
    """
    def __init__(self, input_dim: int, hidden_dim: int, proj_dim: int = 128):
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim  # H
        self.proj_dim = proj_dim      # P

        self.W_x = _fused(input_dim, hidden_dim, 4, bias=True)   # input_dim -> 4H
        self.W_h = _fused(proj_dim, hidden_dim, 4, bias=False)   # P -> 4H (recurrence on projected h)
        self.W_p = nn.Linear(hidden_dim, proj_dim, bias=True)    # recurrent projection H -> P

    def forward(self, x_t: torch.Tensor, state: tuple[torch.Tensor, torch.Tensor]):
        h_prev, c_prev = state  # h_prev: (B, P), c_prev: (B, H)
        z_i, z_f, z_o, z_g = (self.W_x(x_t) + self.W_h(h_prev)).chunk(4, dim=-1)

        i_t = torch.sigmoid(z_i)
        f_t = torch.sigmoid(z_f)
        o_t = torch.sigmoid(z_o)
        g_t = torch.tanh(z_g)

        c_t = f_t * c_prev + i_t * g_t
        m_t = o_t * torch.tanh(c_t)
        h_t = self.W_p(m_t)  # Project down to R^P

        return h_t, c_t


class WMCCell(nn.Module):
    """
    LSTM with Working Memory Connections (WMC, Landi et al. 2021)
    Appendix Equations (7)-(12): the gates also read the cell state through a squashed connection
    i_t = sigmoid(W_xi x_t + W_hi h_{t-1} + tanh(W_ci c_{t-1}) + b_i)
    f_t = sigmoid(W_xf x_t + W_hf h_{t-1} + tanh(W_cf c_{t-1}) + b_f)
    g_t = tanh(W_xg x_t + W_hg h_{t-1} + b_g)
    c_t = f_t * c_{t-1} + i_t * g_t
    o_t = sigmoid(W_xo x_t + W_ho h_{t-1} + tanh(W_co c_t) + b_o)      <- uses the NEW cell state
    h_t = o_t * tanh(c_t)
    """
    def __init__(self, input_dim: int, hidden_dim: int):
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.W_x = _fused(input_dim, hidden_dim, 4, bias=True)    # [W_xi; W_xf; W_xo; W_xg]
        self.W_h = _fused(hidden_dim, hidden_dim, 4, bias=False)  # [W_hi; W_hf; W_ho; W_hg]
        self.W_c = _fused(hidden_dim, hidden_dim, 2, bias=False)  # [W_ci; W_cf]  (read c_{t-1})
        self.W_co = nn.Linear(hidden_dim, hidden_dim, bias=False) # W_co          (reads c_t)

    def forward(self, x_t: torch.Tensor, state: tuple[torch.Tensor, torch.Tensor]):
        h_prev, c_prev = state
        z_i, z_f, z_o, z_g = (self.W_x(x_t) + self.W_h(h_prev)).chunk(4, dim=-1)
        w_i, w_f = torch.tanh(self.W_c(c_prev)).chunk(2, dim=-1)

        i_t = torch.sigmoid(z_i + w_i)
        f_t = torch.sigmoid(z_f + w_f)
        g_t = torch.tanh(z_g)

        c_t = f_t * c_prev + i_t * g_t
        o_t = torch.sigmoid(z_o + torch.tanh(self.W_co(c_t)))
        h_t = o_t * torch.tanh(c_t)

        return h_t, c_t


def convert_legacy_state_dict(sd):
    """Old checkpoints store W_xi, W_xf, ... per gate -> stack them (order i,f,o,g / CIFG f,o,g)
    into the fused W_x / W_h used above."""
    out, groups = {}, {}
    for k, v in sd.items():
        parts = k.rsplit(".", 2)    # e.g. ["encoder.rnn.cell", "W_xi", "weight"]
        if len(parts) == 3 and len(parts[1]) == 4 and parts[1][:3] in ("W_x", "W_h") and parts[1][3] in "ifog":
            groups.setdefault((parts[0], parts[1][:3], parts[2]), {})[parts[1][3]] = v
        else:
            out[k] = v
    for (prefix, w, p), g in groups.items():
        out[f"{prefix}.{w}.{p}"] = torch.cat([g[c] for c in "ifog" if c in g], 0)
    return out
