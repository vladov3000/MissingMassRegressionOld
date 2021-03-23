import torch


def all_loss_fn(out, info, loss_fn=F.mse_loss):
    """ 
    Computes loss for Higgs and neutrinos' masses.

    out consists of 
        Na_Genx/y/z, Nb_Genz

    info consists of 
        La_Visx/y/z, Lb_Visx/y/z, MET_X_Vis, MET_Y_Vis,
        Na_Genx/y/z, Nb_Genx/y/z, Wa_Genx/y/z,  Wb_Genx/y/z, H_Genx/y/z,
        La_VisE, Lb_VisE, Na_GenE, Nb_GenE, Wa_GenE, Wb_GenE, H_GenE,
        Wa_Genm_squared, Wb_Genm_squared, H_Genm_squared
    """
    # unpack variables
    La_Visp = info[:, 0:3]
    Lb_Visp = info[:, 3:6]
    MET_X_Vis = info[:, 6]
    MET_Y_Vis = info[:, 7]

    Na_Genp = info[:, 7:10]
    Nb_Genp = info[:, 10:13]
    Wa_Genp = info[:, 13:16]
    Wb_Genp = info[:, 16:19]
    H_Genp = info[:, 19:21]
    La_VisE = info[:, 21]
    Lb_VisE = info[:, 22]
    Na_GenE = info[:, 23]
    Nb_GenE = info[:, 24]
    Wa_GenE = info[:, 25]
    Wb_GenE = info[:, 26]
    H_GenE = info[:, 27]
    Wa_Genm2 = info[:, 28]
    Wb_Genm2 = info[:, 29]
    H_Genm2 = info[:, 30]

    # Compute momentum
    Na_p = out[:, 0:3]

    Nb_p = torch.zeros_like(Na_p)
    Nb_p[:, 2] = out[:, 3]
    Nb_p[:, 1] = MET_Y_Vis - Na_p[:, 1]
    Nb_p[:, 0] = MET_X_Vis - Na_p[:, 0]

    Wa_p = La_Visp + Na_p
    Wb_p = Lb_Visp + Nb_p
    H_p = Wa_p + Wb_p

    # Compute energy
    Na_E = norm(Na_p)
    Nb_E = norm(Nb_p)

    Wa_E = La_VisE + Na_E
    Wb_E = Lb_VisE + Nb_E
    H_E = Wa_E + Wb_E

    # Compute squared mass
    Wa_m2 = Wa_E**2 - square_norm(Wa_p)
    Wb_m2 = Wa_E**2 - square_norm(Wb_p)
    H_m2 = H_E**2 - square_norm(H_p)

    return [
        loss_fn(Na_p, Na_Genp),
        loss_fn(Nb_p, Nb_Genp),
        loss_fn(Wa_p, Wa_Genp),
        loss_fn(Wb_p, Wb_Genp),
        loss_fn(H_p, H_Genp),
        loss_fn(Na_E, Na_GenE),
        loss_fn(Nb_E, Nb_GenE),
        loss_fn(Wa_E, Wa_GenE),
        loss_fn(Wb_E, Wb_GenE),
        loss_fn(H_E, H_GenE),
        loss_fn(Wa_m2, Wa_Genm2),
        loss_fn(Wb_m2, Wb_Genm2),
        loss_fn(H_m2, H_Genm2),
    ]


def norm(x):
    """ x has shape (?, 3). Returns shape (?, 1) """
    return (x[:, 0]**2 + x[:, 1]**2 + x[:, 2]**2)**0.5


def square_norm(x):
    """ x has shape (?, 3). Returns shape (?, 1) """
    return x[:, 0]**2 + x[:, 1]**2 + x[:, 2]**2
