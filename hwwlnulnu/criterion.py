import torch
import torch.nn.functional as F


def nu_loss_fn(out, target, inputs, loss_fn=F.mse_loss):
    """ 
    Computes loss just for neutrinos' momentums

    input consists of
        La_Visx/y/z, Lb_Visx/y/z, MET_X_Vis, MET_Y_Vis

    out consists of 
        Na_Genx/y/z, Nb_Genz

    target consists of 
        Na_Genx/y/z, Nb_Genz
    """
    return [loss_fn(out, target)]


def momentum_loss_fn(out, target, inputs, loss_fn=F.mse_loss):
    """ 
    Computes loss for Higgs, W bosons, and neutrinos' momentums.

    input consists of
        La_Visx/y/z, Lb_Visx/y/z, MET_X_Vis, MET_Y_Vis

    out consists of 
        Na_Genx/y/z, Nb_Genz

    target consists of 
        Na_Genx/y/z, Nb_Genx/y/z, Wa_Genx/y/z,  Wb_Genx/y/z, H_Genx/y/z
    """
    # momentum is represented as [px, py, pz] torch tensor

    # Unpack variables
    La_Visp = inputs[:, 0:3]
    Lb_Visp = inputs[:, 3:6]
    MET_X_Vis = inputs[:, 6]
    MET_Y_Vis = inputs[:, 7]

    Na_Genp = target[:, 0:3]
    Nb_Genp = target[:, 3:6]
    Wa_Genp = target[:, 6:9]
    Wb_Genp = target[:, 9:12]

    H_Genp = torch.zeros_like(Wb_Genp)
    H_Genp[:, 2] = target[:, 12]

    # Compute our outputs
    Na_p = out[:, 0:3]

    Nb_p = torch.zeros_like(Na_p)
    Nb_p[:, 2] = out[:, 3]
    Nb_p[:, 1] = MET_Y_Vis - Na_p[:, 1]
    Nb_p[:, 0] = MET_X_Vis - Na_p[:, 0]

    Wa_p = La_Visp + Na_p
    Wb_p = Lb_Visp + Nb_p
    H_p = Wa_p + Wb_p

    # Compare to target
    loss = loss_fn(Na_p, Na_Genp)
    loss += loss_fn(Nb_p, Nb_Genp)
    loss += loss_fn(Wa_p, Wa_Genp)
    loss += loss_fn(Wb_p, Wb_Genp)
    loss += loss_fn(H_p, H_Genp)

    return [loss]


def energy_loss_fn(out, target, inputs, loss_fn=F.mse_loss):
    """ 
    Computes loss for Higgs, W bosons, and neutrinos' masses.

    input consists of
        La_Visx/y/z, Lb_Visx/y/z, MET_X_Vis, MET_Y_Vis

    out consists of 
        Na_Genx/y/z, Nb_Genz

    target consists of 
        Na_Genx/y/z, Nb_Genx/y/z, Na_GenE,  Nb_GenE, Wa_GenE, Wb_GenE, H_GenE
    """
    # unpack variables
    La_Visp = inputs[:, 0:3]
    Lb_Visp = inputs[:, 3:6]
    MET_X_Vis = inputs[:, 6]
    MET_Y_Vis = inputs[:, 7]

    Na_Genp = target[:, 0:3]
    Nb_Genp = target[:, 3:6]
    Na_GenE = target[:, 6]
    Nb_GenE = target[:, 7]
    Wa_GenE = target[:, 8]
    Wb_GenE = target[:, 9]
    H_GenE = target[:, 10]

    # Compute our outputs
    Na_p = out[:, 0:3]

    Nb_p = torch.zeros_like(Na_p)
    Nb_p[:, 2] = out[:, 3]
    Nb_p[:, 1] = MET_Y_Vis - Na_p[:, 1]
    Nb_p[:, 0] = MET_X_Vis - Na_p[:, 0]

    Na_E = torch.norm(Na_p, dim=1)
    Nb_E = torch.norm(Nb_p, dim=1)
    La_E = torch.norm(La_Visp, dim=1)
    Lb_E = torch.norm(Lb_Visp, dim=1)

    Wa_E = La_E + Na_E
    Wb_E = Lb_E + Nb_E
    H_E = Wa_E + Wb_E

    # Compare to target
    loss = loss_fn(Na_p, Na_Genp)
    loss += loss_fn(Nb_p, Nb_Genp)
    loss += loss_fn(Na_E, Na_GenE)
    loss += loss_fn(Nb_E, Nb_GenE)
    loss += loss_fn(Wa_E, Wa_GenE)
    loss += loss_fn(Wb_E, Wb_GenE)
    loss += loss_fn(H_E, H_GenE)

    return [loss]


def mass_loss_fn(out, target, inputs, loss_fn=F.mse_loss):
    """ 
    Computes loss for Higgs, W bosons, and neutrinos' masses.

    input consists of
        La_Visx/y/z, Lb_Visx/y/z, MET_X_Vis, MET_Y_Vis

    out consists of 
        Na_Genx/y/z, Nb_Genz

    target consists of 
        Na_Genx/y/z, Nb_Genx/y/z,  Wa_Genm, Wb_Genm, H_Genm
    """
    # unpack variables
    La_Visp = inputs[:, 0:3]
    Lb_Visp = inputs[:, 3:6]
    MET_X_Vis = inputs[:, 6]
    MET_Y_Vis = inputs[:, 7]

    Na_Genp = target[:, 0:3]
    Nb_Genp = target[:, 3:6]
    Wa_Genm = target[:, 6]
    Wb_Genm = target[:, 7]
    H_Genm = target[:, 8]

    # Compute our outputs
    Na_p = out[:, 0:3]

    Nb_p = torch.zeros_like(Na_p)
    Nb_p[:, 2] = out[:, 3]
    Nb_p[:, 1] = MET_Y_Vis - Na_p[:, 1]
    Nb_p[:, 0] = MET_X_Vis - Na_p[:, 0]

    Na_E = torch.norm(Na_p, dim=1)
    Nb_E = torch.norm(Nb_p, dim=1)
    La_E = torch.norm(La_Visp, dim=1)
    Lb_E = torch.norm(Lb_Visp, dim=1)

    Wa_E = La_E + Na_E
    Wb_E = Lb_E + Nb_E

    Wa_p = La_Visp + Na_p
    Wb_p = Lb_Visp + Nb_p
    H_p = Wa_p + Wb_p

    Wa_m = (Wa_E**2 - square_norm(Wa_p))**0.5
    Wb_m = (Wb_E**2 - square_norm(Wb_p))**0.5
    H_m = Wa_m + Wb_m

    # Compare to target
    loss = loss_fn(Na_p, Na_Genp)
    loss += loss_fn(Nb_p, Nb_Genp)
    loss += loss_fn(Wa_m, Wa_Genm)
    loss += loss_fn(Wb_m, Wb_Genm)
    loss += loss_fn(H_m, H_Genm)

    return [loss]


def higgs_mass_loss_fn(out, target, inputs, loss_fn=F.mse_loss):
    """ 
    Computes loss for Higgs and neutrinos' masses.

    input consists of
        La_Visx/y/z, Lb_Visx/y/z, MET_X_Vis, MET_Y_Vis

    out consists of 
        Na_Genx/y/z, Nb_Genz

    target consists of 
        Na_Genx/y/z, Nb_Genx/y/z, La_VisE, Lb_VisE, H_Genm_squared
    """
    # unpack variables
    La_Visp = inputs[:, 0:3]
    Lb_Visp = inputs[:, 3:6]
    MET_X_Vis = inputs[:, 6]
    MET_Y_Vis = inputs[:, 7]

    Na_Genp = target[:, 0:3]
    Nb_Genp = target[:, 3:6]
    La_VisE = target[:, 6]
    Lb_VisE = target[:, 7]
    H_Genm_squared = target[:, 8]

    # Compute our outputs
    Na_p = out[:, 0:3]

    Nb_p = torch.zeros_like(Na_p)
    Nb_p[:, 2] = out[:, 3]
    Nb_p[:, 1] = MET_Y_Vis - Na_p[:, 1]
    Nb_p[:, 0] = MET_X_Vis - Na_p[:, 0]

    Wa_p = La_Visp + Na_p
    Wb_p = Lb_Visp + Nb_p

    Na_E = norm(Na_p)
    Nb_E = norm(Nb_p)

    Wa_E = La_VisE + Na_E
    Wb_E = Lb_VisE + Nb_E

    H_p = Wa_p + Wb_p
    H_E = Wa_E + Wb_E
    H_m_squared = H_E**2 - square_norm(H_p)

    return loss_fn(Na_p,
                   Na_Genp), loss_fn(Nb_p,
                                     Nb_Genp), loss_fn(H_m_squared,
                                                       H_Genm_squared)


def norm(x):
    """ x has shape (?, 3). Returns shape (?, 1) """
    return (x[:, 0]**2 + x[:, 1]**2 + x[:, 2]**2)**0.5


def square_norm(x):
    """ x has shape (?, 3). Returns shape (?, 1) """
    return x[:, 0]**2 + x[:, 1]**2 + x[:, 2]**2
