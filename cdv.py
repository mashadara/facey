import io
import logging

import torch
import torch.distributions.constraints as constraints
import pyro
import pyro.distributions as dist
from pyro.infer import SVI, Trace_ELBO
from pyro.optim import Adam


# Cox-de Vries cone model
def dist_cone(x, w, sensitivity):
    delta = x - w
    return torch.sqrt(torch.sum(torch.pow(delta, 2)*sensitivity, -1))


# Logistic link
def prob(x1, x2, xopt, sensitivity):
    dist1 = dist_cone(x1, xopt, sensitivity)
    dist2 = dist_cone(x2, xopt, sensitivity)
    dd = dist2 - dist1
    prob1 = torch.reciprocal(1 + torch.exp(-dd))
    return prob1


# Model and guide for SVI of the Cox-de Vries model
def model(data):
    d = 512
    dtype = data['x1'].dtype
    device = data['x1'].device

    xopt_mu = torch.zeros(d, dtype=dtype, device=device)
    xopt_sd = torch.ones(d, dtype=dtype, device=device)
    sens_alpha = torch.ones(d, dtype=dtype, device=device)
    sens_beta = torch.ones(d, dtype=dtype, device=device)

    xopt = pyro.sample('xopt', dist.Normal(xopt_mu, xopt_sd).to_event(1))
    sensitivity = pyro.sample('sensitivity', dist.Gamma(sens_alpha, sens_beta).to_event(1))

    prob1 = prob(data['x1'], data['x2'], xopt, sensitivity)

    with pyro.plate('obs', data['pref1'].size()[0]) as ind:
        pyro.sample('prefobs', dist.Bernoulli(prob1[ind]), obs=data['pref1'][ind])


def guide(data):
    d = 512
    dtype = data['x1'].dtype
    device = data['x1'].device

    xopt_mu = pyro.param('xopt_mu', torch.zeros(d, dtype=dtype, device=device))
    xopt_sd = pyro.param('xopt_sd', torch.ones(d, dtype=dtype, device=device), constraint=constraints.positive)
    sensitivity_alpha = pyro.param('sensitivity_alpha', torch.ones(d, dtype=dtype, device=device), constraint=constraints.positive)
    sensitivity_beta = pyro.param('sensitivity_beta', torch.ones(d, dtype=dtype, device=device), constraint=constraints.positive)
    sensitivity = pyro.sample('sensitivity', dist.Gamma(sensitivity_alpha, sensitivity_beta).to_event(1))
    xopt = pyro.sample('xopt', dist.Normal(xopt_mu, xopt_sd).to_event(1))
    return xopt


def posterior_volume(xopt_sd):
    # As the CdV model samples have zero off-diagonal covariance, we can
    # estimate volume relative to the prior by a straightforward product.
    # This gives the "1 SD" volume of the posterior relative to an isotropic
    # 512-dim normal with SD=1.
    return torch.sum(torch.log(xopt_sd)).item()


def learn(paramstore_in, latents_in, comparisons_in, svi_iters, lr, beta1, beta2, seed, cpuonly):
    if paramstore_in != None:
        buf = io.BytesIO(paramstore_in)
        pyro.get_param_store().set_state(torch.load(buf))
    else:
        pyro.get_param_store().clear()

    # Note: seed is currently ignored.  Setting this => NaN results.  Why???
    # torch.manual_seed(seed)

    logging.info('Moving to pytorch')
    x1 = torch.empty(len(comparisons_in), 512)
    x2 = torch.empty(len(comparisons_in), 512)
    pref1 = torch.empty(len(comparisons_in))
    i = 0
    for good_id, bad_id, conf in comparisons_in:
        x1[i,:] = torch.tensor(latents_in[good_id])
        x2[i,:] = torch.tensor(latents_in[bad_id])
        pref1[i] = 1.
        i += 1
        if i % 10000 == 0:
            logging.info('{}/{}'.format(i, len(comparisons_in)))

    if not cpuonly and torch.cuda.is_available():
        logging.info('Moving to GPU')
        x1 = x1.cuda()
        x2 = x2.cuda()
        pref1 = pref1.cuda()

    logging.info('Optimising')
    data = {'x1': x1, 'x2': x2, 'pref1': pref1}
    adam_params = {"lr": lr, "betas": (beta1, beta2)}
    optimizer = Adam(adam_params)
    svi = SVI(model, guide, optimizer, loss=Trace_ELBO())
    for step in range(svi_iters):
        loss = svi.step(data)
        if (step + 1) % 50 == 0:
            logging.info('{}/{} {}'.format(step + 1, svi_iters, posterior_volume(pyro.param('xopt_sd'))))

    buf = io.BytesIO()
    torch.save(pyro.get_param_store().get_state(), buf)
    return buf.getvalue()


def sample(paramstore_in, n, seed, cpuonly):
    if not isinstance(paramstore_in, type(None)):
        buf = io.BytesIO(paramstore_in)
        pyro.get_param_store().set_state(torch.load(buf))
    else:
        pyro.get_param_store().clear()

    torch.manual_seed(seed)

    # Generate empty data to keep Pyro happy
    x1 = torch.empty(0, 512)
    x2 = torch.empty(0, 512)
    pref1 = torch.empty(0)

    if not cpuonly and torch.cuda.is_available():
        logging.info('Moving to GPU')
        x1 = x1.cuda()
        x2 = x2.cuda()
        pref1 = pref1.cuda()

    logging.info('Sampling')
    data = {'x1': x1, 'x2': x2, 'pref1': pref1}
    sampx = torch.empty(n, 512, dtype=x1.dtype, device=x1.device)
    for i in range(n):
        sampx[i,:] = guide(data).detach()

    return sampx.tolist()

#
# def analogy(sl, sm, dm, n, stochastic, seed, gpu):
#     assert (n > 1 and stochastic == True) or (n == 1 and stochastic == False)
#     if stochastic == False:
#         return analogy_fixed(sl, sm, dm)
#     else:
#         return analogy_sampled(sl, sm, dm, n, seed, gpu)
#
#
# def analogy_fixed(sl, sm, dm):
#     assert not isinstance(sm, type(None))
#     assert not isinstance(dm, type(None))
#
#     bakbuf = io.BytesIO()
#     torch.save(pyro.get_param_store().get_state(), bakbuf)
#
#     buf = io.BytesIO(sm)
#     pyro.get_param_store().set_state(torch.load(buf))
#     source_xhat_mu = pyro.param('xopt_mu')
#     source_xhat_sd = pyro.param('xopt_sd')
#
#     buf = io.BytesIO(dm)
#     pyro.get_param_store().set_state(torch.load(buf))
#     dest_xhat_mu = pyro.param('xopt_mu')
#     dest_xhat_sd = pyro.param('xopt_sd')
#
#     pyro.get_param_store().set_state(torch.load(bakbuf))
#
#     sl = torch.tensor(sl)
#     dl = sl - sm + dm
#     return dl.tolist()
#
#
# def analogy_sampled(sl, sm, dm, n, seed, gpu):
#     source_samples = torch.tensor(sample(sm, n, seed, gpu))
#     dest_samples = torch.tensor(sample(dm, n, seed, gpu))
#     sl = torch.tensor(sl)
#     result = sl - source_samples + dest_samples
#     return result.tolist()
