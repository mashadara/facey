import torch


def interpolate_x(source, dest, n):
    # Interpolation on x vectors.  Returns result as a list of 512-vectors.
    source = torch.tensor(source)
    dest = torch.tensor(dest)

    results = []
    source_normed = source / source.norm()
    dest_normed = dest / dest.norm()
    dotprod = (source_normed*dest_normed).sum()
    if dotprod > 0.9995:
        # Fallback to lerp
        delta = dest - source
        for i in range(n):
            results.append((source + delta*float(i)/(n-1)).tolist())
    else:
        # slerp
        omega = torch.acos(dotprod)
        sin_omega = torch.sin(omega)

        for i in range(n):
            f = float(i)/(n-1)
            x = (torch.sin((1.0-f)*omega)/sin_omega)*source + (torch.sin(f*omega)/sin_omega)*dest
            results.append(x.tolist())

    return results


def interpolate_w(source, dest, n):
    # Interpolation on w matrices.
    source = torch.tensor(source)
    dest = torch.tensor(dest)

    delta = dest - source
    results = [(source + delta*float(i)/(n-1)).tolist() for i in range(n)]
    return results
