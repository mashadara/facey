'''Generate faces.

Usage:
  facetool.py face.addthumbnail [--cpuonly] [--seed=<n>] <input.face>...
  facetool.py face.getthumbnail [--outdir=<path>] <input.face>...
  facetool.py face.interpolate [--cpuonly] [--outdir=<path>] (w|x) <source.face> <dest.face> <n>
  facetool.py face.interpolate.x [--nothumb] [--seed=<n>] <source.face> <dest.face> <n> <output.facelist>
  facetool.py face.analogy <source.face> <source.model> <dest.model> <dest.face>
  facetool.py facelist.addthumbnails [--cpuonly] [--seed=<n>] <input.facelist>
  facetool.py facelist.getthumbnails [--outdir=<path>] <input.facelist>
  facetool.py model.sample [--model=<path>] [--nothumb] [--psi=<f>] [--seed=<n>] <n> <output.facelist>
  facetool.py search.additeration [--beta1=<f>] [--beta2=<f>] [--cpuonly] [--lr=<f>] [--niter=<n>] [--seed=<n>] <progress.search>
  facetool.py -h | --help
  facetool.py --version

Options:
  -h --help        Show this screen.
  --version        Show version.
  --beta1=<f>      ADAM beta1 parameter [default: 0.9].
  --beta2=<f>      ADAM beta1 parameter [default: 0.999].
  --cpuonly        Do not use GPU (ie use CPU only). Default is to use GPU if available.
  --lr=<f>         ADAM learning rate [default: 0.005].
  --model=<path>   Model to use for face generation. If missing, generates from the prior.
  --niter=<n>      Number of SVI iterations [default: 500].
  --nothumb        Do not add thumbnails to the generated faces. Default is to automatically generate thumbnails.
  --outdir=<path>  Save images to this output directory [default: .].
  --psi=<f>        Truncation psi [default: 0.7].
  --seed=<n>       Seed for PRNG [default: 27182818].
'''
import logging
import os
import os.path
import sys
import uuid

import docopt
import h5py
import PIL.Image

import cdv
import data
import generator
import latentmath



def genthumbnail(face, seed, cpuonly):
    pil_thumbnail = generator.latent_to_image([face.latents], seed=seed, psi=face.psi, cpuonly=cpuonly, downsample=4)[0]
    return data.Image.fromPIL(pil_thumbnail, seed)


def do_face_addthumbnail(args):
    seed = int(args['--seed'])
    cpuonly = args['--cpuonly']

    for face_path in args['<input.face>']:
        face_basename = os.path.basename(face_path)
        if not face_basename.endswith('.face'):
            logging.error("Input file {} has incorrect extension. All input files must have ending .face".format(face_path))
            sys.exit(1)

        with h5py.File(face_path, 'r') as infile:
            face = data.Face.fromhdf5(infile)
        face.thumbnail = genthumbnail(face, seed, cpuonly)
        with h5py.File(face_path, 'w') as outfile:
            face.tohdf5(outfile)


def do_face_getthumbnail(args):
    for face_path in args['<input.face>']:
        face_basename = os.path.basename(face_path)
        if not face_basename.endswith('.face'):
            logging.error("Input file {} has incorrect extension. All input files must have ending .face".format(face_path))
            sys.exit(1)
        img_path = os.path.join(args['--outdir'], face_basename[:-5] + '.jpg')
        if os.path.exists(img_path):
            logging.error('Destination file {} already exists; exiting.'.format(img_path))
            sys.exit(3)

        with h5py.File(face_path, 'r') as infile:
            face = data.Face.fromhdf5(infile)
        if face.thumbnail == None or face.thumbnail.width == 0:
            logging.warning('No thumbnail present for file {}, skipping'.format(face_path))
        else:
            face.thumbnail.toPIL().save(img_path)


def do_face_interpolate(args):
    n = int(args['<n>'])
    cpuonly = args['--cpuonly']
    mode_w = args['w']
    outdir = args['--outdir']

    with h5py.File(args['<source.face>'], 'r') as sourcefile, h5py.File(args['<dest.face>'], 'r') as destfile:
        source_face = data.Face.fromhdf5(sourcefile)
        dest_face = data.Face.fromhdf5(destfile)

    source_latents = source_face.latents
    dest_latents = dest_face.latents

    if mode_w:
        source_w = generator.latent_to_w(source_latents, source_face.psi, cpuonly)
        dest_w = generator.latent_to_w(dest_latents, dest_face.psi, cpuonly)
        interp_w = latentmath.interpolate_w(source_w, dest_w, n)
    else:
        interp_x = latentmath.interpolate_x(source_latents, dest_latents, n)
        interp_psi = (dest_face.psi-source_face.psi)*range(n)/float(n-1) + source_face.psi
        interp_w = [generator.latent_to_w(x, psi, cpuonly) for x, psi in zip(interp_x, interp_psi)]

    interp_imgs = [generator.w_to_image(w, 1, cpuonly)[0] for w in interp_w]
    for i, interp_image in enumerate(interp_imgs):
        interp_image.save(os.path.join(outdir, 'interp_{}_{}_{}_{:02}_{:.03}.jpg'.format(
            os.path.basename(args['<source.face>'][:-5]),
            os.path.basename(args['<dest.face>'][:-5]), 'w' if mode_w else 'x',
            i, float(i)/(n-1))))


def do_face_interpolate_x(args):
    n = int(args['<n>'])
    seed = int(args['--seed'])
    psi = float(args['--psi'])
    nothumb = args['--nothumb']
    cpuonly = args['--cpuonly']

    with h5py.File(args['<source.face>'], 'r') as sourcefile, h5py.File(args['<dest.face>'], 'r') as destfile:
        source_face = data.Face.fromhdf5(sourcefile)
        dest_face = data.Face.fromhdf5(destfile)

    interp_latents = latentmath.interpolate_x(source_face.latents, dest_face.latents, n)

    # Output latents, optionally creating thumbnails
    faces = []
    for i, x in enumerate([source_face.latents] + interp_latents + [dest_face.latents]):
        face = data.Face(name='interp_{}_{}_{:.3}'.format(source_face.name, dest_face.name, float(i)/(n-1)), latents=x, psi=psi)
        if args['--nothumb'] == False:
            face.thumbnail = genthumbnail(face, seed, cpuonly)
        faces.append(face)

    facelist = data.FaceList(faces)
    with h5py.File(args['<output.facelist>'], 'w') as outfile:
        facelist.tohdf5(outfile)


# def do_face_analogy(args):
#     seed = int(args['--seed'])
#     psi = float(args['--psi'])
#     randnoise = not args['--detnoise']
#
#     with h5py.File(args['<source.face>'], 'r') as sourceface:
#         source_face = data.Face.fromhdf5(sourceface)
#     with h5py.File(args['<source.model>'], 'r') as sourcemodel:
#         source_model = data.Face.fromhdf5(sourcemodel)
#     with h5py.File(args['<dest.model>'], 'r') as destmodel:
#         dest_model = data.Face.fromhdf5(destmodel)
#
#     dest_latents = cdv.analogy(sl=source_face.latents,
#         sm=source_model.pyroparams.paramstore,
#         dm=dest_model.pyroparams.paramstore, n=1, stochastic=False, seed=None,
#         gpu=False)
#
#     face = data.Face(latents=dest_latents)
#     if args['--nothumb'] == False:
#         face.thumbnail = genthumbnail(face, seed)
#     with h5py.File(args['<dest.face>'], 'w') as outfile:
#         face.tohdf5(outfile)


def do_facelist_addthumbnails(args):
    seed = int(args['--seed'])
    cpuonly = args['--cpuonly']

    if not args['<input.facelist>'].endswith('.facelist'):
        logging.error("Input file {} has incorrect extension. Expected .facelist".format(args['<input.facelist>']))
        sys.exit(1)

    with h5py.File(args['<input.facelist>'], 'r') as infile:
        facelist = data.FaceList.fromhdf5(infile)
    for face in facelist.faces:
        face.thumbnail = genthumbnail(face, seed, cpuonly)

    with h5py.File(args['<input.facelist>'], 'w') as destfile:
        facelist.tohdf5(destfile)


def do_facelist_getthumbnails(args):
    if not args['<input.facelist>'].endswith('.facelist'):
        logging.error("Input file {} has incorrect extension. Expected .facelist".format(args['<input.facelist>']))
        sys.exit(1)

    with h5py.File(args['<input.facelist>'], 'r') as infile:
        facelist = data.FaceList.fromhdf5(infile)

    for i, face in enumerate(facelist.faces, 1):
        img_path = os.path.join(args['--outdir'], '{:04}.jpg'.format(i))
        if face.thumbnail == None or face.thumbnail.width == 0:
            logging.warning('No thumbnail present for file {}, skipping'.format(face_path))
        else:
            face.thumbnail.toPIL().save(img_path)


def do_model_sample(args):
    n = int(args['<n>'])
    seed = int(args['--seed'])
    psi = float(args['--psi'])
    cpuonly = True      # Force to CPU only due to PRNG inconsistency issue
                        # between CPU and GPU

    if args['--model'] != None:
        logging.info('Loading model...')
        with h5py.File(args['--model'], 'r') as modelfile:
            model = data.Model.fromhdf5(modelfile)
        paramstore_in = model.pyroparams.paramstore
        if paramstore_in == b'':
            logging.info('Empty model encountered; falling back to prior')
            paramstore_in = None
        model_name = model.name
    else:
        paramstore_in = None
        model_name = 'PRIOR'

    latent_vectors = cdv.sample(paramstore_in, n, seed, cpuonly)

    faces = []
    for x in latent_vectors:
        face = data.Face(name='sample_{}_{}'.format(model_name, str(uuid.uuid4())[:8]), latents=x, psi=psi)
        if args['--nothumb'] == False:
            face.thumbnail = genthumbnail(face, seed)
        faces.append(face)

    facelist = data.FaceList(faces)
    with h5py.File(args['<output.facelist>'], 'w') as destfile:
        facelist.tohdf5(destfile)


def search_to_comparisons(search):
    # Construct a dictionary of latents, and a list of comparison tuples,
    # that encode all InOut data for all search iterations.
    latents = {}
    comparisons = []
    latent_counter = 0      # A globally unique latent ID
    for i, iter in enumerate(search.iterations.iterations):
        assert iter.method == 'SVI_ACCUM_ADAM_InOut'
        this_round_latent_keys = []     # Mapping from round latent indices to the globally unique ID
        for face in iter.input.faces.faces:
            latents[latent_counter] = face.latents
            this_round_latent_keys.append(latent_counter)
            latent_counter += 1
        for in_index in iter.input.in_indices:
            for out_index in iter.input.out_indices:
                comparisons.append((this_round_latent_keys[in_index], this_round_latent_keys[out_index], 1.))
    return (latents, comparisons)


def do_search_additeration(args):
    lr = float(args['--lr'])
    beta1 = float(args['--beta1'])
    beta2 = float(args['--beta2'])
    niter = int(args['--niter'])
    seed = int(args['--seed'])
    cpuonly = args['--cpuonly']

    # Perform a new iteration of search, adding it as a SearchIteration to the
    # SearchIterationList.
    with h5py.File(args['<progress.search>'], 'r') as infile:
        search = data.Search.fromhdf5(infile)

    # Get all latents and comparisons for all iterations of the search
    latents, comparisons = search_to_comparisons(search)

    # Get the starting 'input' parameter store.  Fallback to None (ie prior)
    # if necessary
    if len(search.iterations.iterations) > 0:
        paramstore = search.iterations.iterations[-1].in_params.paramstore
        if paramstore == b'':
            logging.info('Empty model encountered; falling back to prior')
            paramstore = None
    else:
        paramstore = None       # Nothing yet learned, fall back to the prior

    # Do the learning
    newparams = cdv.learn(paramstore, latents, comparisons, niter, lr, beta1, beta2, seed, cpuonly)

    # Add these learned parameters to the latest SearchIteration.  Update other
    # metadata fields as well.
    search.iterations.iterations[-1].out_params = data.PyroParams(paramstore=newparams)
    search.iterations.iterations[-1].input.lr = lr
    search.iterations.iterations[-1].input.beta1 = beta1
    search.iterations.iterations[-1].input.beta2 = beta2
    search.iterations.iterations[-1].input.niter = niter
    search.iterations.iterations[-1].input.seed = seed

    # Add a new SearchIteration stub.
    newinput = data.SearchIterationData_SVI_ACCUM_ADAM_InOut(lr=lr, beta1=beta1,
        beta2=beta2, niter=niter, seed=seed)
    newiter = data.SearchIteration(method='SVI_ACCUM_ADAM_InOut', input=newinput, in_params=data.PyroParams(paramstore=newparams))
    search.iterations.iterations.append(newiter)

    with h5py.File(args['<progress.search>'], 'w') as outfile:
        search.tohdf5(outfile)


if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO)
    args = docopt.docopt(__doc__, version='0.0.2')

    if args['face.addthumbnail']:
        do_face_addthumbnail(args)
    if args['face.getthumbnail']:
        do_face_getthumbnail(args)
    elif args['face.interpolate']:
        do_face_interpolate(args)
    elif args['face.interpolate.x']:
        do_face_interpolate_x(args)
    elif args['facelist.addthumbnails']:
        do_facelist_addthumbnails(args)
    elif args['facelist.getthumbnails']:
        do_facelist_getthumbnails(args)
    elif args['model.sample']:
        do_model_sample(args)
    elif args['search.additeration']:
        do_search_additeration(args)
    else:
        logging.warning('Not implemented')
